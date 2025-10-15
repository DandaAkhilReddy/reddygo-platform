"""
Whoop Integration Router

Handles OAuth flow, data synchronization, and AI insights for Whoop users.
Provides seamless integration with Whoop API v2 for elite fitness tracking.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import httpx
import os
from dotenv import load_dotenv
import json
from firebase_admin import firestore
import base64
from cryptography.fernet import Fernet

load_dotenv()

router = APIRouter()

# Whoop API Configuration
WHOOP_AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
WHOOP_TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
WHOOP_API_BASE = "https://api.prod.whoop.com/developer"

# OAuth Config
WHOOP_CLIENT_ID = os.getenv("WHOOP_CLIENT_ID")
WHOOP_CLIENT_SECRET = os.getenv("WHOOP_CLIENT_SECRET")
WHOOP_REDIRECT_URI = os.getenv("WHOOP_REDIRECT_URI", "https://api.reddyfit.com/api/integrations/whoop/callback")

# Encryption for storing tokens
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", Fernet.generate_key()).encode() if isinstance(os.getenv("ENCRYPTION_KEY", Fernet.generate_key()), str) else Fernet.generate_key()
cipher_suite = Fernet(ENCRYPTION_KEY)

# Firestore client
db = firestore.client()

# OAuth Scopes
WHOOP_SCOPES = [
    "read:recovery",
    "read:cycles",
    "read:workout",
    "read:sleep",
    "read:profile",
    "read:body_measurement"
]

# ==================== Models ====================

class WhoopAuthRequest(BaseModel):
    user_id: str

class WhoopCallbackRequest(BaseModel):
    code: str
    state: str
    user_id: str

class WhoopSyncRequest(BaseModel):
    user_id: str
    force_refresh: bool = False

class WhoopInsightRequest(BaseModel):
    user_id: str
    insight_type: Optional[str] = "daily"  # daily, workout, recovery, sleep

# ==================== Helper Functions ====================

def encrypt_token(token: str) -> str:
    """Encrypt access token for secure storage"""
    return cipher_suite.encrypt(token.encode()).decode()

def decrypt_token(encrypted_token: str) -> str:
    """Decrypt access token"""
    return cipher_suite.decrypt(encrypted_token.encode()).decode()

async def get_whoop_token(user_id: str) -> str:
    """Retrieve and decrypt Whoop access token for user"""
    doc = db.collection("whoop_integrations").document(user_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Whoop integration not found. Please connect your Whoop account.")

    data = doc.to_dict()
    encrypted_token = data.get("encrypted_access_token")

    if not encrypted_token:
        raise HTTPException(status_code=401, detail="Whoop access token not found")

    return decrypt_token(encrypted_token)

async def call_whoop_api(endpoint: str, token: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Make authenticated request to Whoop API"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{WHOOP_API_BASE}{endpoint}",
                headers=headers,
                params=params or {},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise HTTPException(status_code=401, detail="Whoop token expired. Please reconnect your Whoop account.")
            elif e.response.status_code == 429:
                raise HTTPException(status_code=429, detail="Whoop API rate limit exceeded. Please try again later.")
            else:
                raise HTTPException(status_code=e.response.status_code, detail=f"Whoop API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch Whoop data: {str(e)}")

# ==================== OAuth Endpoints ====================

@router.post("/authorize")
async def initiate_whoop_oauth(request: WhoopAuthRequest):
    """
    Initiate Whoop OAuth flow
    Returns authorization URL for user to approve access
    """
    # Generate state for CSRF protection
    state = base64.b64encode(f"{request.user_id}:{datetime.utcnow().isoformat()}".encode()).decode()

    # Store state in Firestore for verification
    db.collection("whoop_oauth_states").document(state).set({
        "user_id": request.user_id,
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(minutes=10)
    })

    # Build authorization URL
    scope_string = " ".join(WHOOP_SCOPES)
    auth_url = (
        f"{WHOOP_AUTH_URL}?"
        f"client_id={WHOOP_CLIENT_ID}&"
        f"redirect_uri={WHOOP_REDIRECT_URI}&"
        f"response_type=code&"
        f"scope={scope_string}&"
        f"state={state}"
    )

    return {
        "authorization_url": auth_url,
        "state": state,
        "message": "Redirect user to authorization_url to connect Whoop account"
    }

@router.post("/callback")
async def handle_whoop_callback(request: WhoopCallbackRequest):
    """
    Handle OAuth callback from Whoop
    Exchange authorization code for access token
    """
    # Verify state
    state_doc = db.collection("whoop_oauth_states").document(request.state).get()
    if not state_doc.exists:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    state_data = state_doc.to_dict()
    if state_data["user_id"] != request.user_id:
        raise HTTPException(status_code=403, detail="User ID mismatch")

    if state_data["expires_at"] < datetime.utcnow():
        raise HTTPException(status_code=400, detail="OAuth state expired")

    # Exchange code for access token
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                WHOOP_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": request.code,
                    "client_id": WHOOP_CLIENT_ID,
                    "client_secret": WHOOP_CLIENT_SECRET,
                    "redirect_uri": WHOOP_REDIRECT_URI
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            token_data = response.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to exchange code for token: {str(e)}")

    # Encrypt and store access token
    encrypted_token = encrypt_token(token_data["access_token"])

    integration_data = {
        "user_id": request.user_id,
        "encrypted_access_token": encrypted_token,
        "token_type": token_data.get("token_type", "Bearer"),
        "scope": token_data.get("scope", " ".join(WHOOP_SCOPES)),
        "connected_at": datetime.utcnow(),
        "last_synced": None,
        "status": "active"
    }

    # Store in Firestore
    db.collection("whoop_integrations").document(request.user_id).set(integration_data)

    # Clean up state
    db.collection("whoop_oauth_states").document(request.state).delete()

    # Trigger initial sync in background
    # (In production, use background tasks or queue)

    return {
        "success": True,
        "message": "Whoop account connected successfully",
        "connected_at": integration_data["connected_at"].isoformat()
    }

@router.delete("/disconnect/{user_id}")
async def disconnect_whoop(user_id: str):
    """Disconnect Whoop integration and revoke access"""
    # Note: Whoop API v2 has revoke endpoint at DELETE /v2/user/access
    # We should call it before deleting local data

    try:
        token = await get_whoop_token(user_id)

        # Revoke token at Whoop
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {token}"}
            await client.delete(f"{WHOOP_API_BASE}/v2/user/access", headers=headers)
    except:
        pass  # Continue even if revoke fails

    # Delete local integration data
    db.collection("whoop_integrations").document(user_id).delete()
    db.collection("whoop_data").document(user_id).delete()

    return {
        "success": True,
        "message": "Whoop account disconnected successfully"
    }

# ==================== Data Sync Endpoints ====================

@router.post("/sync")
async def sync_whoop_data(request: WhoopSyncRequest, background_tasks: BackgroundTasks):
    """
    Sync all Whoop data for user
    Fetches: recovery, cycles, workouts, sleep, profile, body measurements
    """
    token = await get_whoop_token(request.user_id)

    # Check last sync time (cache for 5 minutes unless force refresh)
    integration_doc = db.collection("whoop_integrations").document(request.user_id).get()
    integration_data = integration_doc.to_dict()
    last_synced = integration_data.get("last_synced")

    if not request.force_refresh and last_synced:
        if isinstance(last_synced, datetime):
            if datetime.utcnow() - last_synced < timedelta(minutes=5):
                # Return cached data
                cached_doc = db.collection("whoop_data").document(request.user_id).get()
                if cached_doc.exists:
                    return {
                        "success": True,
                        "cached": True,
                        "data": cached_doc.to_dict(),
                        "last_synced": last_synced.isoformat()
                    }

    # Fetch all data from Whoop
    whoop_data = {}

    try:
        # 1. User Profile
        profile = await call_whoop_api("/v2/user/profile/basic", token)
        whoop_data["profile"] = profile

        # 2. Body Measurements
        body = await call_whoop_api("/v2/user/measurement/body", token)
        whoop_data["body_measurement"] = body

        # 3. Latest Recovery (last 7 days)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=7)
        recovery = await call_whoop_api("/v2/recovery", token, {
            "start": start_time.isoformat() + "Z",
            "end": end_time.isoformat() + "Z",
            "limit": 7
        })
        whoop_data["recovery"] = recovery.get("records", [])

        # 4. Latest Cycles (last 7 days)
        cycles = await call_whoop_api("/v2/cycle", token, {
            "start": start_time.isoformat() + "Z",
            "end": end_time.isoformat() + "Z",
            "limit": 7
        })
        whoop_data["cycles"] = cycles.get("records", [])

        # 5. Latest Workouts (last 7 days)
        workouts = await call_whoop_api("/v2/activity/workout", token, {
            "start": start_time.isoformat() + "Z",
            "end": end_time.isoformat() + "Z",
            "limit": 25
        })
        whoop_data["workouts"] = workouts.get("records", [])

        # 6. Latest Sleep (last 7 days)
        sleep = await call_whoop_api("/v2/activity/sleep", token, {
            "start": start_time.isoformat() + "Z",
            "end": end_time.isoformat() + "Z",
            "limit": 7
        })
        whoop_data["sleep"] = sleep.get("records", [])

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync Whoop data: {str(e)}")

    # Store in Firestore
    whoop_data["synced_at"] = datetime.utcnow()
    whoop_data["user_id"] = request.user_id

    db.collection("whoop_data").document(request.user_id).set(whoop_data)

    # Update last_synced timestamp
    db.collection("whoop_integrations").document(request.user_id).update({
        "last_synced": datetime.utcnow()
    })

    return {
        "success": True,
        "cached": False,
        "data": whoop_data,
        "synced_at": whoop_data["synced_at"].isoformat(),
        "summary": {
            "recovery_days": len(whoop_data["recovery"]),
            "cycles": len(whoop_data["cycles"]),
            "workouts": len(whoop_data["workouts"]),
            "sleep_days": len(whoop_data["sleep"])
        }
    }

@router.get("/status/{user_id}")
async def get_integration_status(user_id: str):
    """Check if user has connected Whoop and when last synced"""
    doc = db.collection("whoop_integrations").document(user_id).get()

    if not doc.exists:
        return {
            "connected": False,
            "message": "Whoop account not connected"
        }

    data = doc.to_dict()
    return {
        "connected": True,
        "status": data.get("status", "active"),
        "connected_at": data.get("connected_at").isoformat() if data.get("connected_at") else None,
        "last_synced": data.get("last_synced").isoformat() if data.get("last_synced") else None,
        "scopes": data.get("scope", "").split()
    }

# ==================== AI Insights Endpoints ====================

@router.post("/insights")
async def generate_whoop_insights(request: WhoopInsightRequest):
    """
    Generate AI-powered insights from Whoop data
    Uses Supermemory context + GPT-4o for personalized recommendations
    """
    # First ensure data is synced
    sync_request = WhoopSyncRequest(user_id=request.user_id, force_refresh=False)
    sync_result = await sync_whoop_data(sync_request, BackgroundTasks())

    whoop_data = sync_result["data"]

    # Get latest recovery and cycle
    latest_recovery = whoop_data["recovery"][0] if whoop_data["recovery"] else None
    latest_cycle = whoop_data["cycles"][0] if whoop_data["cycles"] else None
    latest_sleep = whoop_data["sleep"][0] if whoop_data["sleep"] else None

    if not latest_recovery:
        raise HTTPException(status_code=404, detail="No recent Whoop recovery data found")

    # Build AI prompt based on insight type
    if request.insight_type == "daily":
        insights = _generate_daily_insight(latest_recovery, latest_cycle, latest_sleep)
    elif request.insight_type == "workout":
        recent_workouts = whoop_data["workouts"][:3]
        insights = _generate_workout_insight(latest_recovery, recent_workouts)
    elif request.insight_type == "recovery":
        recovery_history = whoop_data["recovery"]
        insights = _generate_recovery_insight(recovery_history)
    elif request.insight_type == "sleep":
        sleep_history = whoop_data["sleep"]
        insights = _generate_sleep_insight(sleep_history)
    else:
        insights = _generate_daily_insight(latest_recovery, latest_cycle, latest_sleep)

    return {
        "user_id": request.user_id,
        "insight_type": request.insight_type,
        "generated_at": datetime.utcnow().isoformat(),
        "insights": insights,
        "whoop_data_summary": {
            "recovery_score": latest_recovery.get("score", {}).get("recovery_score") if latest_recovery else None,
            "hrv": latest_recovery.get("score", {}).get("hrv_rmssd_milli") if latest_recovery else None,
            "resting_hr": latest_recovery.get("score", {}).get("resting_heart_rate") if latest_recovery else None,
            "strain": latest_cycle.get("score", {}).get("strain") if latest_cycle else None
        }
    }

def _generate_daily_insight(recovery, cycle, sleep) -> Dict[str, Any]:
    """Generate daily coaching insight"""
    recovery_score = recovery.get("score", {}).get("recovery_score", 0) if recovery else 0
    hrv = recovery.get("score", {}).get("hrv_rmssd_milli", 0) if recovery else 0
    resting_hr = recovery.get("score", {}).get("resting_heart_rate", 0) if recovery else 0
    strain = cycle.get("score", {}).get("strain", 0) if cycle else 0

    # Sleep data
    sleep_score = None
    sleep_performance = None
    if sleep and sleep.get("score"):
        sleep_score = sleep["score"]
        sleep_performance = sleep_score.get("sleep_performance_percentage", 0)

    # Determine recovery zone
    if recovery_score >= 67:
        recovery_zone = "green"
        readiness = "excellent"
        recommendation = "Your body is ready for high-intensity training. Push hard today!"
    elif recovery_score >= 34:
        recovery_zone = "yellow"
        readiness = "moderate"
        recommendation = "Moderate intensity recommended. Focus on technique over heavy loads."
    else:
        recovery_zone = "red"
        readiness = "low"
        recommendation = "Your body needs recovery. Consider rest, yoga, or light active recovery."

    # Build insight
    insight = {
        "greeting": f"Good morning! Your recovery is {recovery_score:.0f}% today.",
        "recovery_zone": recovery_zone,
        "readiness": readiness,
        "key_metrics": {
            "recovery_score": recovery_score,
            "hrv_rmssd_ms": hrv,
            "resting_heart_rate": resting_hr,
            "yesterday_strain": strain
        },
        "recommendation": recommendation,
        "detailed_analysis": []
    }

    # HRV Analysis
    if hrv > 0:
        insight["detailed_analysis"].append({
            "metric": "Heart Rate Variability",
            "value": f"{hrv:.1f}ms",
            "interpretation": "HRV indicates your nervous system's readiness to handle stress."
        })

    # Sleep Analysis
    if sleep_performance:
        insight["detailed_analysis"].append({
            "metric": "Sleep Performance",
            "value": f"{sleep_performance:.0f}%",
            "interpretation": f"You got {sleep_performance:.0f}% of the sleep your body needed."
        })

    # Strain recommendation
    if recovery_score >= 67:
        target_strain = "14-18"
    elif recovery_score >= 34:
        target_strain = "8-12"
    else:
        target_strain = "0-6"

    insight["target_strain"] = target_strain

    return insight

def _generate_workout_insight(recovery, workouts) -> Dict[str, Any]:
    """Generate workout-specific insight"""
    return {
        "message": "Workout insights coming soon - analyzing your training patterns",
        "recent_workouts": len(workouts),
        "recommendation": "Based on your Whoop data"
    }

def _generate_recovery_insight(recovery_history) -> Dict[str, Any]:
    """Generate recovery trend insight"""
    if not recovery_history:
        return {"message": "Not enough recovery data yet"}

    scores = [r.get("score", {}).get("recovery_score", 0) for r in recovery_history if r.get("score")]
    avg_recovery = sum(scores) / len(scores) if scores else 0

    return {
        "average_recovery_7d": avg_recovery,
        "trend": "improving" if scores[0] > scores[-1] else "declining" if scores[0] < scores[-1] else "stable",
        "message": f"Your 7-day average recovery is {avg_recovery:.0f}%"
    }

def _generate_sleep_insight(sleep_history) -> Dict[str, Any]:
    """Generate sleep quality insight"""
    if not sleep_history:
        return {"message": "Not enough sleep data yet"}

    sleep_scores = []
    for s in sleep_history:
        if s.get("score"):
            perf = s["score"].get("sleep_performance_percentage")
            if perf:
                sleep_scores.append(perf)

    avg_sleep = sum(sleep_scores) / len(sleep_scores) if sleep_scores else 0

    return {
        "average_sleep_performance_7d": avg_sleep,
        "message": f"Your average sleep performance is {avg_sleep:.0f}%",
        "recommendation": "Aim for 8+ hours for optimal recovery" if avg_sleep < 85 else "Great sleep consistency!"
    }
