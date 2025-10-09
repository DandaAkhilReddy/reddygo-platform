"""
ReddyGo Referrals Router

Two-sided referral system with rewards for both referrer and referee.
Tracks referral conversions, revenue sharing, and provides referrer leaderboards.

All endpoints require Firebase authentication via Bearer token in Authorization header.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from firebase_client import get_firestore_client
from realtime_db_client import get_realtime_db
from firebase_admin import firestore
from auth import get_current_user
import secrets
import string

router = APIRouter()


# ============================================================================
# Constants
# ============================================================================

# Referral rewards
REFERRAL_REWARDS = {
    "referee_signup": 50,           # Points for new user signing up
    "referee_first_challenge": 100,  # Points for completing first challenge
    "referrer_signup": 75,           # Points for referrer when someone signs up
    "referrer_conversion": 200       # Points for referrer when referee completes first challenge
}

# Revenue sharing (10% of premium community revenue goes to referrer)
REFERRAL_REVENUE_SHARE = 0.10

# Referral tier system based on total conversions
REFERRAL_TIERS = {
    1: {"min_conversions": 0, "name": "Starter", "bonus_multiplier": 1.0},
    2: {"min_conversions": 10, "name": "Bronze", "bonus_multiplier": 1.1},
    3: {"min_conversions": 25, "name": "Silver", "bonus_multiplier": 1.2},
    4: {"min_conversions": 50, "name": "Gold", "bonus_multiplier": 1.3},
    5: {"min_conversions": 100, "name": "Platinum", "bonus_multiplier": 1.5},
    6: {"min_conversions": 250, "name": "Diamond", "bonus_multiplier": 2.0}
}


# ============================================================================
# Pydantic Models
# ============================================================================

class ReferralCodeResponse(BaseModel):
    """Referral code information."""
    code: str
    user_id: str
    created_at: str
    total_signups: int
    total_conversions: int
    total_points_earned: int
    total_revenue_earned: float
    current_tier: str
    bonus_multiplier: float


class ApplyReferralCodeRequest(BaseModel):
    """Apply referral code during signup."""
    referral_code: str = Field(..., min_length=6, max_length=12)


class ReferralStatsResponse(BaseModel):
    """User's referral statistics."""
    referral_code: str
    total_signups: int
    total_conversions: int
    conversion_rate: float
    total_points_earned: int
    total_revenue_earned: float
    current_tier: str
    tier_level: int
    bonus_multiplier: float
    next_tier: Optional[Dict[str, Any]]
    recent_referrals: List[Dict[str, Any]]


class ReferralLeaderboardEntry(BaseModel):
    """Leaderboard entry for top referrers."""
    rank: int
    user_id: str
    name: str
    avatar_url: Optional[str]
    total_conversions: int
    total_revenue: float
    tier: str
    tier_level: int


class ReferralHistoryEntry(BaseModel):
    """Individual referral history entry."""
    referred_user_id: str
    referred_user_name: str
    signup_date: str
    converted: bool
    conversion_date: Optional[str]
    points_earned: int
    revenue_earned: float


class RevenueHistoryEntry(BaseModel):
    """Revenue earned from referrals."""
    month: str
    premium_community_revenue: float
    referral_count: int
    details: List[Dict[str, Any]]


# ============================================================================
# Helper Functions
# ============================================================================

def generate_referral_code(user_id: str) -> str:
    """
    Generate unique referral code.

    Format: REDDY_<6_RANDOM_CHARS>
    Example: REDDY_A8X2P9
    """
    random_part = ''.join(secrets.choice(string.ascii_uppercase + string.digits)
                         for _ in range(6))
    return f"REDDY_{random_part}"


def calculate_tier(total_conversions: int) -> Dict[str, Any]:
    """Calculate referral tier based on total conversions."""
    current_tier = REFERRAL_TIERS[1]  # Default to Starter

    for tier_level in sorted(REFERRAL_TIERS.keys(), reverse=True):
        tier_info = REFERRAL_TIERS[tier_level]
        if total_conversions >= tier_info["min_conversions"]:
            current_tier = {
                "level": tier_level,
                "name": tier_info["name"],
                "bonus_multiplier": tier_info["bonus_multiplier"],
                "min_conversions": tier_info["min_conversions"]
            }
            break

    # Find next tier
    next_tier = None
    for tier_level in sorted(REFERRAL_TIERS.keys()):
        if tier_level > current_tier["level"]:
            next_tier_info = REFERRAL_TIERS[tier_level]
            next_tier = {
                "level": tier_level,
                "name": next_tier_info["name"],
                "min_conversions": next_tier_info["min_conversions"],
                "conversions_needed": next_tier_info["min_conversions"] - total_conversions
            }
            break

    return {"current": current_tier, "next": next_tier}


def award_points(db, user_id: str, points: int, reason: str):
    """Award points to user and update their total."""
    user_ref = db.collection('users').document(user_id)
    user_ref.update({
        'stats.total_points': firestore.Increment(points),
        'stats.last_points_update': datetime.utcnow()
    })

    # Log points transaction
    db.collection('points_transactions').add({
        'user_id': user_id,
        'points': points,
        'reason': reason,
        'timestamp': datetime.utcnow()
    })


def send_realtime_notification(realtime_db, user_id: str, notification: Dict[str, Any]):
    """Send real-time notification via Firebase Realtime Database."""
    try:
        realtime_db.child('notifications').child(user_id).push(notification)
    except Exception as e:
        print(f"Failed to send notification to {user_id}: {e}")


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/code", response_model=ReferralCodeResponse)
async def get_referral_code(current_user: str = Depends(get_current_user)):
    """
    Get user's referral code (creates one if doesn't exist).

    Returns referral code with basic stats.
    """
    db = get_firestore_client()

    # Check if user already has a referral code
    referral_query = db.collection('referral_codes').where('user_id', '==', current_user).limit(1).stream()
    referral_docs = list(referral_query)

    if referral_docs:
        # Return existing code
        referral_doc = referral_docs[0]
        referral_data = referral_doc.to_dict()

        # Calculate tier
        tier_info = calculate_tier(referral_data.get('total_conversions', 0))

        return ReferralCodeResponse(
            code=referral_data['code'],
            user_id=referral_data['user_id'],
            created_at=referral_data['created_at'].isoformat(),
            total_signups=referral_data.get('total_signups', 0),
            total_conversions=referral_data.get('total_conversions', 0),
            total_points_earned=referral_data.get('total_points_earned', 0),
            total_revenue_earned=referral_data.get('total_revenue_earned', 0.0),
            current_tier=tier_info['current']['name'],
            bonus_multiplier=tier_info['current']['bonus_multiplier']
        )

    # Create new referral code
    new_code = generate_referral_code(current_user)

    # Ensure code is unique
    max_attempts = 10
    for _ in range(max_attempts):
        existing_code = db.collection('referral_codes').where('code', '==', new_code).limit(1).stream()
        if not list(existing_code):
            break
        new_code = generate_referral_code(current_user)

    # Create referral code document
    referral_data = {
        'code': new_code,
        'user_id': current_user,
        'created_at': datetime.utcnow(),
        'total_signups': 0,
        'total_conversions': 0,
        'total_points_earned': 0,
        'total_revenue_earned': 0.0,
        'tier_level': 1,
        'tier_name': 'Starter',
        'bonus_multiplier': 1.0
    }

    db.collection('referral_codes').add(referral_data)

    return ReferralCodeResponse(
        code=new_code,
        user_id=current_user,
        created_at=referral_data['created_at'].isoformat(),
        total_signups=0,
        total_conversions=0,
        total_points_earned=0,
        total_revenue_earned=0.0,
        current_tier='Starter',
        bonus_multiplier=1.0
    )


@router.post("/apply")
async def apply_referral_code(
    request: ApplyReferralCodeRequest,
    current_user: str = Depends(get_current_user)
):
    """
    Apply referral code (during or after signup).

    Awards signup bonus to both referrer and referee.
    Can only be applied once per user.
    """
    db = get_firestore_client()
    realtime_db = get_realtime_db()

    # Check if user already used a referral code
    user_ref = db.collection('users').document(current_user)
    user_doc = user_ref.get()

    if not user_doc.exists:
        raise HTTPException(404, "User not found")

    user_data = user_doc.to_dict()
    if user_data.get('referral', {}).get('referred_by'):
        raise HTTPException(400, "Referral code already applied")

    # Find referral code
    referral_query = db.collection('referral_codes').where('code', '==', request.referral_code).limit(1).stream()
    referral_docs = list(referral_query)

    if not referral_docs:
        raise HTTPException(404, "Invalid referral code")

    referral_doc = referral_docs[0]
    referral_data = referral_doc.to_dict()
    referrer_id = referral_data['user_id']

    # Prevent self-referral
    if referrer_id == current_user:
        raise HTTPException(400, "Cannot use your own referral code")

    # Get referrer data
    referrer_ref = db.collection('users').document(referrer_id)
    referrer_doc = referrer_ref.get()

    if not referrer_doc.exists:
        raise HTTPException(404, "Referrer not found")

    referrer_data = referrer_doc.to_dict()

    # Apply referral using batch for atomicity
    batch = db.batch()

    # Update referee (current user) - award signup bonus
    batch.update(user_ref, {
        'referral.referred_by': referrer_id,
        'referral.referral_code': request.referral_code,
        'referral.applied_at': datetime.utcnow(),
        'referral.signup_bonus_points': REFERRAL_REWARDS['referee_signup'],
        'stats.total_points': firestore.Increment(REFERRAL_REWARDS['referee_signup'])
    })

    # Update referrer - increment signup count and award points
    tier_info = calculate_tier(referral_data.get('total_conversions', 0))
    referrer_points = int(REFERRAL_REWARDS['referrer_signup'] * tier_info['current']['bonus_multiplier'])

    batch.update(referral_doc.ref, {
        'total_signups': firestore.Increment(1),
        'total_points_earned': firestore.Increment(referrer_points),
        'last_referral_at': datetime.utcnow()
    })

    batch.update(referrer_ref, {
        'stats.total_points': firestore.Increment(referrer_points),
        'stats.total_referrals': firestore.Increment(1)
    })

    # Create referral record
    referral_record = {
        'referrer_id': referrer_id,
        'referee_id': current_user,
        'referral_code': request.referral_code,
        'signup_date': datetime.utcnow(),
        'converted': False,
        'conversion_date': None,
        'referee_points_awarded': REFERRAL_REWARDS['referee_signup'],
        'referrer_points_awarded': referrer_points,
        'total_revenue_generated': 0.0
    }

    referral_record_ref = db.collection('referrals').document()
    batch.set(referral_record_ref, referral_record)

    # Commit all changes
    batch.commit()

    # Send notifications
    send_realtime_notification(realtime_db, current_user, {
        'type': 'referral_applied',
        'title': 'Welcome Bonus!',
        'message': f'You received {REFERRAL_REWARDS["referee_signup"]} points for joining via referral!',
        'points': REFERRAL_REWARDS['referee_signup'],
        'timestamp': datetime.utcnow().isoformat()
    })

    send_realtime_notification(realtime_db, referrer_id, {
        'type': 'new_referral',
        'title': 'New Referral Signup!',
        'message': f'{user_data.get("name", "Someone")} joined using your code! You earned {referrer_points} points.',
        'points': referrer_points,
        'referee_name': user_data.get('name', 'New User'),
        'timestamp': datetime.utcnow().isoformat()
    })

    return {
        "success": True,
        "message": "Referral code applied successfully",
        "points_awarded": REFERRAL_REWARDS['referee_signup'],
        "referrer_name": referrer_data.get('name', 'Unknown')
    }


@router.post("/convert/{referee_id}")
async def mark_referral_converted(
    referee_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Mark referral as converted (internal endpoint - called when referee completes first challenge).

    Awards conversion bonus to both referrer and referee.
    Only works if the current user is the referee.
    """
    db = get_firestore_client()
    realtime_db = get_realtime_db()

    # Security: Only the referee can trigger their own conversion
    if current_user != referee_id:
        raise HTTPException(403, "Cannot mark conversion for another user")

    # Get user's referral info
    user_ref = db.collection('users').document(current_user)
    user_doc = user_ref.get()

    if not user_doc.exists:
        raise HTTPException(404, "User not found")

    user_data = user_doc.to_dict()
    referrer_id = user_data.get('referral', {}).get('referred_by')

    if not referrer_id:
        # User wasn't referred, nothing to convert
        return {"success": False, "message": "No referral to convert"}

    # Check if already converted
    if user_data.get('referral', {}).get('converted', False):
        return {"success": False, "message": "Referral already converted"}

    # Find referral record
    referral_query = db.collection('referrals').where(
        'referee_id', '==', current_user
    ).limit(1).stream()

    referral_docs = list(referral_query)
    if not referral_docs:
        raise HTTPException(404, "Referral record not found")

    referral_doc = referral_docs[0]
    referral_data = referral_doc.to_dict()

    # Get referral code info for tier calculation
    referral_code_query = db.collection('referral_codes').where(
        'user_id', '==', referrer_id
    ).limit(1).stream()

    referral_code_docs = list(referral_code_query)
    if not referral_code_docs:
        raise HTTPException(404, "Referral code not found")

    referral_code_doc = referral_code_docs[0]
    referral_code_data = referral_code_doc.to_dict()

    # Calculate tier and bonus
    tier_info = calculate_tier(referral_code_data.get('total_conversions', 0))
    referrer_points = int(REFERRAL_REWARDS['referrer_conversion'] * tier_info['current']['bonus_multiplier'])

    # Get referrer data
    referrer_ref = db.collection('users').document(referrer_id)
    referrer_doc = referrer_ref.get()

    if not referrer_doc.exists:
        raise HTTPException(404, "Referrer not found")

    referrer_data = referrer_doc.to_dict()

    # Award conversion bonuses using batch
    batch = db.batch()

    # Update referee
    batch.update(user_ref, {
        'referral.converted': True,
        'referral.conversion_date': datetime.utcnow(),
        'referral.conversion_bonus_points': REFERRAL_REWARDS['referee_first_challenge'],
        'stats.total_points': firestore.Increment(REFERRAL_REWARDS['referee_first_challenge'])
    })

    # Update referrer
    batch.update(referrer_ref, {
        'stats.total_points': firestore.Increment(referrer_points),
        'stats.total_conversions': firestore.Increment(1)
    })

    # Update referral code stats
    batch.update(referral_code_doc.ref, {
        'total_conversions': firestore.Increment(1),
        'total_points_earned': firestore.Increment(referrer_points)
    })

    # Update referral record
    batch.update(referral_doc.ref, {
        'converted': True,
        'conversion_date': datetime.utcnow(),
        'referee_conversion_points': REFERRAL_REWARDS['referee_first_challenge'],
        'referrer_conversion_points': referrer_points
    })

    batch.commit()

    # Send notifications
    send_realtime_notification(realtime_db, current_user, {
        'type': 'referral_converted',
        'title': 'First Challenge Completed!',
        'message': f'You earned {REFERRAL_REWARDS["referee_first_challenge"]} bonus points!',
        'points': REFERRAL_REWARDS['referee_first_challenge'],
        'timestamp': datetime.utcnow().isoformat()
    })

    send_realtime_notification(realtime_db, referrer_id, {
        'type': 'referral_conversion',
        'title': 'Referral Converted!',
        'message': f'{user_data.get("name", "Your referral")} completed their first challenge! You earned {referrer_points} points.',
        'points': referrer_points,
        'referee_name': user_data.get('name', 'Unknown'),
        'timestamp': datetime.utcnow().isoformat()
    })

    return {
        "success": True,
        "message": "Referral converted successfully",
        "points_awarded": REFERRAL_REWARDS['referee_first_challenge'],
        "referrer_points": referrer_points
    }


@router.get("/stats", response_model=ReferralStatsResponse)
async def get_referral_stats(current_user: str = Depends(get_current_user)):
    """
    Get comprehensive referral statistics for current user.

    Includes tier info, conversion rate, earnings, and recent referrals.
    """
    db = get_firestore_client()

    # Get referral code
    referral_query = db.collection('referral_codes').where('user_id', '==', current_user).limit(1).stream()
    referral_docs = list(referral_query)

    if not referral_docs:
        raise HTTPException(404, "Referral code not found. Generate one first via GET /code")

    referral_doc = referral_docs[0]
    referral_data = referral_doc.to_dict()

    # Calculate tier
    tier_info = calculate_tier(referral_data.get('total_conversions', 0))

    # Get recent referrals
    recent_referrals_query = db.collection('referrals').where(
        'referrer_id', '==', current_user
    ).order_by('signup_date', direction=firestore.Query.DESCENDING).limit(10).stream()

    recent_referrals = []
    for ref_doc in recent_referrals_query:
        ref_data = ref_doc.to_dict()

        # Get referee name
        referee_doc = db.collection('users').document(ref_data['referee_id']).get()
        referee_name = referee_doc.to_dict().get('name', 'Unknown') if referee_doc.exists else 'Unknown'

        recent_referrals.append({
            'referee_id': ref_data['referee_id'],
            'referee_name': referee_name,
            'signup_date': ref_data['signup_date'].isoformat(),
            'converted': ref_data.get('converted', False),
            'conversion_date': ref_data.get('conversion_date').isoformat() if ref_data.get('conversion_date') else None
        })

    # Calculate conversion rate
    total_signups = referral_data.get('total_signups', 0)
    total_conversions = referral_data.get('total_conversions', 0)
    conversion_rate = (total_conversions / total_signups * 100) if total_signups > 0 else 0.0

    return ReferralStatsResponse(
        referral_code=referral_data['code'],
        total_signups=total_signups,
        total_conversions=total_conversions,
        conversion_rate=round(conversion_rate, 2),
        total_points_earned=referral_data.get('total_points_earned', 0),
        total_revenue_earned=referral_data.get('total_revenue_earned', 0.0),
        current_tier=tier_info['current']['name'],
        tier_level=tier_info['current']['level'],
        bonus_multiplier=tier_info['current']['bonus_multiplier'],
        next_tier=tier_info['next'],
        recent_referrals=recent_referrals
    )


@router.get("/history", response_model=List[ReferralHistoryEntry])
async def get_referral_history(
    current_user: str = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=100)
):
    """
    Get complete referral history for current user.

    Returns all referrals with conversion status and earnings.
    """
    db = get_firestore_client()

    # Get all referrals
    referrals_query = db.collection('referrals').where(
        'referrer_id', '==', current_user
    ).order_by('signup_date', direction=firestore.Query.DESCENDING).limit(limit).stream()

    history = []
    for ref_doc in referrals_query:
        ref_data = ref_doc.to_dict()

        # Get referee name
        referee_doc = db.collection('users').document(ref_data['referee_id']).get()
        referee_name = referee_doc.to_dict().get('name', 'Unknown') if referee_doc.exists else 'Unknown'

        # Calculate points earned from this referral
        points_earned = ref_data.get('referrer_points_awarded', 0)
        if ref_data.get('converted', False):
            points_earned += ref_data.get('referrer_conversion_points', 0)

        history.append(ReferralHistoryEntry(
            referred_user_id=ref_data['referee_id'],
            referred_user_name=referee_name,
            signup_date=ref_data['signup_date'].isoformat(),
            converted=ref_data.get('converted', False),
            conversion_date=ref_data.get('conversion_date').isoformat() if ref_data.get('conversion_date') else None,
            points_earned=points_earned,
            revenue_earned=ref_data.get('total_revenue_generated', 0.0)
        ))

    return history


@router.get("/leaderboard", response_model=List[ReferralLeaderboardEntry])
async def get_referral_leaderboard(
    limit: int = Query(50, ge=1, le=100),
    current_user: str = Depends(get_current_user)
):
    """
    Get global referral leaderboard (top referrers by conversions).

    Publicly visible leaderboard of top performers.
    """
    db = get_firestore_client()

    # Get top referrers by conversions
    top_referrers_query = db.collection('referral_codes').order_by(
        'total_conversions', direction=firestore.Query.DESCENDING
    ).limit(limit).stream()

    leaderboard = []
    rank = 1

    for ref_doc in top_referrers_query:
        ref_data = ref_doc.to_dict()

        # Get user info
        user_doc = db.collection('users').document(ref_data['user_id']).get()
        if not user_doc.exists:
            continue

        user_data = user_doc.to_dict()

        # Calculate tier
        tier_info = calculate_tier(ref_data.get('total_conversions', 0))

        leaderboard.append(ReferralLeaderboardEntry(
            rank=rank,
            user_id=ref_data['user_id'],
            name=user_data.get('name', 'Unknown'),
            avatar_url=user_data.get('avatar_url'),
            total_conversions=ref_data.get('total_conversions', 0),
            total_revenue=ref_data.get('total_revenue_earned', 0.0),
            tier=tier_info['current']['name'],
            tier_level=tier_info['current']['level']
        ))

        rank += 1

    return leaderboard


@router.get("/revenue-history", response_model=List[RevenueHistoryEntry])
async def get_revenue_history(
    current_user: str = Depends(get_current_user),
    months: int = Query(12, ge=1, le=24)
):
    """
    Get monthly revenue history from referrals.

    Shows 10% revenue share from premium communities joined by referrals.
    """
    db = get_firestore_client()

    # Get revenue transactions for this referrer
    revenue_query = db.collection('referral_revenue').where(
        'referrer_id', '==', current_user
    ).order_by('month', direction=firestore.Query.DESCENDING).limit(months).stream()

    revenue_history = []
    for rev_doc in revenue_query:
        rev_data = rev_doc.to_dict()

        revenue_history.append(RevenueHistoryEntry(
            month=rev_data['month'],
            premium_community_revenue=rev_data.get('total_revenue', 0.0),
            referral_count=rev_data.get('referral_count', 0),
            details=rev_data.get('details', [])
        ))

    return revenue_history


@router.post("/track-revenue")
async def track_referral_revenue(
    referee_id: str,
    community_id: str,
    revenue_amount: float,
    current_user: str = Depends(get_current_user)
):
    """
    Track revenue generated by referral (internal endpoint).

    Called when a referred user joins a premium community.
    Awards 10% of community revenue to referrer.
    """
    db = get_firestore_client()
    realtime_db = get_realtime_db()

    # Get referee's referral info
    referee_doc = db.collection('users').document(referee_id).get()
    if not referee_doc.exists:
        raise HTTPException(404, "Referee not found")

    referee_data = referee_doc.to_dict()
    referrer_id = referee_data.get('referral', {}).get('referred_by')

    if not referrer_id:
        # User wasn't referred, no revenue to track
        return {"success": False, "message": "No referrer to credit"}

    # Calculate referrer's share (10%)
    referrer_revenue = revenue_amount * REFERRAL_REVENUE_SHARE

    # Update referral code stats
    referral_code_query = db.collection('referral_codes').where(
        'user_id', '==', referrer_id
    ).limit(1).stream()

    referral_code_docs = list(referral_code_query)
    if referral_code_docs:
        referral_code_doc = referral_code_docs[0]
        referral_code_doc.ref.update({
            'total_revenue_earned': firestore.Increment(referrer_revenue)
        })

    # Update referral record
    referral_query = db.collection('referrals').where(
        'referee_id', '==', referee_id
    ).limit(1).stream()

    referral_docs = list(referral_query)
    if referral_docs:
        referral_doc = referral_docs[0]
        referral_doc.ref.update({
            'total_revenue_generated': firestore.Increment(referrer_revenue)
        })

    # Record monthly revenue
    current_month = datetime.utcnow().strftime('%Y-%m')
    revenue_doc_id = f"{referrer_id}_{current_month}"
    revenue_ref = db.collection('referral_revenue').document(revenue_doc_id)
    revenue_doc = revenue_ref.get()

    if revenue_doc.exists:
        # Update existing month
        revenue_ref.update({
            'total_revenue': firestore.Increment(referrer_revenue),
            'referral_count': firestore.Increment(1),
            'details': firestore.ArrayUnion([{
                'referee_id': referee_id,
                'community_id': community_id,
                'revenue': referrer_revenue,
                'date': datetime.utcnow().isoformat()
            }])
        })
    else:
        # Create new month record
        revenue_ref.set({
            'referrer_id': referrer_id,
            'month': current_month,
            'total_revenue': referrer_revenue,
            'referral_count': 1,
            'details': [{
                'referee_id': referee_id,
                'community_id': community_id,
                'revenue': referrer_revenue,
                'date': datetime.utcnow().isoformat()
            }]
        })

    # Send notification to referrer
    send_realtime_notification(realtime_db, referrer_id, {
        'type': 'referral_revenue',
        'title': 'Referral Revenue Earned!',
        'message': f'Your referral joined a premium community. You earned ${referrer_revenue:.2f}!',
        'revenue': referrer_revenue,
        'timestamp': datetime.utcnow().isoformat()
    })

    return {
        "success": True,
        "referrer_id": referrer_id,
        "revenue_credited": referrer_revenue
    }
