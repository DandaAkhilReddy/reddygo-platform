"""
ReddyGo Users Router

API endpoints for user management, profiles, and location updates.
"""

from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime

from models import (
    UserCreate,
    UserProfile,
    UserLocationUpdate,
    NearbyUser,
    NearbySearchRequest
)
from database import get_supabase_client

router = APIRouter()

@router.post("/register", response_model=UserProfile)
async def register_user(user: UserCreate):
    """Register a new user."""
    supabase = get_supabase_client()

    # Check if email exists
    existing = supabase.table("users").select("id").eq("email", user.email).execute()

    if existing.data:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create user
    result = supabase.table("users").insert({
        "email": user.email,
        "name": user.name
    }).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create user")

    user_data = result.data[0]

    return UserProfile(
        id=user_data["id"],
        email=user_data["email"],
        name=user_data["name"],
        created_at=datetime.fromisoformat(user_data["created_at"]),
        stats=user_data.get("stats", {}),
        preferences=user_data.get("preferences", {})
    )

@router.get("/{user_id}", response_model=UserProfile)
async def get_user(user_id: str):
    """Get user profile by ID."""
    supabase = get_supabase_client()

    result = supabase.table("users").select("*").eq("id", user_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = result.data[0]

    return UserProfile(
        id=user_data["id"],
        email=user_data["email"],
        name=user_data["name"],
        created_at=datetime.fromisoformat(user_data["created_at"]),
        stats=user_data.get("stats", {}),
        preferences=user_data.get("preferences", {})
    )

@router.put("/{user_id}/location", status_code=204)
async def update_location(user_id: str, location_update: UserLocationUpdate):
    """
    Update user's last known location.

    Used for:
    - Finding nearby users for challenges
    - Safety features (know where users are during challenges)
    - Location-based recommendations
    """
    supabase = get_supabase_client()

    # Update last_location using PostGIS POINT
    result = supabase.table("users").update({
        "last_location": f"POINT({location_update.location.lon} {location_update.location.lat})"
    }).eq("id", user_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="User not found")

    return None

@router.post("/nearby", response_model=List[NearbyUser])
async def find_nearby_users(search: NearbySearchRequest):
    """
    Find users within radius of location.

    Used for:
    - Creating ad-hoc challenges with nearby users
    - Social discovery
    - Workout buddy matching
    """
    supabase = get_supabase_client()

    # Call PostGIS function
    result = supabase.rpc(
        "find_nearby_users",
        {
            "user_lat": search.location.lat,
            "user_lon": search.location.lon,
            "radius_meters": search.radius_meters
        }
    ).execute()

    if not result.data:
        return []

    return [
        NearbyUser(
            user_id=row["user_id"],
            name=row["name"],
            distance_meters=row["distance_meters"]
        )
        for row in result.data
    ]

@router.get("/{user_id}/stats")
async def get_user_stats(user_id: str):
    """Get user fitness statistics."""
    supabase = get_supabase_client()

    # Get user stats
    user_result = supabase.table("users").select("stats").eq("id", user_id).execute()

    if not user_result.data:
        raise HTTPException(status_code=404, detail="User not found")

    # Get challenge history
    challenges = supabase.table("challenge_participants").select("status, stats").eq("user_id", user_id).execute()

    completed_challenges = [c for c in challenges.data if c["status"] == "completed"] if challenges.data else []

    return {
        "overall_stats": user_result.data[0]["stats"],
        "challenges_completed": len(completed_challenges),
        "recent_challenges": challenges.data[:10] if challenges.data else []
    }
