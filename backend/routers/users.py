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
from firebase_client import (
    get_firestore_client,
    create_user,
    get_user,
    update_user,
    find_nearby_users
)
from firebase_admin import firestore

router = APIRouter()

@router.post("/register", response_model=UserProfile)
async def register_user(user: UserCreate):
    """Register a new user."""
    db = get_firestore_client()

    # Check if email exists
    existing = db.collection('users').where('email', '==', user.email).limit(1).stream()

    if list(existing):
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create user with auto-generated ID
    user_ref = db.collection('users').document()
    uid = user_ref.id

    user_data = {
        'email': user.email,
        'name': user.name,
        'created_at': firestore.SERVER_TIMESTAMP,
        'location': None,
        'stats': {
            'challenges_completed': 0,
            'total_distance_km': 0,
            'total_calories': 0,
            'challenges_this_week': 0,
            'points_balance': 0,
            'current_streak': 0
        },
        'preferences': {}
    }

    user_ref.set(user_data)

    return UserProfile(
        id=uid,
        email=user.email,
        name=user.name,
        created_at=datetime.utcnow(),
        stats=user_data['stats'],
        preferences=user_data['preferences']
    )

@router.get("/{user_id}", response_model=UserProfile)
async def get_user_profile(user_id: str):
    """Get user profile by ID."""
    user_data = get_user(user_id)

    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    return UserProfile(
        id=user_data["id"],
        email=user_data["email"],
        name=user_data["name"],
        created_at=user_data.get("created_at") or datetime.utcnow(),
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
    db = get_firestore_client()

    # Update location using Firebase GeoPoint
    user_ref = db.collection('users').document(user_id)

    user_ref.update({
        'location': firestore.GeoPoint(
            location_update.location.lat,
            location_update.location.lon
        )
    })

    return None

@router.post("/nearby", response_model=List[NearbyUser])
async def find_nearby_users_endpoint(search: NearbySearchRequest):
    """
    Find users within radius of location.

    Used for:
    - Creating ad-hoc challenges with nearby users
    - Social discovery
    - Workout buddy matching
    """
    # Use Firebase Haversine-based nearby search
    nearby = find_nearby_users(
        lat=search.location.lat,
        lon=search.location.lon,
        radius_meters=search.radius_meters
    )

    return [
        NearbyUser(
            user_id=row["user_id"],
            name=row["name"],
            distance_meters=row["distance_meters"]
        )
        for row in nearby
    ]

@router.get("/{user_id}/stats")
async def get_user_stats_endpoint(user_id: str):
    """Get user fitness statistics."""
    db = get_firestore_client()

    # Get user stats
    user_doc = db.collection('users').document(user_id).get()

    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user_doc.to_dict()
    stats = user_data.get("stats", {})

    # Get challenge history from subcollections
    challenges_ref = db.collection_group('participants').where('user_id', '==', user_id)
    challenges = challenges_ref.stream()

    challenge_list = []
    completed_count = 0

    for challenge_doc in challenges:
        challenge_data = challenge_doc.to_dict()
        challenge_list.append({
            'status': challenge_data.get('status'),
            'stats': challenge_data.get('stats', {})
        })
        if challenge_data.get('status') == 'completed':
            completed_count += 1

    return {
        "overall_stats": stats,
        "challenges_completed": completed_count,
        "recent_challenges": challenge_list[:10]
    }
