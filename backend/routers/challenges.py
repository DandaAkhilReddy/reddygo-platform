"""
ReddyGo Challenges Router

API endpoints for challenge creation, discovery, and management.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime
import uuid

from models import (
    ChallengeCreate,
    ChallengeResponse,
    ChallengeJoinRequest,
    ChallengeParticipant,
    NearbySearchRequest,
    NearbyChallenge,
    Location,
    ChallengeStatus,
    ParticipantStatus
)
from database import get_supabase_client

router = APIRouter()

@router.post("/create", response_model=ChallengeResponse)
async def create_challenge(challenge: ChallengeCreate, user_id: str):
    """
    Create a new geo-challenge.

    Validates:
    - Location is not in restricted zone (school, hospital)
    - Time window is reasonable (10 min - 4 hours)
    - Zone radius is safe (50m - 5km)
    """
    supabase = get_supabase_client()

    # Validate challenge duration
    duration_minutes = (challenge.end_time - challenge.start_time).total_seconds() / 60
    if duration_minutes < 10:
        raise HTTPException(status_code=400, detail="Challenge must be at least 10 minutes")
    if duration_minutes > 240:
        raise HTTPException(status_code=400, detail="Challenge cannot exceed 4 hours")

    # TODO: Check for restricted zones (schools, hospitals) using safety agent

    # Insert challenge
    result = supabase.table("challenges").insert({
        "creator_id": user_id,
        "title": challenge.title,
        "description": challenge.description,
        "challenge_type": challenge.challenge_type.value,
        "zone_center": f"POINT({challenge.zone_center.lon} {challenge.zone_center.lat})",
        "zone_radius_meters": challenge.zone_radius_meters,
        "start_time": challenge.start_time.isoformat(),
        "end_time": challenge.end_time.isoformat(),
        "max_participants": challenge.max_participants,
        "rules": challenge.rules.dict(),
        "status": ChallengeStatus.PENDING.value
    }).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create challenge")

    challenge_data = result.data[0]

    # Auto-join creator
    supabase.table("challenge_participants").insert({
        "challenge_id": challenge_data["id"],
        "user_id": user_id,
        "status": ParticipantStatus.ACTIVE.value
    }).execute()

    return ChallengeResponse(
        id=challenge_data["id"],
        creator_id=challenge_data["creator_id"],
        title=challenge_data["title"],
        description=challenge_data["description"],
        challenge_type=challenge_data["challenge_type"],
        zone_center=Location(
            lat=float(challenge.zone_center.lat),
            lon=float(challenge.zone_center.lon)
        ),
        zone_radius_meters=challenge_data["zone_radius_meters"],
        start_time=datetime.fromisoformat(challenge_data["start_time"]),
        end_time=datetime.fromisoformat(challenge_data["end_time"]),
        max_participants=challenge_data["max_participants"],
        rules=challenge.rules,
        status=ChallengeStatus(challenge_data["status"]),
        created_at=datetime.fromisoformat(challenge_data["created_at"]),
        participant_count=1
    )

@router.post("/nearby", response_model=List[NearbyChallenge])
async def find_nearby_challenges(search: NearbySearchRequest):
    """
    Find challenges within radius of user location.

    Uses PostGIS spatial query for efficient geospatial search.
    """
    supabase = get_supabase_client()

    # Call PostGIS function
    result = supabase.rpc(
        "find_nearby_challenges",
        {
            "user_lat": search.location.lat,
            "user_lon": search.location.lon,
            "radius_meters": search.radius_meters
        }
    ).execute()

    if not result.data:
        return []

    return [
        NearbyChallenge(
            id=row["challenge_id"],
            title=row["title"],
            distance_meters=row["distance_meters"]
        )
        for row in result.data
    ]

@router.get("/{challenge_id}", response_model=ChallengeResponse)
async def get_challenge(challenge_id: str):
    """Get challenge details by ID."""
    supabase = get_supabase_client()

    result = supabase.table("challenges").select("*").eq("id", challenge_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Challenge not found")

    challenge_data = result.data[0]

    # Get participant count
    participants = supabase.table("challenge_participants").select("id").eq("challenge_id", challenge_id).execute()
    participant_count = len(participants.data) if participants.data else 0

    # Parse zone_center from PostGIS format "POINT(lon lat)"
    zone_center_str = challenge_data["zone_center"]
    # Extract coordinates from "POINT(-122.123 37.456)" format
    coords = zone_center_str.replace("POINT(", "").replace(")", "").split()
    lon, lat = float(coords[0]), float(coords[1])

    return ChallengeResponse(
        id=challenge_data["id"],
        creator_id=challenge_data["creator_id"],
        title=challenge_data["title"],
        description=challenge_data["description"],
        challenge_type=challenge_data["challenge_type"],
        zone_center=Location(lat=lat, lon=lon),
        zone_radius_meters=challenge_data["zone_radius_meters"],
        start_time=datetime.fromisoformat(challenge_data["start_time"]),
        end_time=datetime.fromisoformat(challenge_data["end_time"]),
        max_participants=challenge_data["max_participants"],
        rules=challenge_data["rules"],
        status=ChallengeStatus(challenge_data["status"]),
        created_at=datetime.fromisoformat(challenge_data["created_at"]),
        participant_count=participant_count
    )

@router.post("/{challenge_id}/join", response_model=ChallengeParticipant)
async def join_challenge(challenge_id: str, join_request: ChallengeJoinRequest, user_id: str):
    """
    Join an existing challenge.

    Validates:
    - Challenge is active
    - User is within zone radius
    - Challenge not full
    - User not already joined
    """
    supabase = get_supabase_client()

    # Get challenge
    challenge_result = supabase.table("challenges").select("*").eq("id", challenge_id).execute()

    if not challenge_result.data:
        raise HTTPException(status_code=404, detail="Challenge not found")

    challenge = challenge_result.data[0]

    # Check status
    if challenge["status"] != ChallengeStatus.ACTIVE.value:
        raise HTTPException(status_code=400, detail="Challenge is not active")

    # Check if already joined
    existing = supabase.table("challenge_participants").select("id").eq("challenge_id", challenge_id).eq("user_id", user_id).execute()

    if existing.data:
        raise HTTPException(status_code=400, detail="Already joined this challenge")

    # Check capacity
    participants = supabase.table("challenge_participants").select("id").eq("challenge_id", challenge_id).execute()
    if len(participants.data) >= challenge["max_participants"]:
        raise HTTPException(status_code=400, detail="Challenge is full")

    # TODO: Validate user is within zone_radius using PostGIS

    # Get user details
    user_result = supabase.table("users").select("name").eq("id", user_id).execute()
    user_name = user_result.data[0]["name"] if user_result.data else "Unknown"

    # Join challenge
    participant_result = supabase.table("challenge_participants").insert({
        "challenge_id": challenge_id,
        "user_id": user_id,
        "status": ParticipantStatus.ACTIVE.value
    }).execute()

    if not participant_result.data:
        raise HTTPException(status_code=500, detail="Failed to join challenge")

    participant_data = participant_result.data[0]

    return ChallengeParticipant(
        id=participant_data["id"],
        user_id=user_id,
        user_name=user_name,
        joined_at=datetime.fromisoformat(participant_data["joined_at"]),
        status=ParticipantStatus(participant_data["status"]),
        stats=participant_data.get("stats", {})
    )

@router.get("/{challenge_id}/participants", response_model=List[ChallengeParticipant])
async def get_challenge_participants(challenge_id: str):
    """Get all participants in a challenge."""
    supabase = get_supabase_client()

    # Get participants with user details
    result = supabase.table("challenge_participants").select("*, users(name)").eq("challenge_id", challenge_id).execute()

    if not result.data:
        return []

    return [
        ChallengeParticipant(
            id=row["id"],
            user_id=row["user_id"],
            user_name=row["users"]["name"] if row.get("users") else "Unknown",
            joined_at=datetime.fromisoformat(row["joined_at"]),
            status=ParticipantStatus(row["status"]),
            stats=row.get("stats", {})
        )
        for row in result.data
    ]

@router.delete("/{challenge_id}", status_code=204)
async def cancel_challenge(challenge_id: str, user_id: str):
    """Cancel a challenge (only creator can cancel)."""
    supabase = get_supabase_client()

    # Verify creator
    challenge_result = supabase.table("challenges").select("creator_id").eq("id", challenge_id).execute()

    if not challenge_result.data:
        raise HTTPException(status_code=404, detail="Challenge not found")

    if challenge_result.data[0]["creator_id"] != user_id:
        raise HTTPException(status_code=403, detail="Only creator can cancel challenge")

    # Update status
    supabase.table("challenges").update({"status": ChallengeStatus.CANCELLED.value}).eq("id", challenge_id).execute()

    return None
