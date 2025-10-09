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
from firebase_client import get_firestore_client, find_nearby_challenges
from firebase_admin import firestore

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
    db = get_firestore_client()

    # Validate challenge duration
    duration_minutes = (challenge.end_time - challenge.start_time).total_seconds() / 60
    if duration_minutes < 10:
        raise HTTPException(status_code=400, detail="Challenge must be at least 10 minutes")
    if duration_minutes > 240:
        raise HTTPException(status_code=400, detail="Challenge cannot exceed 4 hours")

    # TODO: Check for restricted zones (schools, hospitals) using safety agent

    # Create challenge document
    challenge_ref = db.collection('challenges').document()
    challenge_id = challenge_ref.id

    challenge_data = {
        "creator_id": user_id,
        "title": challenge.title,
        "description": challenge.description,
        "challenge_type": challenge.challenge_type.value,
        "zone_center": firestore.GeoPoint(challenge.zone_center.lat, challenge.zone_center.lon),
        "zone_radius_meters": challenge.zone_radius_meters,
        "start_time": challenge.start_time,
        "end_time": challenge.end_time,
        "max_participants": challenge.max_participants,
        "rules": challenge.rules.dict(),
        "status": ChallengeStatus.PENDING.value,
        "created_at": firestore.SERVER_TIMESTAMP
    }

    challenge_ref.set(challenge_data)

    # Auto-join creator as participant (subcollection)
    participant_ref = challenge_ref.collection('participants').document(user_id)
    participant_ref.set({
        "user_id": user_id,
        "joined_at": firestore.SERVER_TIMESTAMP,
        "status": ParticipantStatus.ACTIVE.value,
        "stats": {}
    })

    return ChallengeResponse(
        id=challenge_id,
        creator_id=user_id,
        title=challenge.title,
        description=challenge.description,
        challenge_type=challenge.challenge_type.value,
        zone_center=challenge.zone_center,
        zone_radius_meters=challenge.zone_radius_meters,
        start_time=challenge.start_time,
        end_time=challenge.end_time,
        max_participants=challenge.max_participants,
        rules=challenge.rules,
        status=ChallengeStatus.PENDING,
        created_at=datetime.utcnow(),
        participant_count=1
    )

@router.post("/nearby", response_model=List[NearbyChallenge])
async def find_nearby_challenges_endpoint(search: NearbySearchRequest):
    """
    Find challenges within radius of user location.

    Uses Firebase Haversine distance calculation for geospatial search.
    """
    # Use Firebase Haversine-based nearby search
    nearby = find_nearby_challenges(
        lat=search.location.lat,
        lon=search.location.lon,
        radius_meters=search.radius_meters
    )

    return [
        NearbyChallenge(
            id=row["challenge_id"],
            title=row["title"],
            distance_meters=row["distance_meters"]
        )
        for row in nearby
    ]

@router.get("/{challenge_id}", response_model=ChallengeResponse)
async def get_challenge(challenge_id: str):
    """Get challenge details by ID."""
    db = get_firestore_client()

    challenge_doc = db.collection('challenges').document(challenge_id).get()

    if not challenge_doc.exists:
        raise HTTPException(status_code=404, detail="Challenge not found")

    challenge_data = challenge_doc.to_dict()

    # Get participant count from subcollection
    participants = db.collection('challenges').document(challenge_id).collection('participants').stream()
    participant_count = len(list(participants))

    # Extract coordinates from GeoPoint
    zone_center = challenge_data["zone_center"]
    lat = zone_center.latitude
    lon = zone_center.longitude

    return ChallengeResponse(
        id=challenge_id,
        creator_id=challenge_data["creator_id"],
        title=challenge_data["title"],
        description=challenge_data["description"],
        challenge_type=challenge_data["challenge_type"],
        zone_center=Location(lat=lat, lon=lon),
        zone_radius_meters=challenge_data["zone_radius_meters"],
        start_time=challenge_data["start_time"],
        end_time=challenge_data["end_time"],
        max_participants=challenge_data["max_participants"],
        rules=challenge_data["rules"],
        status=ChallengeStatus(challenge_data["status"]),
        created_at=challenge_data.get("created_at") or datetime.utcnow(),
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
    db = get_firestore_client()

    # Get challenge
    challenge_doc = db.collection('challenges').document(challenge_id).get()

    if not challenge_doc.exists:
        raise HTTPException(status_code=404, detail="Challenge not found")

    challenge = challenge_doc.to_dict()

    # Check status
    if challenge["status"] != ChallengeStatus.ACTIVE.value:
        raise HTTPException(status_code=400, detail="Challenge is not active")

    # Check if already joined
    participant_doc = db.collection('challenges').document(challenge_id).collection('participants').document(user_id).get()

    if participant_doc.exists:
        raise HTTPException(status_code=400, detail="Already joined this challenge")

    # Check capacity
    participants = list(db.collection('challenges').document(challenge_id).collection('participants').stream())
    if len(participants) >= challenge["max_participants"]:
        raise HTTPException(status_code=400, detail="Challenge is full")

    # TODO: Validate user is within zone_radius using Haversine distance

    # Get user details
    user_doc = db.collection('users').document(user_id).get()
    user_name = user_doc.to_dict().get("name", "Unknown") if user_doc.exists else "Unknown"

    # Join challenge as participant (subcollection)
    participant_ref = db.collection('challenges').document(challenge_id).collection('participants').document(user_id)
    participant_data = {
        "user_id": user_id,
        "joined_at": firestore.SERVER_TIMESTAMP,
        "status": ParticipantStatus.ACTIVE.value,
        "stats": {}
    }
    participant_ref.set(participant_data)

    return ChallengeParticipant(
        id=user_id,  # Use user_id as participant ID
        user_id=user_id,
        user_name=user_name,
        joined_at=datetime.utcnow(),
        status=ParticipantStatus.ACTIVE,
        stats={}
    )

@router.get("/{challenge_id}/participants", response_model=List[ChallengeParticipant])
async def get_challenge_participants(challenge_id: str):
    """Get all participants in a challenge."""
    db = get_firestore_client()

    # Get participants from subcollection
    participants_ref = db.collection('challenges').document(challenge_id).collection('participants')
    participants = participants_ref.stream()

    result = []
    for participant_doc in participants:
        participant_data = participant_doc.to_dict()
        user_id = participant_data["user_id"]

        # Get user name
        user_doc = db.collection('users').document(user_id).get()
        user_name = user_doc.to_dict().get("name", "Unknown") if user_doc.exists else "Unknown"

        result.append(ChallengeParticipant(
            id=user_id,
            user_id=user_id,
            user_name=user_name,
            joined_at=participant_data.get("joined_at") or datetime.utcnow(),
            status=ParticipantStatus(participant_data["status"]),
            stats=participant_data.get("stats", {})
        ))

    return result

@router.delete("/{challenge_id}", status_code=204)
async def cancel_challenge(challenge_id: str, user_id: str):
    """Cancel a challenge (only creator can cancel)."""
    db = get_firestore_client()

    # Verify creator
    challenge_doc = db.collection('challenges').document(challenge_id).get()

    if not challenge_doc.exists:
        raise HTTPException(status_code=404, detail="Challenge not found")

    challenge_data = challenge_doc.to_dict()

    if challenge_data["creator_id"] != user_id:
        raise HTTPException(status_code=403, detail="Only creator can cancel challenge")

    # Update status
    db.collection('challenges').document(challenge_id).update({
        "status": ChallengeStatus.CANCELLED.value
    })

    return None
