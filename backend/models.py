"""
ReddyGo Pydantic Models

Request/response validation schemas for the API.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# Enums
class ChallengeType(str, Enum):
    RACE = "race"
    ZONE_CONTROL = "zone_control"
    SCAVENGER_HUNT = "scavenger_hunt"

class ChallengeStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class ParticipantStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    DISQUALIFIED = "disqualified"

# Location models
class Location(BaseModel):
    """Geographic coordinates."""
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)

    @validator('lat', 'lon')
    def validate_coords(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError("Coordinates must be numeric")
        return v

class GPSPoint(BaseModel):
    """Single GPS tracking point."""
    lat: float
    lon: float
    timestamp: datetime
    accuracy: float  # meters
    speed: Optional[float] = None  # m/s
    altitude: Optional[float] = None  # meters

# User models
class UserCreate(BaseModel):
    email: str
    name: str

class UserProfile(BaseModel):
    id: str
    email: str
    name: str
    created_at: datetime
    stats: Dict[str, Any] = {}
    preferences: Dict[str, Any] = {}

class UserLocationUpdate(BaseModel):
    location: Location

# Challenge models
class ChallengeRules(BaseModel):
    """Challenge-specific rules configuration."""
    min_distance_km: Optional[float] = None
    max_time_minutes: Optional[int] = None
    checkpoints: Optional[List[Location]] = None
    scoring_type: str = "time"  # 'time', 'distance', 'points'

class ChallengeCreate(BaseModel):
    """Request to create a new challenge."""
    title: str = Field(..., min_length=3, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    challenge_type: ChallengeType
    zone_center: Location
    zone_radius_meters: int = Field(..., ge=50, le=5000)
    start_time: datetime
    end_time: datetime
    max_participants: int = Field(10, ge=2, le=50)
    rules: ChallengeRules

    @validator('end_time')
    def validate_end_time(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError("end_time must be after start_time")
        return v

class ChallengeResponse(BaseModel):
    """Challenge details response."""
    id: str
    creator_id: str
    title: str
    description: Optional[str]
    challenge_type: ChallengeType
    zone_center: Location
    zone_radius_meters: int
    start_time: datetime
    end_time: datetime
    max_participants: int
    rules: ChallengeRules
    status: ChallengeStatus
    created_at: datetime
    participant_count: int = 0
    distance_from_user: Optional[float] = None  # meters

class ChallengeJoinRequest(BaseModel):
    """Request to join a challenge."""
    user_location: Location

class ChallengeParticipant(BaseModel):
    """Participant in a challenge."""
    id: str
    user_id: str
    user_name: str
    joined_at: datetime
    status: ParticipantStatus
    stats: Dict[str, Any] = {}

# GPS Tracking models
class SensorData(BaseModel):
    """Device sensor data for anti-cheat validation."""
    accelerometer: Optional[List[Dict[str, float]]] = None
    gyroscope: Optional[List[Dict[str, float]]] = None
    device_integrity: Optional[Dict[str, Any]] = None

class GPSTrackSubmission(BaseModel):
    """Submit GPS track for validation."""
    challenge_id: str
    track_data: List[GPSPoint]
    sensor_data: Optional[SensorData] = None

class ValidationResult(BaseModel):
    """GPS track validation result."""
    is_valid: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    flags: List[str] = []
    details: Dict[str, Any] = {}

# Nearby search models
class NearbyChallenge(BaseModel):
    """Challenge with distance information."""
    id: str
    title: str
    distance_meters: float

class NearbyUser(BaseModel):
    """User with distance information."""
    user_id: str
    name: str
    distance_meters: float

class NearbySearchRequest(BaseModel):
    """Request to find nearby challenges/users."""
    location: Location
    radius_meters: int = Field(5000, ge=100, le=10000)

# AI Coach models
class CoachingRequest(BaseModel):
    """Request for AI coaching advice."""
    user_id: str
    query: str = Field(..., min_length=3, max_length=500)
    context: Optional[Dict[str, Any]] = None

class CoachingResponse(BaseModel):
    """AI coach response."""
    advice: str
    memory_updated: bool = False
    relevant_history: List[str] = []
