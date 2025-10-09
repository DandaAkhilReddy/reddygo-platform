"""
ReddyGo Firebase Client

Firebase Firestore database and authentication.
Replaces Supabase with Firebase Admin SDK.
"""

import os
import firebase_admin
from firebase_admin import credentials, firestore, auth
from typing import Dict, Any, List, Optional
from datetime import datetime
import math

# Initialize Firebase Admin SDK
_firebase_app = None
_firestore_client = None


def initialize_firebase():
    """Initialize Firebase Admin SDK (call once on startup)."""
    global _firebase_app, _firestore_client

    if _firebase_app is None:
        # Get Firebase credentials from environment
        firebase_creds = {
            "type": "service_account",
            "project_id": os.getenv("FIREBASE_PROJECT_ID"),
            "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
            "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n"),
            "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
            "client_id": os.getenv("FIREBASE_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv("FIREBASE_CERT_URL")
        }

        cred = credentials.Certificate(firebase_creds)
        _firebase_app = firebase_admin.initialize_app(cred)
        _firestore_client = firestore.client()

        print("🔥 Firebase initialized successfully")

    return _firestore_client


def get_firestore_client():
    """Get Firestore client instance."""
    if _firestore_client is None:
        return initialize_firebase()
    return _firestore_client


def verify_firebase_token(id_token: str) -> Dict[str, Any]:
    """
    Verify Firebase ID token from client.

    Args:
        id_token: Firebase ID token from client

    Returns:
        Decoded token with user info
    """
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        raise ValueError(f"Invalid Firebase token: {str(e)}")


# ============================================================================
# Firestore Collections
# ============================================================================

"""
Firestore Structure:

/users/{user_id}
  - email: string
  - name: string
  - created_at: timestamp
  - location: geopoint
  - stats: map
  - preferences: map

/challenges/{challenge_id}
  - creator_id: string
  - title: string
  - challenge_type: string
  - zone_center: geopoint
  - zone_radius_meters: number
  - start_time: timestamp
  - end_time: timestamp
  - max_participants: number
  - rules: map
  - status: string
  - created_at: timestamp

/challenges/{challenge_id}/participants/{user_id}
  - joined_at: timestamp
  - status: string
  - stats: map

/gps_tracks/{track_id}
  - user_id: string
  - challenge_id: string
  - track_data: array
  - sensor_data: map
  - validation_score: number
  - created_at: timestamp

/encrypted_data/{data_id}
  - user_id: string
  - data_type: string
  - encrypted_data: string
  - encryption_metadata: map
  - created_at: timestamp

/user_rewards/{user_id}
  - points_balance: number
  - lifetime_points: number
  - current_tier: string
  - current_streak: number
  - longest_streak: number
  - badges: array
  - perks: array
  - last_activity_date: timestamp
  - created_at: timestamp
"""


# ============================================================================
# Geospatial Helper Functions (replacing PostGIS)
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points in meters (Haversine formula).

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def find_nearby_challenges(lat: float, lon: float, radius_meters: int = 5000) -> List[Dict[str, Any]]:
    """
    Find challenges within radius of a location.

    Firebase doesn't have native geospatial queries like PostGIS,
    so we use bounding box + Haversine filtering.

    Args:
        lat: Latitude
        lon: Longitude
        radius_meters: Search radius

    Returns:
        List of nearby challenges with distance
    """
    db = get_firestore_client()

    # Calculate bounding box (rough approximation)
    lat_delta = radius_meters / 111000  # 1 degree lat ≈ 111km
    lon_delta = radius_meters / (111000 * math.cos(math.radians(lat)))

    # Query challenges within bounding box
    challenges_ref = db.collection('challenges')
    query = challenges_ref.where('status', '==', 'active')

    # Get all active challenges (optimize later with geohashing)
    challenges = query.stream()

    nearby = []
    for challenge_doc in challenges:
        challenge_data = challenge_doc.to_dict()
        challenge_data['id'] = challenge_doc.id

        # Get challenge location
        zone_center = challenge_data.get('zone_center')
        if not zone_center:
            continue

        challenge_lat = zone_center.latitude
        challenge_lon = zone_center.longitude

        # Calculate distance
        distance = haversine_distance(lat, lon, challenge_lat, challenge_lon)

        # Filter by radius
        if distance <= radius_meters:
            nearby.append({
                'challenge_id': challenge_data['id'],
                'title': challenge_data.get('title'),
                'distance_meters': distance,
                **challenge_data
            })

    # Sort by distance
    nearby.sort(key=lambda x: x['distance_meters'])

    return nearby


def find_nearby_users(lat: float, lon: float, radius_meters: int = 200) -> List[Dict[str, Any]]:
    """
    Find users within radius of a location.

    Args:
        lat: Latitude
        lon: Longitude
        radius_meters: Search radius

    Returns:
        List of nearby users with distance
    """
    db = get_firestore_client()

    # Get all users with location set
    users_ref = db.collection('users')
    users = users_ref.stream()

    nearby = []
    for user_doc in users:
        user_data = user_doc.to_dict()
        user_data['id'] = user_doc.id

        # Get user location
        location = user_data.get('location')
        if not location:
            continue

        user_lat = location.latitude
        user_lon = location.longitude

        # Calculate distance
        distance = haversine_distance(lat, lon, user_lat, user_lon)

        # Filter by radius
        if distance <= radius_meters:
            nearby.append({
                'user_id': user_data['id'],
                'name': user_data.get('name'),
                'distance_meters': distance
            })

    # Sort by distance
    nearby.sort(key=lambda x: x['distance_meters'])

    return nearby


# ============================================================================
# Firestore Helper Functions
# ============================================================================

def create_user(email: str, name: str, uid: str) -> Dict[str, Any]:
    """
    Create a new user in Firestore.

    Args:
        email: User email
        name: User name
        uid: Firebase Auth UID

    Returns:
        Created user data
    """
    db = get_firestore_client()

    user_data = {
        'email': email,
        'name': name,
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

    db.collection('users').document(uid).set(user_data)

    user_data['id'] = uid
    return user_data


def get_user(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user by ID."""
    db = get_firestore_client()
    user_doc = db.collection('users').document(user_id).get()

    if not user_doc.exists:
        return None

    user_data = user_doc.to_dict()
    user_data['id'] = user_doc.id
    return user_data


def update_user(user_id: str, updates: Dict[str, Any]) -> bool:
    """Update user data."""
    db = get_firestore_client()
    db.collection('users').document(user_id).update(updates)
    return True


def get_user_stats(user_id: str) -> Dict[str, Any]:
    """Get user statistics."""
    user = get_user(user_id)
    if not user:
        return {}
    return user.get('stats', {})


def update_user_stats(user_id: str, stats_update: Dict[str, Any]) -> bool:
    """Update user stats."""
    db = get_firestore_client()
    db.collection('users').document(user_id).update({'stats': stats_update})
    return True
