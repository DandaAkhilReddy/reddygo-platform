"""
ReddyGo Firebase Realtime Database Client

Handles real-time data for:
- Live challenge tracking (participant positions, rankings)
- Real-time leaderboards (global, family, friends, community)
- Live notifications and updates

Uses Firebase Realtime Database for sub-second latency updates.
"""

import os
import firebase_admin
from firebase_admin import db
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import json

# Realtime Database reference
_realtime_db = None


def initialize_realtime_db():
    """Initialize Firebase Realtime Database (call once on startup)."""
    global _realtime_db

    if _realtime_db is None:
        # Get database URL from environment
        database_url = os.getenv("FIREBASE_DATABASE_URL", "https://reddygo-platform-default-rtdb.firebaseio.com")

        # Initialize with existing Firebase app
        _realtime_db = db.reference('/', url=database_url)

        print(f"🔥 Firebase Realtime Database initialized: {database_url}")

    return _realtime_db


def get_realtime_db():
    """Get Realtime Database reference."""
    if _realtime_db is None:
        return initialize_realtime_db()
    return _realtime_db


# ============================================================================
# Live Challenge Tracking
# ============================================================================

def update_participant_position(
    challenge_id: str,
    user_id: str,
    lat: float,
    lon: float,
    speed: float,
    distance: float
) -> bool:
    """
    Update participant's real-time position during challenge.

    Structure: /live_challenges/{challenge_id}/participants/{user_id}

    Args:
        challenge_id: Challenge ID
        user_id: User ID
        lat: Current latitude
        lon: Current longitude
        speed: Current speed (m/s)
        distance: Total distance covered (meters)

    Returns:
        Success boolean
    """
    ref = get_realtime_db()

    participant_data = {
        "lat": lat,
        "lon": lon,
        "speed": speed,
        "distance": distance,
        "last_update": int(datetime.utcnow().timestamp() * 1000)  # Milliseconds
    }

    ref.child(f"live_challenges/{challenge_id}/participants/{user_id}").set(participant_data)

    return True


def get_live_challenge_participants(challenge_id: str) -> Dict[str, Any]:
    """
    Get all participants' real-time positions for a challenge.

    Returns:
        Dict of user_id -> {lat, lon, speed, distance, last_update, rank}
    """
    ref = get_realtime_db()

    participants_ref = ref.child(f"live_challenges/{challenge_id}/participants")
    participants = participants_ref.get() or {}

    # Calculate rankings by distance
    participants_list = [
        {"user_id": uid, **data}
        for uid, data in participants.items()
    ]

    # Sort by distance (descending)
    participants_list.sort(key=lambda x: x.get("distance", 0), reverse=True)

    # Add rank
    for rank, participant in enumerate(participants_list, start=1):
        participant["rank"] = rank

    return {p["user_id"]: p for p in participants_list}


def listen_to_challenge(challenge_id: str, callback: Callable[[Dict], None]):
    """
    Listen to real-time updates for a challenge.

    Args:
        challenge_id: Challenge ID
        callback: Function to call when data changes

    Example:
        def on_update(participants):
            print(f"Updated rankings: {participants}")

        listen_to_challenge("challenge_123", on_update)
    """
    ref = get_realtime_db()

    def listener(event):
        if event.event_type == 'put' or event.event_type == 'patch':
            participants = get_live_challenge_participants(challenge_id)
            callback(participants)

    participants_ref = ref.child(f"live_challenges/{challenge_id}/participants")
    participants_ref.listen(listener)


def end_live_challenge(challenge_id: str) -> Dict[str, Any]:
    """
    End live challenge and get final rankings.

    Returns final participant data, then deletes from Realtime DB.
    """
    ref = get_realtime_db()

    # Get final standings
    final_standings = get_live_challenge_participants(challenge_id)

    # Delete from Realtime DB (move to Firestore for history)
    ref.child(f"live_challenges/{challenge_id}").delete()

    return final_standings


# ============================================================================
# Real-Time Leaderboards
# ============================================================================

def update_leaderboard(
    leaderboard_type: str,
    leaderboard_id: str,
    timeframe: str,
    user_id: str,
    points: int,
    user_name: str,
    user_photo_url: Optional[str] = None
):
    """
    Update leaderboard entry in real-time.

    Args:
        leaderboard_type: "global" | "family" | "friends" | "community"
        leaderboard_id: ID of the leaderboard (subscription_id for family, community_id for community, user_id for friends)
        timeframe: "weekly" | "monthly" | "all_time"
        user_id: User ID
        points: User's points
        user_name: User's name
        user_photo_url: Optional profile photo

    Structure: /leaderboards/{type}/{id}/{timeframe}/{user_id}
    """
    ref = get_realtime_db()

    leaderboard_entry = {
        "user_id": user_id,
        "name": user_name,
        "points": points,
        "photo_url": user_photo_url or "",
        "last_updated": int(datetime.utcnow().timestamp() * 1000)
    }

    path = f"leaderboards/{leaderboard_type}/{leaderboard_id}/{timeframe}/{user_id}"
    ref.child(path).set(leaderboard_entry)


def get_leaderboard(
    leaderboard_type: str,
    leaderboard_id: str,
    timeframe: str,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get leaderboard rankings.

    Returns:
        List of {user_id, name, points, photo_url, rank} sorted by points
    """
    ref = get_realtime_db()

    path = f"leaderboards/{leaderboard_type}/{leaderboard_id}/{timeframe}"
    leaderboard_ref = ref.child(path)

    # Get all entries
    entries = leaderboard_ref.order_by_child("points").limit_to_last(limit).get() or {}

    # Convert to list and sort
    leaderboard = [
        {**data, "user_id": uid}
        for uid, data in entries.items()
    ]

    # Sort by points (descending)
    leaderboard.sort(key=lambda x: x["points"], reverse=True)

    # Add rank
    for rank, entry in enumerate(leaderboard, start=1):
        entry["rank"] = rank

    return leaderboard[:limit]


def listen_to_leaderboard(
    leaderboard_type: str,
    leaderboard_id: str,
    timeframe: str,
    callback: Callable[[List[Dict]], None]
):
    """
    Listen to real-time leaderboard updates.

    Args:
        leaderboard_type: "global" | "family" | "friends" | "community"
        leaderboard_id: Leaderboard ID
        timeframe: "weekly" | "monthly" | "all_time"
        callback: Function to call when leaderboard changes
    """
    ref = get_realtime_db()

    def listener(event):
        if event.event_type == 'put' or event.event_type == 'patch':
            leaderboard = get_leaderboard(leaderboard_type, leaderboard_id, timeframe)
            callback(leaderboard)

    path = f"leaderboards/{leaderboard_type}/{leaderboard_id}/{timeframe}"
    leaderboard_ref = ref.child(path)
    leaderboard_ref.listen(listener)


def reset_weekly_leaderboards():
    """
    Reset all weekly leaderboards (call every Monday at midnight).

    Should be called by a scheduled task/cron job.
    """
    ref = get_realtime_db()

    # Get all leaderboards
    leaderboards_ref = ref.child("leaderboards")
    all_leaderboards = leaderboards_ref.get() or {}

    for leaderboard_type in all_leaderboards:
        for leaderboard_id in all_leaderboards[leaderboard_type]:
            # Delete weekly timeframe
            path = f"leaderboards/{leaderboard_type}/{leaderboard_id}/weekly"
            ref.child(path).delete()

    print("📊 Weekly leaderboards reset")


def reset_monthly_leaderboards():
    """
    Reset all monthly leaderboards (call on 1st of each month).
    """
    ref = get_realtime_db()

    leaderboards_ref = ref.child("leaderboards")
    all_leaderboards = leaderboards_ref.get() or {}

    for leaderboard_type in all_leaderboards:
        for leaderboard_id in all_leaderboards[leaderboard_type]:
            # Delete monthly timeframe
            path = f"leaderboards/{leaderboard_type}/{leaderboard_id}/monthly"
            ref.child(path).delete()

    print("📊 Monthly leaderboards reset")


# ============================================================================
# Real-Time Notifications
# ============================================================================

def send_realtime_notification(user_id: str, notification: Dict[str, Any]):
    """
    Send real-time notification to user.

    Args:
        user_id: User ID
        notification: {type, title, body, data, timestamp}
    """
    ref = get_realtime_db()

    notification_data = {
        **notification,
        "timestamp": int(datetime.utcnow().timestamp() * 1000),
        "read": False
    }

    # Add to user's notifications
    notification_ref = ref.child(f"notifications/{user_id}").push(notification_data)

    return notification_ref.key


def get_user_notifications(user_id: str, unread_only: bool = False) -> List[Dict[str, Any]]:
    """
    Get user's notifications.

    Args:
        user_id: User ID
        unread_only: Only return unread notifications
    """
    ref = get_realtime_db()

    notifications_ref = ref.child(f"notifications/{user_id}")
    notifications = notifications_ref.order_by_child("timestamp").limit_to_last(50).get() or {}

    notifications_list = [
        {**data, "id": nid}
        for nid, data in notifications.items()
    ]

    if unread_only:
        notifications_list = [n for n in notifications_list if not n.get("read", False)]

    # Sort by timestamp (descending)
    notifications_list.sort(key=lambda x: x["timestamp"], reverse=True)

    return notifications_list


def mark_notification_read(user_id: str, notification_id: str):
    """Mark notification as read."""
    ref = get_realtime_db()

    ref.child(f"notifications/{user_id}/{notification_id}/read").set(True)


# ============================================================================
# Real-Time Activity Feed
# ============================================================================

def add_activity(user_id: str, activity_type: str, activity_data: Dict[str, Any]):
    """
    Add activity to user's feed and friends' feeds.

    Args:
        user_id: User who performed the activity
        activity_type: "challenge_completed" | "badge_earned" | "milestone_reached"
        activity_data: Activity details
    """
    ref = get_realtime_db()

    activity = {
        "user_id": user_id,
        "type": activity_type,
        "data": activity_data,
        "timestamp": int(datetime.utcnow().timestamp() * 1000)
    }

    # Add to user's own feed
    ref.child(f"activity_feed/{user_id}").push(activity)

    # TODO: Add to friends' feeds (requires querying Firestore for friends list)

    return True


def get_activity_feed(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get user's activity feed.

    Returns:
        List of recent activities
    """
    ref = get_realtime_db()

    feed_ref = ref.child(f"activity_feed/{user_id}")
    activities = feed_ref.order_by_child("timestamp").limit_to_last(limit).get() or {}

    activities_list = [
        {**data, "id": aid}
        for aid, data in activities.items()
    ]

    # Sort by timestamp (descending)
    activities_list.sort(key=lambda x: x["timestamp"], reverse=True)

    return activities_list
