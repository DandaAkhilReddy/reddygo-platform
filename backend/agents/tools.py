"""
Shared Tools for ReddyGo Agents

Common functions that agents can call to interact with:
- Database (Supabase)
- External APIs (weather, geolocation)
- Search (SearXNG)
- Memory (Supermemory/Mem0)
"""

import os
import httpx
from typing import Dict, List, Any, Optional
from database import get_supabase_client
import json

# ============================================================================
# Database Tools
# ============================================================================

def query_nearby_users(lat: float, lon: float, radius_meters: int = 200) -> List[Dict[str, Any]]:
    """
    Find users within radius of a location using PostGIS.

    Args:
        lat: Latitude
        lon: Longitude
        radius_meters: Search radius in meters (default: 200m)

    Returns:
        List of nearby users with distance
    """
    supabase = get_supabase_client()

    result = supabase.rpc(
        "find_nearby_users",
        {
            "user_lat": lat,
            "user_lon": lon,
            "radius_meters": radius_meters
        }
    ).execute()

    if not result.data:
        return []

    return result.data


def create_challenge_instance(
    creator_id: str,
    title: str,
    challenge_type: str,
    zone_center: Dict[str, float],
    zone_radius_meters: int,
    rules: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a new challenge in the database.

    Args:
        creator_id: User ID of challenge creator
        title: Challenge title
        challenge_type: Type ('race', 'zone_control', 'scavenger_hunt')
        zone_center: {'lat': float, 'lon': float}
        zone_radius_meters: Challenge zone radius
        rules: Challenge rules configuration

    Returns:
        Created challenge data
    """
    from datetime import datetime, timedelta

    supabase = get_supabase_client()

    start_time = datetime.utcnow()
    end_time = start_time + timedelta(minutes=30)  # Default 30 min

    result = supabase.table("challenges").insert({
        "creator_id": creator_id,
        "title": title,
        "challenge_type": challenge_type,
        "zone_center": f"POINT({zone_center['lon']} {zone_center['lat']})",
        "zone_radius_meters": zone_radius_meters,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "max_participants": 10,
        "rules": rules,
        "status": "active"
    }).execute()

    return result.data[0] if result.data else None


def get_user_stats(user_id: str) -> Dict[str, Any]:
    """
    Get user fitness statistics.

    Args:
        user_id: User ID

    Returns:
        User stats dictionary
    """
    supabase = get_supabase_client()

    result = supabase.table("users").select("stats").eq("id", user_id).execute()

    if not result.data:
        return {}

    return result.data[0].get("stats", {})


def update_user_stats(user_id: str, stats_update: Dict[str, Any]) -> bool:
    """
    Update user fitness statistics.

    Args:
        user_id: User ID
        stats_update: Stats to update

    Returns:
        Success boolean
    """
    supabase = get_supabase_client()

    result = supabase.table("users").update({"stats": stats_update}).eq("id", user_id).execute()

    return bool(result.data)


# ============================================================================
# External API Tools
# ============================================================================

async def check_weather(lat: float, lon: float) -> Dict[str, Any]:
    """
    Get current weather conditions for a location.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Weather data (temperature, conditions, alerts)
    """
    # Using free Open-Meteo API (no key required)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()

    current = data.get("current_weather", {})

    return {
        "temperature_celsius": current.get("temperature"),
        "windspeed_kmh": current.get("windspeed"),
        "weather_code": current.get("weathercode"),
        "is_safe_for_outdoor": current.get("weathercode", 0) < 70  # No severe weather
    }


async def check_blocked_zones(lat: float, lon: float) -> Dict[str, Any]:
    """
    Check if location is in a blocked safety zone (schools, hospitals, etc.).

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        {
            "is_blocked": bool,
            "reason": str,
            "zone_type": str
        }
    """
    # TODO: Integrate with OpenStreetMap Overpass API or Google Places API
    # For now, return placeholder

    # Example implementation:
    # 1. Query OSM for nearby schools, hospitals within 100m
    # 2. If found, mark as blocked
    # 3. Check time-based restrictions (no challenges near residential zones at night)

    return {
        "is_blocked": False,
        "reason": None,
        "zone_type": None
    }


# ============================================================================
# Search Tools
# ============================================================================

async def searxng_search(query: str, categories: str = "general") -> List[Dict[str, str]]:
    """
    Privacy-preserving web search using self-hosted SearXNG.

    Args:
        query: Search query
        categories: Search categories (general, fitness, health)

    Returns:
        List of search results
    """
    searxng_url = os.getenv("SEARXNG_URL", "https://searxng.reddyfit.club")

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{searxng_url}/search",
            params={
                "q": query,
                "format": "json",
                "categories": categories
            }
        )

    data = response.json()
    results = data.get("results", [])

    # Return top 5 results
    return [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "content": r.get("content", "")[:200]  # Truncate
        }
        for r in results[:5]
    ]


# ============================================================================
# Memory Tools (Supermemory/Mem0)
# ============================================================================

async def get_user_memory(user_id: str, query: Optional[str] = None) -> List[str]:
    """
    Retrieve user memory from Supermemory (Mem0).

    Args:
        user_id: User ID
        query: Optional search query to filter memories

    Returns:
        List of relevant memories
    """
    # TODO: Integrate with actual Supermemory API
    # For now, use Supabase preferences as simple memory

    supabase = get_supabase_client()
    result = supabase.table("users").select("preferences, stats").eq("id", user_id).execute()

    if not result.data:
        return []

    user_data = result.data[0]
    preferences = user_data.get("preferences", {})
    stats = user_data.get("stats", {})

    memories = []

    # Convert preferences to memory strings
    if preferences.get("injuries"):
        memories.append(f"User has injury history: {', '.join(preferences['injuries'])}")

    if preferences.get("fitness_level"):
        memories.append(f"Fitness level: {preferences['fitness_level']}")

    if stats.get("challenges_completed", 0) > 0:
        memories.append(f"Completed {stats['challenges_completed']} challenges")

    if stats.get("favorite_challenge_type"):
        memories.append(f"Prefers {stats['favorite_challenge_type']} challenges")

    return memories


async def update_user_memory(user_id: str, memory: str) -> bool:
    """
    Store a new memory for the user.

    Args:
        user_id: User ID
        memory: Memory to store (e.g., "User mentioned knee pain during squats")

    Returns:
        Success boolean
    """
    # TODO: Integrate with actual Supermemory API
    # For now, append to preferences

    supabase = get_supabase_client()

    # Get current preferences
    result = supabase.table("users").select("preferences").eq("id", user_id).execute()

    if not result.data:
        return False

    preferences = result.data[0].get("preferences", {})

    # Add to memories array
    if "memories" not in preferences:
        preferences["memories"] = []

    preferences["memories"].append({
        "text": memory,
        "timestamp": "utcnow()"  # Would be actual datetime
    })

    # Keep only last 50 memories
    preferences["memories"] = preferences["memories"][-50:]

    # Update
    update_result = supabase.table("users").update({"preferences": preferences}).eq("id", user_id).execute()

    return bool(update_result.data)


# ============================================================================
# Notification Tools
# ============================================================================

async def send_push_notification(user_id: str, title: str, body: str, data: Optional[Dict] = None) -> bool:
    """
    Send push notification to user's mobile device.

    Args:
        user_id: User ID
        title: Notification title
        body: Notification body
        data: Optional extra data

    Returns:
        Success boolean
    """
    # TODO: Integrate with Firebase Cloud Messaging or similar
    # For now, log notification

    print(f"📱 NOTIFICATION to {user_id}: {title} - {body}")

    return True


# ============================================================================
# Workout Analysis Tools
# ============================================================================

def calculate_workout_metrics(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate workout metrics from sensor data.

    Args:
        sensor_data: GPS tracks, accelerometer data, etc.

    Returns:
        Calculated metrics (distance, calories, duration, etc.)
    """
    import numpy as np

    metrics = {}

    # Distance from GPS track
    if "gps_track" in sensor_data:
        track = sensor_data["gps_track"]

        total_distance = 0.0
        for i in range(1, len(track)):
            # Simplified haversine
            lat1, lon1 = track[i-1]["lat"], track[i-1]["lon"]
            lat2, lon2 = track[i]["lat"], track[i]["lon"]

            # Rough distance calculation (meters)
            distance = ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5 * 111000
            total_distance += distance

        metrics["distance_meters"] = total_distance
        metrics["distance_km"] = total_distance / 1000

    # Calories (very rough estimate: 1 km = 60 calories)
    if "distance_km" in metrics:
        metrics["calories_burned"] = int(metrics["distance_km"] * 60)

    # Duration
    if "gps_track" in sensor_data and len(sensor_data["gps_track"]) > 1:
        start_time = sensor_data["gps_track"][0].get("timestamp", 0)
        end_time = sensor_data["gps_track"][-1].get("timestamp", 0)
        metrics["duration_minutes"] = int((end_time - start_time) / 60)

    return metrics
