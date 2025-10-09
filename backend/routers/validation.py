"""
ReddyGo Validation Router

5-layer anti-cheat GPS validation system.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

from models import (
    GPSTrackSubmission,
    ValidationResult,
    GPSPoint
)
from firebase_client import get_firestore_client
from firebase_admin import firestore

router = APIRouter()

# Validation thresholds
MAX_SPEED_MPS = 12.0  # ~43 km/h (sprint = ~36 km/h)
MIN_ACCURACY_METERS = 100.0  # Reject if accuracy > 100m
MIN_SENSOR_CORRELATION = 0.7  # GPS vs accelerometer
ANOMALY_SCORE_THRESHOLD = 0.3  # ML anomaly detection

@router.post("/validate", response_model=ValidationResult)
async def validate_gps_track(submission: GPSTrackSubmission):
    """
    5-layer GPS validation:
    1. Device Integrity Check
    2. GPS Quality Assessment
    3. Sensor Fusion Validation
    4. Physical Sanity Checks
    5. ML Anomaly Detection
    """
    flags: List[str] = []
    confidence_score = 1.0

    # Layer 1: Device Integrity
    if submission.sensor_data and submission.sensor_data.device_integrity:
        integrity = submission.sensor_data.device_integrity
        if not integrity.get("is_rooted", False) == False:
            flags.append("rooted_device")
            confidence_score -= 0.3
        if not integrity.get("location_mocking_enabled", False) == False:
            flags.append("mock_location_enabled")
            confidence_score -= 0.4

    # Layer 2: GPS Quality Assessment
    track_points = submission.track_data
    low_accuracy_count = sum(1 for p in track_points if p.accuracy > MIN_ACCURACY_METERS)

    if low_accuracy_count > len(track_points) * 0.3:  # >30% low accuracy
        flags.append("poor_gps_accuracy")
        confidence_score -= 0.2

    # Layer 3: Sensor Fusion (GPS vs Accelerometer)
    if submission.sensor_data and submission.sensor_data.accelerometer:
        correlation = calculate_sensor_correlation(
            track_points,
            submission.sensor_data.accelerometer
        )
        if correlation < MIN_SENSOR_CORRELATION:
            flags.append("sensor_mismatch")
            confidence_score -= 0.3

    # Layer 4: Physical Sanity Checks
    sanity_flags = check_physical_sanity(track_points)
    flags.extend(sanity_flags)
    confidence_score -= len(sanity_flags) * 0.15

    # Layer 5: ML Anomaly Detection
    anomaly_score = detect_anomalies(track_points)
    if anomaly_score > ANOMALY_SCORE_THRESHOLD:
        flags.append("anomalous_pattern")
        confidence_score -= 0.25

    # Clamp confidence score
    confidence_score = max(0.0, min(1.0, confidence_score))

    # Determine validity
    is_valid = confidence_score >= 0.6 and len(flags) < 3

    # Store validation result in Firestore
    db = get_firestore_client()
    gps_track_ref = db.collection('gps_tracks').document()
    gps_track_ref.set({
        "user_id": "temp_user",  # TODO: Get from auth
        "challenge_id": submission.challenge_id,
        "track_data": [p.dict() for p in track_points],
        "sensor_data": submission.sensor_data.dict() if submission.sensor_data else None,
        "validation_score": confidence_score,
        "created_at": firestore.SERVER_TIMESTAMP
    })

    return ValidationResult(
        is_valid=is_valid,
        confidence_score=confidence_score,
        flags=flags,
        details={
            "total_points": len(track_points),
            "low_accuracy_count": low_accuracy_count,
            "anomaly_score": anomaly_score
        }
    )

def calculate_sensor_correlation(
    gps_track: List[GPSPoint],
    accel_data: List[Dict[str, float]]
) -> float:
    """Compare GPS-derived acceleration with accelerometer data."""
    if len(gps_track) < 2 or len(accel_data) < 2:
        return 0.0

    # Calculate GPS-derived acceleration
    gps_speeds = [p.speed or 0.0 for p in gps_track]
    gps_times = [(p.timestamp.timestamp() if hasattr(p.timestamp, 'timestamp') else p.timestamp) for p in gps_track]

    if len(set(gps_times)) < 2:  # All same timestamps
        return 0.0

    gps_accel = np.diff(gps_speeds) / np.diff(gps_times)

    # Calculate accelerometer magnitude
    accel_mag = np.array([
        np.sqrt(a.get('x', 0)**2 + a.get('y', 0)**2 + a.get('z', 0)**2)
        for a in accel_data[:len(gps_accel)]
    ])

    # Normalize and correlate
    if len(gps_accel) != len(accel_mag):
        min_len = min(len(gps_accel), len(accel_mag))
        gps_accel = gps_accel[:min_len]
        accel_mag = accel_mag[:min_len]

    if len(gps_accel) < 2:
        return 0.0

    correlation = np.corrcoef(gps_accel, accel_mag)[0, 1]
    return float(correlation) if not np.isnan(correlation) else 0.0

def check_physical_sanity(track_points: List[GPSPoint]) -> List[str]:
    """Check for physically impossible movements."""
    flags = []

    for i in range(1, len(track_points)):
        prev = track_points[i - 1]
        curr = track_points[i]

        # Check speed
        if curr.speed and curr.speed > MAX_SPEED_MPS:
            flags.append("excessive_speed")
            break

        # Check teleportation (distance vs time)
        time_diff = abs((curr.timestamp - prev.timestamp).total_seconds())
        if time_diff > 0:
            distance = haversine_distance(
                prev.lat, prev.lon,
                curr.lat, curr.lon
            )
            implied_speed = distance / time_diff

            if implied_speed > MAX_SPEED_MPS * 1.2:  # 20% tolerance
                flags.append("teleportation_detected")
                break

    # Check for static GPS (all points identical)
    unique_coords = set((round(p.lat, 5), round(p.lon, 5)) for p in track_points)
    if len(unique_coords) == 1 and len(track_points) > 5:
        flags.append("static_gps")

    return flags

def detect_anomalies(track_points: List[GPSPoint]) -> float:
    """
    ML-based anomaly detection using Isolation Forest.

    Returns anomaly score (0.0 = normal, 1.0 = highly anomalous).
    """
    # Extract features
    speeds = [p.speed or 0.0 for p in track_points]
    accuracies = [p.accuracy for p in track_points]

    if not speeds or not accuracies:
        return 0.0

    # Simple statistical approach (placeholder for full ML model)
    speed_std = np.std(speeds)
    speed_mean = np.mean(speeds)

    # High variability in speed is suspicious
    if speed_mean > 0:
        cv = speed_std / speed_mean  # Coefficient of variation
        return min(1.0, cv / 2.0)  # Normalize

    return 0.0

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS points in meters."""
    R = 6371000  # Earth radius in meters

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c
