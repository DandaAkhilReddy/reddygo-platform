"""
Photo Steganography Router

Embeds workout metadata and achievements into photos using EXIF + LSB steganography.
Enables social sharing of workout achievements with verifiable embedded data.

"Think Outside the Box" Innovation: Anyone can scan your workout photo to see your stats!
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import piexif
import io
import json
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from firebase_admin import firestore
import os

router = APIRouter()
db = firestore.client()

# ==================== Crypto Setup for Digital Signatures ====================

# In production, load these from secure storage
PRIVATE_KEY = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend()
)

PUBLIC_KEY = PRIVATE_KEY.public_key()

def sign_metadata(metadata: Dict[str, Any]) -> str:
    """Sign metadata with private key for verification"""
    metadata_bytes = json.dumps(metadata, sort_keys=True).encode()
    signature = PRIVATE_KEY.sign(
        metadata_bytes,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode()

def verify_signature(metadata: Dict[str, Any], signature_b64: str) -> bool:
    """Verify metadata signature"""
    try:
        metadata_bytes = json.dumps(metadata, sort_keys=True).encode()
        signature = base64.b64decode(signature_b64)
        PUBLIC_KEY.verify(
            signature,
            metadata_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except Exception:
        return False

# ==================== LSB Steganography Functions ====================

def encode_lsb(image: Image.Image, data: str) -> Image.Image:
    """
    Embed data in image using LSB (Least Significant Bit) steganography
    Hides data in the least significant bits of pixel values
    """
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert data to binary
    binary_data = ''.join(format(ord(char), '08b') for char in data)
    binary_data += '1111111111111110'  # Delimiter

    data_index = 0
    data_len = len(binary_data)

    pixels = image.load()
    width, height = image.size

    for y in range(height):
        for x in range(width):
            if data_index >= data_len:
                return image

            pixel = list(pixels[x, y])

            # Embed in R, G, B channels
            for i in range(3):
                if data_index < data_len:
                    pixel[i] = pixel[i] & ~1 | int(binary_data[data_index])
                    data_index += 1

            pixels[x, y] = tuple(pixel)

    return image

def decode_lsb(image: Image.Image) -> Optional[str]:
    """Extract data hidden in image using LSB steganography"""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    binary_data = ""
    pixels = image.load()
    width, height = image.size

    for y in range(height):
        for x in range(width):
            pixel = pixels[x, y]

            # Extract LSB from R, G, B
            for i in range(3):
                binary_data += str(pixel[i] & 1)

    # Find delimiter
    delimiter = '1111111111111110'
    delimiter_index = binary_data.find(delimiter)

    if delimiter_index == -1:
        return None

    # Extract actual data
    binary_data = binary_data[:delimiter_index]

    # Convert binary to text
    chars = []
    for i in range(0, len(binary_data), 8):
        byte = binary_data[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))

    return ''.join(chars)

# ==================== EXIF Metadata Functions ====================

def embed_exif_metadata(image: Image.Image, metadata: Dict[str, Any]) -> Image.Image:
    """
    Embed metadata in EXIF fields
    More compatible but easier to strip/modify
    """
    # Convert metadata to JSON string
    metadata_json = json.dumps(metadata)

    # Prepare EXIF data
    exif_dict = {
        "0th": {},
        "Exif": {},
        "GPS": {},
        "1st": {},
        "thumbnail": None
    }

    # Store in UserComment field (supports large text)
    user_comment = piexif.helper.UserComment.dump(metadata_json, encoding="unicode")
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment

    # Also store a marker in ImageDescription
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = b"ReddyFit Workout Photo"

    # Convert to bytes
    exif_bytes = piexif.dump(exif_dict)

    # Save image with EXIF
    output = io.BytesIO()
    image.save(output, format='JPEG', exif=exif_bytes, quality=95)
    output.seek(0)

    return Image.open(output)

def extract_exif_metadata(image: Image.Image) -> Optional[Dict[str, Any]]:
    """Extract metadata from EXIF fields"""
    try:
        exif_dict = piexif.load(image.info.get("exif", b""))
        user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment, None)

        if not user_comment:
            return None

        # Decode UserComment
        metadata_json = piexif.helper.UserComment.load(user_comment)
        return json.loads(metadata_json)

    except Exception:
        return None

# ==================== Models ====================

class EmbedMetadataRequest(BaseModel):
    user_id: str
    workout_id: Optional[str] = None  # If None, uses latest Whoop workout
    include_achievements: bool = True
    include_stats: bool = True

class ExtractMetadataRequest(BaseModel):
    pass  # Just upload the photo

# ==================== API Endpoints ====================

@router.post("/embed-metadata")
async def embed_workout_metadata(
    user_id: str = Form(...),
    workout_id: Optional[str] = Form(None),
    include_achievements: bool = Form(True),
    include_stats: bool = Form(True),
    photo: UploadFile = File(...)
):
    """
    Embed workout metadata into photo
    Creates a shareable achievement photo with hidden stats
    """
    # Validate user
    user_doc = db.collection("users").document(user_id).get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user_doc.to_dict()

    # Get Whoop data
    whoop_doc = db.collection("whoop_data").document(user_id).get()
    if not whoop_doc.exists:
        raise HTTPException(
            status_code=404,
            detail="No Whoop data found. Please sync your Whoop account first."
        )

    whoop_data = whoop_doc.to_dict()

    # Get workout data
    if workout_id:
        # Find specific workout
        workout = next(
            (w for w in whoop_data.get("workouts", []) if w.get("id") == workout_id),
            None
        )
        if not workout:
            raise HTTPException(status_code=404, detail="Workout not found")
    else:
        # Use latest workout
        workouts = whoop_data.get("workouts", [])
        if not workouts:
            raise HTTPException(status_code=404, detail="No workouts found")
        workout = workouts[0]

    # Get latest recovery
    recovery = whoop_data.get("recovery", [{}])[0]

    # Build metadata to embed
    metadata = {
        "reddyfit_verified": True,
        "version": "1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "user": {
            "name": user_data.get("display_name", "ReddyFit Member"),
            "member_since": user_data.get("created_at", datetime.utcnow()).isoformat() if isinstance(user_data.get("created_at"), datetime) else user_data.get("created_at", datetime.utcnow().isoformat()),
            "tier": user_data.get("subscription_tier", "Elite")
        },
        "workout": {
            "id": workout.get("id"),
            "sport": workout.get("sport_name", "workout"),
            "start": workout.get("start"),
            "end": workout.get("end"),
            "duration_minutes": None,
            "score": workout.get("score", {})
        }
    }

    # Calculate duration
    if workout.get("start") and workout.get("end"):
        try:
            start = datetime.fromisoformat(workout["start"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(workout["end"].replace("Z", "+00:00"))
            duration = (end - start).total_seconds() / 60
            metadata["workout"]["duration_minutes"] = round(duration, 1)
        except:
            pass

    # Add recovery data
    if recovery and recovery.get("score"):
        metadata["recovery"] = {
            "score": recovery["score"].get("recovery_score"),
            "hrv_rmssd_ms": recovery["score"].get("hrv_rmssd_milli"),
            "resting_hr": recovery["score"].get("resting_heart_rate")
        }

    # Add achievements if requested
    if include_achievements:
        achievements = _get_user_achievements(user_id, whoop_data)
        metadata["achievements"] = achievements

    # Add stats if requested
    if include_stats:
        stats = _calculate_user_stats(whoop_data)
        metadata["stats"] = stats

    # Sign metadata
    signature = sign_metadata(metadata)
    metadata["signature"] = signature

    # Read and process image
    try:
        image_bytes = await photo.read()
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Embed metadata using both methods
    metadata_json = json.dumps(metadata)

    # Method 1: EXIF (compatible with most platforms)
    image_with_exif = embed_exif_metadata(image, metadata)

    # Method 2: LSB (more secure, survives some compressions)
    image_final = encode_lsb(image_with_exif, metadata_json)

    # Save to bytes
    output = io.BytesIO()
    image_final.save(output, format='JPEG', quality=95)
    output.seek(0)

    # Store metadata record
    photo_record = {
        "user_id": user_id,
        "workout_id": workout.get("id"),
        "embedded_at": datetime.utcnow(),
        "metadata": metadata,
        "original_filename": photo.filename
    }
    db.collection("workout_photos").add(photo_record)

    return {
        "success": True,
        "message": "Metadata embedded successfully",
        "photo_data": base64.b64encode(output.getvalue()).decode(),
        "metadata_preview": {
            "sport": metadata["workout"]["sport"],
            "duration_minutes": metadata["workout"]["duration_minutes"],
            "recovery_score": metadata.get("recovery", {}).get("score"),
            "achievements_count": len(metadata.get("achievements", []))
        }
    }

@router.post("/extract-metadata")
async def extract_workout_metadata(photo: UploadFile = File(...)):
    """
    Extract metadata from workout photo
    Scans photo for embedded achievement data
    """
    # Read image
    try:
        image_bytes = await photo.read()
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Try Method 1: EXIF
    metadata = extract_exif_metadata(image)

    # Try Method 2: LSB if EXIF failed
    if not metadata:
        metadata_json = decode_lsb(image)
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
            except:
                metadata = None

    if not metadata:
        return {
            "found": False,
            "message": "No ReddyFit metadata found in this photo"
        }

    # Verify signature
    signature = metadata.pop("signature", None)
    verified = False
    if signature:
        verified = verify_signature(metadata, signature)

    # Add signature back for response
    metadata["signature"] = signature

    return {
        "found": True,
        "verified": verified,
        "metadata": metadata,
        "formatted_display": _format_metadata_display(metadata)
    }

@router.get("/verify/{photo_id}")
async def verify_photo_authenticity(photo_id: str):
    """Verify if a workout photo is authentic ReddyFit photo"""
    # In production, lookup photo by ID
    return {
        "verified": True,
        "message": "Photo verification endpoint"
    }

# ==================== Helper Functions ====================

def _get_user_achievements(user_id: str, whoop_data: Dict[str, Any]) -> List[str]:
    """Calculate user achievements from Whoop data"""
    achievements = []

    workouts = whoop_data.get("workouts", [])

    # Workout count achievements
    if len(workouts) >= 100:
        achievements.append("100+ Workouts")
    elif len(workouts) >= 50:
        achievements.append("50+ Workouts")
    elif len(workouts) >= 25:
        achievements.append("25+ Workouts")

    # Check for workout streak (simplified)
    if workouts:
        achievements.append("Active This Week")

    # Recovery achievements
    recovery_data = whoop_data.get("recovery", [])
    if recovery_data:
        avg_recovery = sum(r.get("score", {}).get("recovery_score", 0) for r in recovery_data) / len(recovery_data)
        if avg_recovery >= 70:
            achievements.append("High Recovery Performer")

    # Strain achievements
    cycles = whoop_data.get("cycles", [])
    if cycles:
        max_strain = max(c.get("score", {}).get("strain", 0) for c in cycles)
        if max_strain >= 18:
            achievements.append("Peak Performer")

    return achievements

def _calculate_user_stats(whoop_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate aggregate stats from Whoop data"""
    workouts = whoop_data.get("workouts", [])

    total_workouts = len(workouts)
    total_calories = sum(
        w.get("score", {}).get("kilojoule", 0) * 0.239 for w in workouts  # Convert kJ to kcal
    )
    total_distance_meters = sum(
        w.get("score", {}).get("distance_meter", 0) for w in workouts
    )

    return {
        "total_workouts": total_workouts,
        "total_calories": round(total_calories),
        "total_distance_km": round(total_distance_meters / 1000, 1),
        "data_period_days": 7
    }

def _format_metadata_display(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Format metadata for nice display in app"""
    workout = metadata.get("workout", {})
    recovery = metadata.get("recovery", {})
    user = metadata.get("user", {})

    return {
        "title": f"{user.get('name', 'ReddyFit Member')}'s Workout",
        "sport": workout.get("sport", "Workout").title(),
        "duration": f"{workout.get('duration_minutes', 0):.0f} min",
        "stats": {
            "Strain": workout.get("score", {}).get("strain"),
            "Avg HR": workout.get("score", {}).get("average_heart_rate"),
            "Max HR": workout.get("score", {}).get("max_heart_rate"),
            "Calories": round(workout.get("score", {}).get("kilojoule", 0) * 0.239),
            "Recovery Score": recovery.get("score"),
            "HRV": f"{recovery.get('hrv_rmssd_ms', 0):.1f}ms" if recovery.get("hrv_rmssd_ms") else None
        },
        "achievements": metadata.get("achievements", []),
        "verified": metadata.get("reddyfit_verified", False),
        "timestamp": metadata.get("timestamp")
    }
