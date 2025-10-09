"""
ReddyGo Encryption Router

Handles key exchange and encrypted data operations.
Enables end-to-end encryption for sensitive user data.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from encryption.e2ee import E2EEManager
from encryption.key_management import KeyManager
from database import get_supabase_client
import base64

router = APIRouter()


# Request/Response Models
class KeyExchangeRequest(BaseModel):
    """Request to exchange public keys with another user."""
    user_id: str
    other_user_id: str


class KeyExchangeResponse(BaseModel):
    """Public key exchange response."""
    user_id: str
    public_key: str  # Base64-encoded


class EncryptedDataRequest(BaseModel):
    """Upload encrypted data."""
    user_id: str
    data_type: str  # gps_track, sensor_data, message, etc.
    encrypted: str  # Base64-encoded ciphertext
    nonce: str  # Base64-encoded nonce
    metadata: Optional[Dict[str, Any]] = None


class DecryptRequest(BaseModel):
    """Request to retrieve encrypted data for decryption."""
    user_id: str
    data_id: str


@router.post("/exchange-keys", response_model=KeyExchangeResponse)
async def exchange_public_keys(request: KeyExchangeRequest):
    """
    Exchange public keys for shared challenge encryption.

    When users participate in the same challenge, they need each other's
    public keys to encrypt/decrypt shared data.

    Returns:
        Other user's public key (for X25519 key exchange)
    """
    supabase = get_supabase_client()

    # Get other user's public key
    result = supabase.table("users").select("public_key").eq("id", request.other_user_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="User not found")

    public_key = result.data[0].get("public_key")

    if not public_key:
        raise HTTPException(status_code=404, detail="User has not set up encryption")

    return KeyExchangeResponse(
        user_id=request.other_user_id,
        public_key=base64.b64encode(public_key).decode()
    )


@router.post("/setup-keys")
async def setup_user_keys(user_id: str, password: str):
    """
    Generate and store encryption keys for a new user.

    Creates:
    - X25519 keypair for key exchange
    - Encrypts private key with password
    - Stores public key in database
    - Returns public key

    The private key is encrypted with the user's password and stored.
    The user must remember their password to decrypt their data.
    """
    try:
        # Generate keypair
        keypair = E2EEManager.generate_keypair()

        # Encrypt private key with password
        encrypted_private_key = KeyManager.encrypt_private_key(
            private_key=keypair["private_key"],
            password=password,
            user_id=user_id
        )

        # Store in database
        supabase = get_supabase_client()
        supabase.table("users").update({
            "public_key": keypair["public_key"],
            "encrypted_private_key": encrypted_private_key
        }).eq("id", user_id).execute()

        return {
            "success": True,
            "public_key": base64.b64encode(keypair["public_key"]).decode(),
            "message": "Encryption keys set up successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Key setup failed: {str(e)}")


@router.post("/upload-encrypted")
async def upload_encrypted_data(request: EncryptedDataRequest):
    """
    Upload encrypted data (GPS track, sensor data, etc.).

    Server stores encrypted blob without ability to decrypt.
    Only the user with the encryption key can read the data.
    """
    supabase = get_supabase_client()

    # Store encrypted data
    result = supabase.table("encrypted_data").insert({
        "user_id": request.user_id,
        "data_type": request.data_type,
        "encrypted_data": request.encrypted,
        "encryption_metadata": {
            "nonce": request.nonce,
            "algorithm": "XChaCha20-Poly1305",
            "version": "1.0",
            **(request.metadata or {})
        }
    }).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to store encrypted data")

    return {
        "success": True,
        "data_id": result.data[0]["id"],
        "stored_at": result.data[0].get("created_at")
    }


@router.get("/retrieve-encrypted/{data_id}")
async def retrieve_encrypted_data(data_id: str, user_id: str):
    """
    Retrieve encrypted data for client-side decryption.

    Returns the encrypted blob and metadata.
    Client must decrypt with their key.
    """
    supabase = get_supabase_client()

    # Get encrypted data
    result = supabase.table("encrypted_data").select("*").eq("id", data_id).eq("user_id", user_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Encrypted data not found")

    data = result.data[0]

    return {
        "data_id": data["id"],
        "data_type": data["data_type"],
        "encrypted": data["encrypted_data"],
        "metadata": data.get("encryption_metadata", {}),
        "created_at": data.get("created_at")
    }


@router.post("/derive-challenge-key")
async def derive_challenge_key(user_id: str, challenge_id: str, password: str):
    """
    Derive challenge-specific encryption key.

    Uses user's master key (derived from password) to create
    a unique key for each challenge.

    This allows:
    - Separate encryption for each challenge
    - Key rotation per challenge
    - Granular access control
    """
    try:
        # Derive master key from password
        master_key = KeyManager.derive_master_key(password, user_id)

        # Derive challenge key
        challenge_key = KeyManager.derive_challenge_key(master_key, challenge_id)

        # Return as base64 (client will use for encryption/decryption)
        return {
            "challenge_id": challenge_id,
            "challenge_key": base64.b64encode(challenge_key).decode(),
            "algorithm": "XChaCha20-Poly1305",
            "expires_in": 3600  # Key valid for 1 hour (client should cache)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Key derivation failed: {str(e)}")


@router.post("/rotate-password")
async def rotate_password(user_id: str, old_password: str, new_password: str):
    """
    Rotate user's password (master key).

    This requires re-encrypting all user data with the new key.
    May take time for users with lots of encrypted data.
    """
    supabase = get_supabase_client()

    try:
        # Get all encrypted data for user
        result = supabase.table("encrypted_data").select("*").eq("user_id", user_id).execute()

        encrypted_data_list = result.data if result.data else []

        # Rotate keys
        rotation_result = KeyManager.rotate_master_key(
            old_password=old_password,
            new_password=new_password,
            user_id=user_id,
            encrypted_data_list=encrypted_data_list
        )

        if not rotation_result["success"]:
            raise HTTPException(status_code=400, detail=rotation_result.get("error", "Rotation failed"))

        # Update all encrypted data in database
        for i, encrypted_item in enumerate(rotation_result["new_encrypted_data"]):
            data_id = encrypted_data_list[i]["id"]
            supabase.table("encrypted_data").update({
                "encrypted_data": encrypted_item["encrypted"],
                "encryption_metadata": {
                    "nonce": encrypted_item["nonce"],
                    "algorithm": encrypted_item["algorithm"],
                    "version": encrypted_item["version"]
                }
            }).eq("id", data_id).execute()

        return {
            "success": True,
            "re_encrypted_count": rotation_result["re_encrypted_count"],
            "message": "Password rotated successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Password rotation failed: {str(e)}")


@router.get("/encryption-status/{user_id}")
async def get_encryption_status(user_id: str):
    """
    Check if user has set up encryption.

    Returns:
    - Whether user has keys configured
    - Number of encrypted data items
    - Encryption version
    """
    supabase = get_supabase_client()

    # Check if user has public key
    user_result = supabase.table("users").select("public_key, encrypted_private_key").eq("id", user_id).execute()

    if not user_result.data:
        raise HTTPException(status_code=404, detail="User not found")

    has_keys = user_result.data[0].get("public_key") is not None

    # Count encrypted data
    encrypted_count = 0
    if has_keys:
        data_result = supabase.table("encrypted_data").select("id").eq("user_id", user_id).execute()
        encrypted_count = len(data_result.data) if data_result.data else 0

    return {
        "user_id": user_id,
        "encryption_enabled": has_keys,
        "encrypted_data_count": encrypted_count,
        "encryption_version": "1.0",
        "algorithm": "XChaCha20-Poly1305",
        "kdf": "Argon2id"
    }
