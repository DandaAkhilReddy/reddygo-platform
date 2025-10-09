"""
Key Management for E2EE

Handles password-based key derivation using Argon2.
Secure key storage and rotation.
"""

import nacl.pwhash
import nacl.utils
import nacl.secret
import base64
from typing import Dict, Any, Optional
import os


class KeyManager:
    """
    Manages encryption keys for users.

    - Master key derived from password (never stored)
    - Per-challenge keys derived from master key
    - Secure key rotation
    """

    # Argon2 parameters (secure but not too slow)
    OPSLIMIT = nacl.pwhash.argon2id.OPSLIMIT_MODERATE  # 3 iterations
    MEMLIMIT = nacl.pwhash.argon2id.MEMLIMIT_MODERATE  # 64 MB

    @staticmethod
    def derive_master_key(password: str, user_id: str, device_secret: Optional[str] = None) -> bytes:
        """
        Derive master encryption key from password.

        Uses Argon2id (memory-hard, GPU-resistant).

        Args:
            password: User's password
            user_id: User ID (used as salt)
            device_secret: Optional device-specific secret

        Returns:
            32-byte master key
        """
        # Combine password with device secret if provided
        password_input = password
        if device_secret:
            password_input = f"{password}:{device_secret}"

        # Use user_id as deterministic salt
        # In production: store random salt in database
        salt = nacl.pwhash.argon2id.kdf(
            size=nacl.pwhash.argon2id.SALTBYTES,
            password=user_id.encode(),
            salt=b"reddygo" + user_id[:8].encode().ljust(8, b"\x00"),  # Deterministic salt
            opslimit=KeyManager.OPSLIMIT,
            memlimit=KeyManager.MEMLIMIT
        )[:nacl.pwhash.argon2id.SALTBYTES]

        # Derive key
        master_key = nacl.pwhash.argon2id.kdf(
            size=nacl.secret.SecretBox.KEY_SIZE,
            password=password_input.encode(),
            salt=salt,
            opslimit=KeyManager.OPSLIMIT,
            memlimit=KeyManager.MEMLIMIT
        )

        return master_key

    @staticmethod
    def derive_challenge_key(master_key: bytes, challenge_id: str) -> bytes:
        """
        Derive per-challenge encryption key from master key.

        Uses HKDF-like construction.

        Args:
            master_key: User's master key
            challenge_id: Challenge UUID

        Returns:
            32-byte challenge-specific key
        """
        # Use challenge_id as info parameter for key derivation
        info = f"challenge:{challenge_id}".encode()

        # Simple HKDF-like derivation (in production: use full HKDF)
        challenge_key = nacl.pwhash.argon2id.kdf(
            size=nacl.secret.SecretBox.KEY_SIZE,
            password=master_key,
            salt=info[:nacl.pwhash.argon2id.SALTBYTES].ljust(nacl.pwhash.argon2id.SALTBYTES, b"\x00"),
            opslimit=1,  # Fast derivation (master key already secure)
            memlimit=8 * 1024 * 1024  # 8 MB
        )

        return challenge_key

    @staticmethod
    def encrypt_private_key(private_key: bytes, password: str, user_id: str) -> Dict[str, str]:
        """
        Encrypt user's private key with their password (for storage).

        Args:
            private_key: X25519 private key to protect
            password: User's password
            user_id: User ID

        Returns:
            {
                "encrypted_key": base64 string,
                "nonce": base64 string,
                "algorithm": "XChaCha20-Poly1305",
                "kdf": "Argon2id"
            }
        """
        # Derive encryption key from password
        encryption_key = KeyManager.derive_master_key(password, user_id)

        # Encrypt private key
        box = nacl.secret.SecretBox(encryption_key)
        encrypted = box.encrypt(private_key)

        # Extract nonce and ciphertext
        nonce = encrypted[:nacl.secret.SecretBox.NONCE_SIZE]
        ciphertext = encrypted[nacl.secret.SecretBox.NONCE_SIZE:]

        return {
            "encrypted_key": base64.b64encode(ciphertext).decode(),
            "nonce": base64.b64encode(nonce).decode(),
            "algorithm": "XChaCha20-Poly1305",
            "kdf": "Argon2id",
            "version": "1.0"
        }

    @staticmethod
    def decrypt_private_key(encrypted_data: Dict[str, str], password: str, user_id: str) -> bytes:
        """
        Decrypt user's private key.

        Args:
            encrypted_data: Dict with encrypted_key and nonce
            password: User's password
            user_id: User ID

        Returns:
            Decrypted private key
        """
        # Derive decryption key from password
        decryption_key = KeyManager.derive_master_key(password, user_id)

        # Decrypt
        box = nacl.secret.SecretBox(decryption_key)

        ciphertext = base64.b64decode(encrypted_data["encrypted_key"])
        nonce = base64.b64decode(encrypted_data["nonce"])

        encrypted_message = nonce + ciphertext

        private_key = box.decrypt(encrypted_message)

        return private_key

    @staticmethod
    def rotate_master_key(old_password: str, new_password: str, user_id: str, encrypted_data_list: list) -> Dict[str, Any]:
        """
        Rotate master key when user changes password.

        Re-encrypts all user data with new key.

        Args:
            old_password: Current password
            new_password: New password
            user_id: User ID
            encrypted_data_list: List of encrypted data to re-encrypt

        Returns:
            {
                "success": bool,
                "re_encrypted_count": int,
                "new_encrypted_data": list
            }
        """
        # Derive keys
        old_key = KeyManager.derive_master_key(old_password, user_id)
        new_key = KeyManager.derive_master_key(new_password, user_id)

        re_encrypted = []

        try:
            for encrypted_item in encrypted_data_list:
                # Decrypt with old key
                box_old = nacl.secret.SecretBox(old_key)
                ciphertext = base64.b64decode(encrypted_item["encrypted"])
                nonce = base64.b64decode(encrypted_item["nonce"])
                encrypted_message = nonce + ciphertext
                plaintext = box_old.decrypt(encrypted_message)

                # Re-encrypt with new key
                box_new = nacl.secret.SecretBox(new_key)
                new_encrypted = box_new.encrypt(plaintext)
                new_nonce = new_encrypted[:nacl.secret.SecretBox.NONCE_SIZE]
                new_ciphertext = new_encrypted[nacl.secret.SecretBox.NONCE_SIZE:]

                re_encrypted.append({
                    "encrypted": base64.b64encode(new_ciphertext).decode(),
                    "nonce": base64.b64encode(new_nonce).decode(),
                    "algorithm": "XChaCha20-Poly1305",
                    "version": "1.0"
                })

            return {
                "success": True,
                "re_encrypted_count": len(re_encrypted),
                "new_encrypted_data": re_encrypted
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "re_encrypted_count": 0
            }

    @staticmethod
    def generate_device_secret() -> str:
        """
        Generate random device secret (stored in device keychain).

        Adds additional layer of security (password + device secret).

        Returns:
            Base64-encoded random string
        """
        secret = nacl.utils.random(32)
        return base64.b64encode(secret).decode()
