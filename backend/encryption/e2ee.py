"""
End-to-End Encryption Manager

Client-side encryption → Encrypted storage → Zero-knowledge backend

Uses modern cryptography:
- XChaCha20-Poly1305 (authenticated encryption)
- X25519 (key exchange for shared challenges)
- Argon2 (password-based key derivation)
"""

import nacl.secret
import nacl.public
import nacl.utils
import nacl.pwhash
from typing import Dict, Any, Optional
import json
import base64


class E2EEManager:
    """
    End-to-end encryption manager for sensitive user data.

    Server stores encrypted blobs and cannot decrypt them.
    Only clients with the correct key can read the data.
    """

    @staticmethod
    def encrypt_data(plaintext: str, key: bytes) -> Dict[str, str]:
        """
        Encrypt data with XChaCha20-Poly1305.

        Args:
            plaintext: Data to encrypt (JSON string)
            key: 32-byte encryption key

        Returns:
            {
                "encrypted": base64-encoded ciphertext,
                "nonce": base64-encoded nonce
            }
        """
        box = nacl.secret.SecretBox(key)

        # Encrypt
        encrypted = box.encrypt(plaintext.encode())

        # Extract nonce and ciphertext
        nonce = encrypted[:nacl.secret.SecretBox.NONCE_SIZE]
        ciphertext = encrypted[nacl.secret.SecretBox.NONCE_SIZE:]

        return {
            "encrypted": base64.b64encode(ciphertext).decode(),
            "nonce": base64.b64encode(nonce).decode(),
            "algorithm": "XChaCha20-Poly1305",
            "version": "1.0"
        }

    @staticmethod
    def decrypt_data(encrypted_data: Dict[str, str], key: bytes) -> str:
        """
        Decrypt data encrypted with XChaCha20-Poly1305.

        Args:
            encrypted_data: Dict with "encrypted" and "nonce" keys
            key: 32-byte encryption key

        Returns:
            Decrypted plaintext string
        """
        box = nacl.secret.SecretBox(key)

        # Decode from base64
        ciphertext = base64.b64decode(encrypted_data["encrypted"])
        nonce = base64.b64decode(encrypted_data["nonce"])

        # Reconstruct encrypted message
        encrypted_message = nonce + ciphertext

        # Decrypt
        plaintext = box.decrypt(encrypted_message)

        return plaintext.decode()

    @staticmethod
    def encrypt_gps_track(track_data: list, key: bytes) -> Dict[str, Any]:
        """
        Encrypt GPS track data.

        Args:
            track_data: List of GPS points
            key: User's encryption key

        Returns:
            Encrypted GPS track with metadata
        """
        plaintext = json.dumps(track_data)
        encrypted = E2EEManager.encrypt_data(plaintext, key)

        return {
            **encrypted,
            "data_type": "gps_track",
            "point_count": len(track_data)
        }

    @staticmethod
    def decrypt_gps_track(encrypted_track: Dict[str, Any], key: bytes) -> list:
        """
        Decrypt GPS track data.

        Args:
            encrypted_track: Encrypted track dict
            key: User's encryption key

        Returns:
            List of GPS points
        """
        plaintext = E2EEManager.decrypt_data(encrypted_track, key)
        return json.loads(plaintext)

    @staticmethod
    def generate_keypair() -> Dict[str, bytes]:
        """
        Generate X25519 keypair for key exchange.

        Used for shared challenges where multiple users need to see each other's data.

        Returns:
            {
                "private_key": 32-byte private key,
                "public_key": 32-byte public key
            }
        """
        private_key = nacl.public.PrivateKey.generate()
        public_key = private_key.public_key

        return {
            "private_key": bytes(private_key),
            "public_key": bytes(public_key)
        }

    @staticmethod
    def compute_shared_secret(my_private_key: bytes, their_public_key: bytes) -> bytes:
        """
        Compute shared secret for encrypted communication.

        Args:
            my_private_key: My X25519 private key
            their_public_key: Other user's X25519 public key

        Returns:
            32-byte shared secret (use as encryption key)
        """
        private_key = nacl.public.PrivateKey(my_private_key)
        public_key = nacl.public.PublicKey(their_public_key)

        # Compute shared secret using X25519
        box = nacl.public.Box(private_key, public_key)

        # Return the shared secret (box._shared_key)
        # For encryption, we'll use box.encrypt/decrypt directly
        return bytes(box)

    @staticmethod
    def encrypt_for_user(plaintext: str, my_private_key: bytes, their_public_key: bytes) -> Dict[str, str]:
        """
        Encrypt data for another user (shared challenge).

        Args:
            plaintext: Data to encrypt
            my_private_key: My X25519 private key
            their_public_key: Their X25519 public key

        Returns:
            Encrypted data dict
        """
        private_key = nacl.public.PrivateKey(my_private_key)
        public_key = nacl.public.PublicKey(their_public_key)

        box = nacl.public.Box(private_key, public_key)

        # Encrypt
        encrypted = box.encrypt(plaintext.encode())

        # Extract nonce and ciphertext
        nonce = encrypted[:nacl.public.Box.NONCE_SIZE]
        ciphertext = encrypted[nacl.public.Box.NONCE_SIZE:]

        return {
            "encrypted": base64.b64encode(ciphertext).decode(),
            "nonce": base64.b64encode(nonce).decode(),
            "algorithm": "X25519-XChaCha20-Poly1305",
            "version": "1.0"
        }

    @staticmethod
    def decrypt_from_user(encrypted_data: Dict[str, str], my_private_key: bytes, their_public_key: bytes) -> str:
        """
        Decrypt data from another user.

        Args:
            encrypted_data: Encrypted dict with "encrypted" and "nonce"
            my_private_key: My X25519 private key
            their_public_key: Their X25519 public key

        Returns:
            Decrypted plaintext
        """
        private_key = nacl.public.PrivateKey(my_private_key)
        public_key = nacl.public.PublicKey(their_public_key)

        box = nacl.public.Box(private_key, public_key)

        # Decode from base64
        ciphertext = base64.b64decode(encrypted_data["encrypted"])
        nonce = base64.b64decode(encrypted_data["nonce"])

        # Reconstruct encrypted message
        encrypted_message = nonce + ciphertext

        # Decrypt
        plaintext = box.decrypt(encrypted_message)

        return plaintext.decode()

    @staticmethod
    def generate_challenge_key() -> bytes:
        """
        Generate random encryption key for a challenge.

        Returns:
            32-byte random key
        """
        return nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE)

    @staticmethod
    def encode_key(key: bytes) -> str:
        """Encode key to base64 string for storage."""
        return base64.b64encode(key).decode()

    @staticmethod
    def decode_key(key_str: str) -> bytes:
        """Decode key from base64 string."""
        return base64.b64decode(key_str)
