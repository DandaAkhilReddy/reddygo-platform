"""
ReddyGo Authentication Module

Provides authentication dependencies for FastAPI endpoints using Firebase Auth.
"""

from fastapi import Header, HTTPException, Depends
from firebase_admin import auth as firebase_auth
from typing import Optional


async def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    """
    Authentication dependency that verifies Firebase ID tokens.

    Extracts the user ID from the Authorization header Bearer token.

    Args:
        authorization: Authorization header value (Bearer <token>)

    Returns:
        str: Authenticated user ID (Firebase UID)

    Raises:
        HTTPException: 401 if token is missing, invalid, or expired

    Usage:
        @router.get("/protected")
        async def protected_endpoint(current_user: str = Depends(get_current_user)):
            # current_user contains the authenticated Firebase UID
            pass
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Extract token from "Bearer <token>"
    token = authorization.split("Bearer ", 1)[1]

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Empty bearer token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    try:
        # Verify Firebase ID token
        decoded_token = firebase_auth.verify_id_token(token)
        user_id = decoded_token.get('uid')

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid token: missing user ID",
                headers={"WWW-Authenticate": "Bearer"}
            )

        return user_id

    except firebase_auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid Firebase ID token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except firebase_auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=401,
            detail="Firebase ID token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except firebase_auth.RevokedIdTokenError:
        raise HTTPException(
            status_code=401,
            detail="Firebase ID token has been revoked",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except firebase_auth.CertificateFetchError:
        raise HTTPException(
            status_code=503,
            detail="Failed to fetch Firebase certificates. Service temporarily unavailable."
        )
    except Exception as e:
        # Catch-all for any other Firebase auth errors
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )


# Optional: Create a dependency that allows both authenticated and anonymous access
async def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """
    Optional authentication dependency.

    Returns user ID if valid token is provided, None otherwise.
    Useful for endpoints that have different behavior for authenticated vs anonymous users.

    Args:
        authorization: Authorization header value (Bearer <token>)

    Returns:
        Optional[str]: Authenticated user ID or None
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None

    token = authorization.split("Bearer ", 1)[1]

    try:
        decoded_token = firebase_auth.verify_id_token(token)
        return decoded_token.get('uid')
    except Exception:
        # Silently fail for optional auth
        return None
