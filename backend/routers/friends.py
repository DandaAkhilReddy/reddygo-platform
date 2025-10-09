"""
ReddyGo Friends Router

Social features for friend connections, friend requests, and friend leaderboards.

All endpoints require Firebase authentication via Bearer token in Authorization header.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
from firebase_client import get_firestore_client
from firebase_admin import firestore
from realtime_db_client import get_leaderboard, update_leaderboard
from auth import get_current_user

router = APIRouter()


# Request/Response Models
class FriendRequest(BaseModel):
    """Request to send a friend request."""
    to_user_id: str  # from_user_id is extracted from authenticated token


class FriendRequestResponse(BaseModel):
    """Friend request details."""
    request_id: str
    from_user_id: str
    from_user_name: str
    from_user_photo: Optional[str]
    to_user_id: str
    status: str  # pending, accepted, rejected
    created_at: datetime


class Friend(BaseModel):
    """Friend profile."""
    user_id: str
    name: str
    photo_url: Optional[str]
    added_at: datetime
    stats: Dict[str, Any] = {}


class FriendLeaderboardEntry(BaseModel):
    """Friend leaderboard entry."""
    user_id: str
    name: str
    photo_url: Optional[str]
    points: int
    rank: int
    challenges_this_week: int


# ============================================================================
# Friend Requests
# ============================================================================

@router.post("/request", response_model=FriendRequestResponse)
async def send_friend_request(
    request: FriendRequest,
    current_user: str = Depends(get_current_user)
):
    """
    Send a friend request to another user.

    Requires Firebase authentication via Bearer token.

    Validates:
    - Not sending to yourself
    - Users exist
    - Not already friends
    - No pending request exists (including reverse direction)
    """
    db = get_firestore_client()

    # Prevent self-friend requests
    if current_user == request.to_user_id:
        raise HTTPException(status_code=400, detail="Cannot send friend request to yourself")

    # Validate users exist
    from_user_doc = db.collection('users').document(current_user).get()
    to_user_doc = db.collection('users').document(request.to_user_id).get()

    if not from_user_doc.exists or not to_user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if already friends (check both directions)
    existing_friendship1 = db.collection('friendships').where(
        'user1_id', '==', current_user
    ).where('user2_id', '==', request.to_user_id).limit(1).stream()

    existing_friendship2 = db.collection('friendships').where(
        'user1_id', '==', request.to_user_id
    ).where('user2_id', '==', current_user).limit(1).stream()

    if list(existing_friendship1) or list(existing_friendship2):
        raise HTTPException(status_code=400, detail="Already friends")

    # Check for existing pending request (same direction)
    existing_request = db.collection('friend_requests').where(
        'from_user_id', '==', current_user
    ).where('to_user_id', '==', request.to_user_id).where(
        'status', '==', 'pending'
    ).limit(1).stream()

    if list(existing_request):
        raise HTTPException(status_code=400, detail="Friend request already sent")

    # Check for reverse-direction pending request (auto-accept if exists)
    reverse_request_query = db.collection('friend_requests').where(
        'from_user_id', '==', request.to_user_id
    ).where('to_user_id', '==', current_user).where(
        'status', '==', 'pending'
    ).limit(1).stream()

    reverse_requests = list(reverse_request_query)
    if reverse_requests:
        # Auto-accept: create friendship and remove pending request
        reverse_request_doc = reverse_requests[0]
        reverse_request_id = reverse_request_doc.id

        # Validate user documents exist before calling .to_dict() (Issue #5)
        if not from_user_doc.exists:
            raise HTTPException(status_code=404, detail="Authenticated user not found")
        if not to_user_doc.exists:
            raise HTTPException(status_code=404, detail="Target user not found")

        from_user_data = from_user_doc.to_dict()
        to_user_data = to_user_doc.to_dict()

        # Use batch to ensure atomicity
        batch = db.batch()

        # Create bidirectional friendship
        friendship_ref = db.collection('friendships').document()
        batch.set(friendship_ref, {
            "user1_id": current_user,
            "user2_id": request.to_user_id,
            "created_at": firestore.SERVER_TIMESTAMP,
            "status": "active"
        })

        # Add to each other's friend subcollections
        batch.set(
            db.collection('users').document(current_user).collection('friends').document(request.to_user_id),
            {
                "friend_id": request.to_user_id,
                "name": to_user_data.get('name', 'Unknown'),
                "photo_url": to_user_data.get('profile_photo_url'),
                "added_at": firestore.SERVER_TIMESTAMP
            }
        )

        batch.set(
            db.collection('users').document(request.to_user_id).collection('friends').document(current_user),
            {
                "friend_id": current_user,
                "name": from_user_data.get('name', 'Unknown'),
                "photo_url": from_user_data.get('profile_photo_url'),
                "added_at": firestore.SERVER_TIMESTAMP
            }
        )

        # Update friend counts
        batch.update(
            db.collection('users').document(current_user),
            {"social.friends_count": firestore.Increment(1)}
        )
        batch.update(
            db.collection('users').document(request.to_user_id),
            {"social.friends_count": firestore.Increment(1)}
        )

        # Mark reverse request as accepted
        batch.update(
            db.collection('friend_requests').document(reverse_request_id),
            {
                "status": "accepted",
                "responded_at": firestore.SERVER_TIMESTAMP
            }
        )

        # Commit all changes atomically
        batch.commit()

        # Send notifications
        from realtime_db_client import send_realtime_notification
        send_realtime_notification(request.to_user_id, {
            "type": "friend_request_accepted",
            "title": "Friend Request Auto-Accepted",
            "body": f"{from_user_data.get('name')} accepted your friend request",
            "data": {"friend_id": current_user}
        })
        send_realtime_notification(current_user, {
            "type": "friendship_created",
            "title": "You Are Now Friends",
            "body": f"You and {to_user_data.get('name')} are now friends",
            "data": {"friend_id": request.to_user_id}
        })

        # Return accepted friendship response
        return FriendRequestResponse(
            request_id=reverse_request_id,
            from_user_id=request.to_user_id,
            from_user_name=to_user_data.get('name', 'Unknown'),
            from_user_photo=to_user_data.get('profile_photo_url'),
            to_user_id=current_user,
            status="accepted",
            created_at=datetime.utcnow()
        )

    # Create friend request
    from_user_data = from_user_doc.to_dict()

    friend_request_ref = db.collection('friend_requests').document()
    friend_request_data = {
        "from_user_id": current_user,
        "from_user_name": from_user_data.get('name', 'Unknown'),
        "from_user_photo": from_user_data.get('profile_photo_url'),
        "to_user_id": request.to_user_id,
        "status": "pending",
        "created_at": firestore.SERVER_TIMESTAMP
    }

    friend_request_ref.set(friend_request_data)

    # Send real-time notification
    from realtime_db_client import send_realtime_notification
    send_realtime_notification(request.to_user_id, {
        "type": "friend_request",
        "title": "New Friend Request",
        "body": f"{from_user_data.get('name')} sent you a friend request",
        "data": {"request_id": friend_request_ref.id}
    })

    return FriendRequestResponse(
        request_id=friend_request_ref.id,
        from_user_id=current_user,
        from_user_name=from_user_data.get('name', 'Unknown'),
        from_user_photo=from_user_data.get('profile_photo_url'),
        to_user_id=request.to_user_id,
        status="pending",
        created_at=datetime.utcnow()
    )


@router.post("/accept/{request_id}", response_model=Friend)
async def accept_friend_request(
    request_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Accept a friend request.

    Requires Firebase authentication via Bearer token.
    Creates bidirectional friendship and updates both users' friend subcollections.
    """
    db = get_firestore_client()

    # Get friend request
    request_doc = db.collection('friend_requests').document(request_id).get()

    if not request_doc.exists:
        raise HTTPException(status_code=404, detail="Friend request not found")

    request_data = request_doc.to_dict()

    # Verify user is recipient
    if request_data['to_user_id'] != current_user:
        raise HTTPException(status_code=403, detail="Not authorized to accept this request")

    # Check if already accepted
    if request_data['status'] != 'pending':
        raise HTTPException(status_code=400, detail="Request already processed")

    # Create bidirectional friendship
    friendship_ref = db.collection('friendships').document()
    friendship_data = {
        "user1_id": request_data['from_user_id'],
        "user2_id": current_user,
        "created_at": firestore.SERVER_TIMESTAMP,
        "status": "active"
    }
    friendship_ref.set(friendship_data)

    # Get both users' data
    from_user_doc = db.collection('users').document(request_data['from_user_id']).get()
    to_user_doc = db.collection('users').document(current_user).get()

    from_user_data = from_user_doc.to_dict()
    to_user_data = to_user_doc.to_dict()

    # Add to each other's friend subcollections
    db.collection('users').document(current_user).collection('friends').document(request_data['from_user_id']).set({
        "friend_id": request_data['from_user_id'],
        "name": from_user_data.get('name', 'Unknown'),
        "photo_url": from_user_data.get('profile_photo_url'),
        "added_at": firestore.SERVER_TIMESTAMP
    })

    db.collection('users').document(request_data['from_user_id']).collection('friends').document(current_user).set({
        "friend_id": current_user,
        "name": to_user_data.get('name', 'Unknown'),
        "photo_url": to_user_data.get('profile_photo_url'),
        "added_at": firestore.SERVER_TIMESTAMP
    })

    # Update friend request status
    db.collection('friend_requests').document(request_id).update({
        "status": "accepted",
        "responded_at": firestore.SERVER_TIMESTAMP
    })

    # Update friend counts
    db.collection('users').document(current_user).update({
        "social.friends_count": firestore.Increment(1)
    })
    db.collection('users').document(request_data['from_user_id']).update({
        "social.friends_count": firestore.Increment(1)
    })

    # Send notification
    from realtime_db_client import send_realtime_notification
    send_realtime_notification(request_data['from_user_id'], {
        "type": "friend_request_accepted",
        "title": "Friend Request Accepted",
        "body": f"{to_user_data.get('name')} accepted your friend request",
        "data": {"friend_id": current_user}
    })

    return Friend(
        user_id=request_data['from_user_id'],
        name=from_user_data.get('name', 'Unknown'),
        photo_url=from_user_data.get('profile_photo_url'),
        added_at=datetime.utcnow(),
        stats=from_user_data.get('stats', {})
    )


@router.post("/reject/{request_id}", status_code=204)
async def reject_friend_request(
    request_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Reject a friend request.

    Requires Firebase authentication via Bearer token.
    """
    db = get_firestore_client()

    # Get friend request
    request_doc = db.collection('friend_requests').document(request_id).get()

    if not request_doc.exists:
        raise HTTPException(status_code=404, detail="Friend request not found")

    request_data = request_doc.to_dict()

    # Verify user is recipient
    if request_data['to_user_id'] != current_user:
        raise HTTPException(status_code=403, detail="Not authorized to reject this request")

    # Update status
    db.collection('friend_requests').document(request_id).update({
        "status": "rejected",
        "responded_at": firestore.SERVER_TIMESTAMP
    })

    return None


@router.get("/requests", response_model=List[FriendRequestResponse])
async def get_friend_requests(
    current_user: str = Depends(get_current_user),
    status: str = "pending"
):
    """
    Get friend requests for the authenticated user.

    Requires Firebase authentication via Bearer token.

    Args:
        status: pending | accepted | rejected (default: pending)
    """
    db = get_firestore_client()

    # Get incoming requests
    requests_ref = db.collection('friend_requests').where(
        'to_user_id', '==', current_user
    ).where('status', '==', status).order_by('created_at', direction=firestore.Query.DESCENDING)

    requests = requests_ref.stream()

    result = []
    for request_doc in requests:
        request_data = request_doc.to_dict()
        result.append(FriendRequestResponse(
            request_id=request_doc.id,
            from_user_id=request_data['from_user_id'],
            from_user_name=request_data.get('from_user_name', 'Unknown'),
            from_user_photo=request_data.get('from_user_photo'),
            to_user_id=current_user,
            status=request_data['status'],
            created_at=request_data.get('created_at', datetime.utcnow())
        ))

    return result


# ============================================================================
# Friends List
# ============================================================================

@router.get("/", response_model=List[Friend])
async def get_friends(current_user: str = Depends(get_current_user)):
    """
    Get authenticated user's friends list.

    Requires Firebase authentication via Bearer token.
    """
    db = get_firestore_client()

    # Get friends from subcollection
    friends_ref = db.collection('users').document(current_user).collection('friends')
    friends = friends_ref.stream()

    result = []
    for friend_doc in friends:
        friend_data = friend_doc.to_dict()

        # Get friend's current stats
        friend_user_doc = db.collection('users').document(friend_data['friend_id']).get()
        friend_stats = {}
        if friend_user_doc.exists:
            friend_stats = friend_user_doc.to_dict().get('stats', {})

        result.append(Friend(
            user_id=friend_data['friend_id'],
            name=friend_data.get('name', 'Unknown'),
            photo_url=friend_data.get('photo_url'),
            added_at=friend_data.get('added_at', datetime.utcnow()),
            stats=friend_stats
        ))

    return result


@router.delete("/remove/{friend_id}", status_code=204)
async def remove_friend(
    friend_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Remove a friend.

    Requires Firebase authentication via Bearer token.
    Deletes bidirectional friendship and updates both users' friend subcollections.
    """
    db = get_firestore_client()

    # Find and delete friendship (check both directions)
    friendship_query1 = db.collection('friendships').where(
        'user1_id', '==', current_user
    ).where('user2_id', '==', friend_id).limit(1)

    friendship_query2 = db.collection('friendships').where(
        'user1_id', '==', friend_id
    ).where('user2_id', '==', current_user).limit(1)

    friendship_docs = list(friendship_query1.stream()) + list(friendship_query2.stream())

    if not friendship_docs:
        raise HTTPException(status_code=404, detail="Friendship not found")

    # Delete friendship
    for friendship_doc in friendship_docs:
        friendship_doc.reference.delete()

    # Remove from both users' friend subcollections
    db.collection('users').document(current_user).collection('friends').document(friend_id).delete()
    db.collection('users').document(friend_id).collection('friends').document(current_user).delete()

    # Update friend counts
    db.collection('users').document(current_user).update({
        "social.friends_count": firestore.Increment(-1)
    })
    db.collection('users').document(friend_id).update({
        "social.friends_count": firestore.Increment(-1)
    })

    return None


# ============================================================================
# Friend Leaderboard
# ============================================================================

@router.get("/leaderboard", response_model=List[FriendLeaderboardEntry])
async def get_friend_leaderboard(
    current_user: str = Depends(get_current_user),
    timeframe: str = "weekly"
):
    """
    Get friend leaderboard for the authenticated user.

    Requires Firebase authentication via Bearer token.

    Args:
        timeframe: weekly | monthly | all_time

    Returns:
        Ranked list of friends by points
    """
    if timeframe not in ["weekly", "monthly", "all_time"]:
        raise HTTPException(status_code=400, detail="Invalid timeframe. Must be weekly, monthly, or all_time")

    # Get leaderboard from Realtime Database
    leaderboard = get_leaderboard(
        leaderboard_type="friends",
        leaderboard_id=current_user,
        timeframe=timeframe,
        limit=100
    )

    # If empty, populate from Firestore
    if not leaderboard:
        db = get_firestore_client()

        # Get user's friends
        friends_ref = db.collection('users').document(current_user).collection('friends')
        friends = friends_ref.stream()

        leaderboard_data = []
        for friend_doc in friends:
            friend_data = friend_doc.to_dict()
            friend_id = friend_data['friend_id']

            # Get friend's stats
            friend_user_doc = db.collection('users').document(friend_id).get()
            if friend_user_doc.exists:
                friend_user_data = friend_user_doc.to_dict()
                stats = friend_user_data.get('stats', {})

                leaderboard_data.append({
                    "user_id": friend_id,
                    "name": friend_user_data.get('name', 'Unknown'),
                    "photo_url": friend_user_data.get('profile_photo_url'),
                    "points": stats.get('points_balance', 0),
                    "challenges_this_week": stats.get('challenges_this_week', 0)
                })

        # Sort by points
        leaderboard_data.sort(key=lambda x: x['points'], reverse=True)

        # Add ranks
        for rank, entry in enumerate(leaderboard_data, start=1):
            entry['rank'] = rank

        # Update Realtime DB for future fast access
        for entry in leaderboard_data:
            update_leaderboard(
                leaderboard_type="friends",
                leaderboard_id=current_user,
                timeframe=timeframe,
                user_id=entry['user_id'],
                points=entry['points'],
                user_name=entry['name'],
                user_photo_url=entry.get('photo_url')
            )

        leaderboard = leaderboard_data

    # Recompute ranks if missing from cached data
    if leaderboard and any('rank' not in entry for entry in leaderboard):
        # Sort by points descending
        leaderboard.sort(key=lambda x: x.get('points', 0), reverse=True)
        # Assign ranks
        for rank, entry in enumerate(leaderboard, start=1):
            entry['rank'] = rank

    return [
        FriendLeaderboardEntry(
            user_id=entry['user_id'],
            name=entry.get('name', 'Unknown'),
            photo_url=entry.get('photo_url'),
            points=entry.get('points', 0),
            rank=entry.get('rank', 0),
            challenges_this_week=entry.get('challenges_this_week', 0)
        )
        for entry in leaderboard
    ]
