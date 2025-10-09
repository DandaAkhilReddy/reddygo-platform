"""
ReddyGo Communities Router

Public/private communities with monetization, founder benefits, and member management.

All endpoints require Firebase authentication via Bearer token in Authorization header.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from firebase_client import get_firestore_client
from firebase_admin import firestore
from auth import get_current_user

router = APIRouter()


# ============================================================================
# Pydantic Models
# ============================================================================

class CommunityCreate(BaseModel):
    """Request to create a new community."""
    name: str = Field(..., min_length=3, max_length=100)
    description: str = Field(..., max_length=1000)
    photo_url: Optional[str] = None
    banner_url: Optional[str] = None
    type: str = Field(..., pattern="^(public|private)$")
    sport_type: str = Field(..., pattern="^(running|cycling|mixed)$")
    is_premium: bool = False
    monthly_fee: Optional[float] = Field(None, ge=0, le=99.99)


class CommunityUpdate(BaseModel):
    """Update community details."""
    name: Optional[str] = Field(None, min_length=3, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    photo_url: Optional[str] = None
    banner_url: Optional[str] = None
    type: Optional[str] = Field(None, pattern="^(public|private)$")
    sport_type: Optional[str] = Field(None, pattern="^(running|cycling|mixed)$")
    is_premium: Optional[bool] = None
    monthly_fee: Optional[float] = Field(None, ge=0, le=99.99)


class CommunityResponse(BaseModel):
    """Community details."""
    id: str
    name: str
    description: str
    photo_url: Optional[str]
    banner_url: Optional[str]
    founder_id: str
    founder_name: str
    created_at: datetime
    type: str
    sport_type: str
    member_count: int
    active_members_this_week: int
    is_premium: bool
    monthly_fee: Optional[float]
    founder_revenue_share: int
    is_member: bool = False
    user_role: Optional[str] = None


class CommunityMember(BaseModel):
    """Community member details."""
    user_id: str
    name: str
    photo_url: Optional[str]
    joined_at: datetime
    role: str  # admin | moderator | member
    subscription_status: Optional[str] = None  # active | expired (for premium communities)


class MemberRoleUpdate(BaseModel):
    """Update member role."""
    role: str = Field(..., pattern="^(admin|moderator|member)$")


# ============================================================================
# Create Community
# ============================================================================

@router.post("/create", response_model=CommunityResponse)
async def create_community(
    community: CommunityCreate,
    current_user: str = Depends(get_current_user)
):
    """
    Create a new community.

    Requires Firebase authentication. The authenticated user becomes the community founder.

    Validates:
    - User exists
    - Premium communities must have a monthly fee
    - User is not at maximum community limit (10 as founder)
    """
    db = get_firestore_client()

    # Validate user exists
    user_doc = db.collection('users').document(current_user).get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user_doc.to_dict()

    # Validate premium community has fee
    if community.is_premium and (not community.monthly_fee or community.monthly_fee <= 0):
        raise HTTPException(
            status_code=400,
            detail="Premium communities must have a monthly fee greater than $0"
        )

    # Check community founder limit (max 10 founded communities)
    founded_communities = db.collection('communities').where(
        'founder_id', '==', current_user
    ).stream()

    if len(list(founded_communities)) >= 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum limit of 10 founded communities reached"
        )

    # Create community
    community_ref = db.collection('communities').document()
    community_id = community_ref.id

    community_data = {
        "name": community.name,
        "description": community.description,
        "photo_url": community.photo_url,
        "banner_url": community.banner_url,
        "founder_id": current_user,
        "founder_name": user_data.get('name', 'Unknown'),
        "created_at": firestore.SERVER_TIMESTAMP,
        "type": community.type,
        "sport_type": community.sport_type,
        "member_count": 1,  # Founder is first member
        "active_members_this_week": 1,
        "monetization": {
            "is_premium": community.is_premium,
            "monthly_fee": community.monthly_fee or 0,
            "founder_revenue_share": 60 if community.is_premium else 0
        },
        "founder_benefits": {
            "founder_badge": True,
            "verified": False,  # Verified at 1000+ members
            "lifetime_revenue_earned": 0.00,
            "merchandise_unlocked": [],
            "custom_branding": {
                "logo_url": community.photo_url,
                "primary_color": "#FF5733",
                "banner_url": community.banner_url
            }
        },
        "stats": {
            "total_challenges_created": 0,
            "total_participants": 0,
            "total_distance_km": 0
        }
    }

    community_ref.set(community_data)

    # Add founder as admin member
    member_ref = community_ref.collection('members').document(current_user)
    member_ref.set({
        "user_id": current_user,
        "name": user_data.get('name', 'Unknown'),
        "photo_url": user_data.get('profile_photo_url'),
        "joined_at": firestore.SERVER_TIMESTAMP,
        "role": "admin",
        "subscription_status": "active" if community.is_premium else None
    })

    # Update user's communities list
    user_social = user_data.get('social', {})
    user_communities = user_social.get('communities', [])
    user_communities.append(community_id)

    db.collection('users').document(current_user).update({
        "social.communities": user_communities
    })

    # Create founder rewards tracking
    db.collection('founder_rewards').document(current_user).set({
        "user_id": current_user,
        "community_founder": firestore.ArrayUnion([{
            "community_id": community_id,
            "community_name": community.name,
            "member_count": 1,
            "revenue_earned": 0.00,
            "verified": False,
            "merchandise_unlocked": []
        }])
    }, merge=True)

    return CommunityResponse(
        id=community_id,
        name=community.name,
        description=community.description,
        photo_url=community.photo_url,
        banner_url=community.banner_url,
        founder_id=current_user,
        founder_name=user_data.get('name', 'Unknown'),
        created_at=datetime.utcnow(),
        type=community.type,
        sport_type=community.sport_type,
        member_count=1,
        active_members_this_week=1,
        is_premium=community.is_premium,
        monthly_fee=community.monthly_fee,
        founder_revenue_share=60 if community.is_premium else 0,
        is_member=True,
        user_role="admin"
    )


# ============================================================================
# Get Communities
# ============================================================================

@router.get("/", response_model=List[CommunityResponse])
async def list_communities(
    current_user: str = Depends(get_current_user),
    type: Optional[str] = Query(None, pattern="^(public|private)$"),
    sport_type: Optional[str] = Query(None, pattern="^(running|cycling|mixed)$"),
    is_premium: Optional[bool] = None,
    limit: int = Query(20, ge=1, le=100)
):
    """
    List communities with optional filters.

    Requires Firebase authentication.

    Filters:
    - type: public | private
    - sport_type: running | cycling | mixed
    - is_premium: true | false
    - limit: max results (default 20, max 100)
    """
    db = get_firestore_client()

    # Build query
    query = db.collection('communities')

    if type:
        query = query.where('type', '==', type)
    if sport_type:
        query = query.where('sport_type', '==', sport_type)
    if is_premium is not None:
        query = query.where('monetization.is_premium', '==', is_premium)

    # Order by member count (most popular first)
    query = query.order_by('member_count', direction=firestore.Query.DESCENDING).limit(limit)

    communities = query.stream()

    result = []
    for community_doc in communities:
        community_data = community_doc.to_dict()
        community_id = community_doc.id

        # Check if user is a member
        member_doc = db.collection('communities').document(community_id).collection('members').document(current_user).get()
        is_member = member_doc.exists
        user_role = member_doc.to_dict().get('role') if member_doc.exists else None

        result.append(CommunityResponse(
            id=community_id,
            name=community_data['name'],
            description=community_data['description'],
            photo_url=community_data.get('photo_url'),
            banner_url=community_data.get('banner_url'),
            founder_id=community_data['founder_id'],
            founder_name=community_data.get('founder_name', 'Unknown'),
            created_at=community_data.get('created_at', datetime.utcnow()),
            type=community_data['type'],
            sport_type=community_data['sport_type'],
            member_count=community_data.get('member_count', 0),
            active_members_this_week=community_data.get('active_members_this_week', 0),
            is_premium=community_data.get('monetization', {}).get('is_premium', False),
            monthly_fee=community_data.get('monetization', {}).get('monthly_fee'),
            founder_revenue_share=community_data.get('monetization', {}).get('founder_revenue_share', 0),
            is_member=is_member,
            user_role=user_role
        ))

    return result


@router.get("/my-communities", response_model=List[CommunityResponse])
async def get_my_communities(current_user: str = Depends(get_current_user)):
    """
    Get communities the authenticated user is a member of.

    Requires Firebase authentication.
    """
    db = get_firestore_client()

    # Get user's community IDs
    user_doc = db.collection('users').document(current_user).get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user_doc.to_dict()
    community_ids = user_data.get('social', {}).get('communities', [])

    if not community_ids:
        return []

    result = []
    for community_id in community_ids:
        community_doc = db.collection('communities').document(community_id).get()
        if not community_doc.exists:
            continue

        community_data = community_doc.to_dict()

        # Get user's role
        member_doc = db.collection('communities').document(community_id).collection('members').document(current_user).get()
        user_role = member_doc.to_dict().get('role') if member_doc.exists else None

        result.append(CommunityResponse(
            id=community_id,
            name=community_data['name'],
            description=community_data['description'],
            photo_url=community_data.get('photo_url'),
            banner_url=community_data.get('banner_url'),
            founder_id=community_data['founder_id'],
            founder_name=community_data.get('founder_name', 'Unknown'),
            created_at=community_data.get('created_at', datetime.utcnow()),
            type=community_data['type'],
            sport_type=community_data['sport_type'],
            member_count=community_data.get('member_count', 0),
            active_members_this_week=community_data.get('active_members_this_week', 0),
            is_premium=community_data.get('monetization', {}).get('is_premium', False),
            monthly_fee=community_data.get('monetization', {}).get('monthly_fee'),
            founder_revenue_share=community_data.get('monetization', {}).get('founder_revenue_share', 0),
            is_member=True,
            user_role=user_role
        ))

    return result


@router.get("/{community_id}", response_model=CommunityResponse)
async def get_community(
    community_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Get community details.

    Requires Firebase authentication.
    """
    db = get_firestore_client()

    community_doc = db.collection('communities').document(community_id).get()
    if not community_doc.exists:
        raise HTTPException(status_code=404, detail="Community not found")

    community_data = community_doc.to_dict()

    # Check if user is a member
    member_doc = db.collection('communities').document(community_id).collection('members').document(current_user).get()
    is_member = member_doc.exists
    user_role = member_doc.to_dict().get('role') if member_doc.exists else None

    return CommunityResponse(
        id=community_id,
        name=community_data['name'],
        description=community_data['description'],
        photo_url=community_data.get('photo_url'),
        banner_url=community_data.get('banner_url'),
        founder_id=community_data['founder_id'],
        founder_name=community_data.get('founder_name', 'Unknown'),
        created_at=community_data.get('created_at', datetime.utcnow()),
        type=community_data['type'],
        sport_type=community_data['sport_type'],
        member_count=community_data.get('member_count', 0),
        active_members_this_week=community_data.get('active_members_this_week', 0),
        is_premium=community_data.get('monetization', {}).get('is_premium', False),
        monthly_fee=community_data.get('monetization', {}).get('monthly_fee'),
        founder_revenue_share=community_data.get('monetization', {}).get('founder_revenue_share', 0),
        is_member=is_member,
        user_role=user_role
    )


# ============================================================================
# Update/Delete Community
# ============================================================================

@router.put("/{community_id}", response_model=CommunityResponse)
async def update_community(
    community_id: str,
    updates: CommunityUpdate,
    current_user: str = Depends(get_current_user)
):
    """
    Update community details.

    Requires Firebase authentication and admin/founder role.
    """
    db = get_firestore_client()

    # Get community
    community_doc = db.collection('communities').document(community_id).get()
    if not community_doc.exists:
        raise HTTPException(status_code=404, detail="Community not found")

    community_data = community_doc.to_dict()

    # Check if user is admin or founder
    member_doc = db.collection('communities').document(community_id).collection('members').document(current_user).get()
    if not member_doc.exists:
        raise HTTPException(status_code=403, detail="Not a member of this community")

    member_data = member_doc.to_dict()
    if member_data.get('role') not in ['admin'] and community_data['founder_id'] != current_user:
        raise HTTPException(status_code=403, detail="Only admins or founder can update community")

    # Build update dict
    update_dict = {}
    if updates.name is not None:
        update_dict['name'] = updates.name
    if updates.description is not None:
        update_dict['description'] = updates.description
    if updates.photo_url is not None:
        update_dict['photo_url'] = updates.photo_url
    if updates.banner_url is not None:
        update_dict['banner_url'] = updates.banner_url
    if updates.type is not None:
        update_dict['type'] = updates.type
    if updates.sport_type is not None:
        update_dict['sport_type'] = updates.sport_type
    if updates.is_premium is not None:
        update_dict['monetization.is_premium'] = updates.is_premium
    if updates.monthly_fee is not None:
        update_dict['monetization.monthly_fee'] = updates.monthly_fee

    if not update_dict:
        raise HTTPException(status_code=400, detail="No updates provided")

    # Validate premium fee
    if updates.is_premium and (not updates.monthly_fee or updates.monthly_fee <= 0):
        if not community_data.get('monetization', {}).get('monthly_fee'):
            raise HTTPException(
                status_code=400,
                detail="Premium communities must have a monthly fee greater than $0"
            )

    # Update community
    db.collection('communities').document(community_id).update(update_dict)

    # Get updated community
    updated_community = db.collection('communities').document(community_id).get().to_dict()

    return CommunityResponse(
        id=community_id,
        name=updated_community['name'],
        description=updated_community['description'],
        photo_url=updated_community.get('photo_url'),
        banner_url=updated_community.get('banner_url'),
        founder_id=updated_community['founder_id'],
        founder_name=updated_community.get('founder_name', 'Unknown'),
        created_at=updated_community.get('created_at', datetime.utcnow()),
        type=updated_community['type'],
        sport_type=updated_community['sport_type'],
        member_count=updated_community.get('member_count', 0),
        active_members_this_week=updated_community.get('active_members_this_week', 0),
        is_premium=updated_community.get('monetization', {}).get('is_premium', False),
        monthly_fee=updated_community.get('monetization', {}).get('monthly_fee'),
        founder_revenue_share=updated_community.get('monetization', {}).get('founder_revenue_share', 0),
        is_member=True,
        user_role=member_data.get('role')
    )


@router.delete("/{community_id}", status_code=204)
async def delete_community(
    community_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Delete a community.

    Requires Firebase authentication and must be the community founder.
    Only communities with less than 10 members can be deleted.
    """
    db = get_firestore_client()

    # Get community
    community_doc = db.collection('communities').document(community_id).get()
    if not community_doc.exists:
        raise HTTPException(status_code=404, detail="Community not found")

    community_data = community_doc.to_dict()

    # Check if user is founder
    if community_data['founder_id'] != current_user:
        raise HTTPException(status_code=403, detail="Only the founder can delete this community")

    # Check member count
    if community_data.get('member_count', 0) >= 10:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete communities with 10 or more members. Please transfer ownership or remove members first."
        )

    # Delete all members
    members = db.collection('communities').document(community_id).collection('members').stream()
    for member in members:
        member.reference.delete()

        # Remove community from user's communities list
        member_id = member.id
        user_doc = db.collection('users').document(member_id).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            user_communities = user_data.get('social', {}).get('communities', [])
            if community_id in user_communities:
                user_communities.remove(community_id)
                db.collection('users').document(member_id).update({
                    "social.communities": user_communities
                })

    # Delete community
    db.collection('communities').document(community_id).delete()

    return None


# ============================================================================
# Join/Leave Community
# ============================================================================

@router.post("/{community_id}/join", response_model=CommunityMember)
async def join_community(
    community_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Join a community.

    Requires Firebase authentication.

    For premium communities:
    - User must have an active Pro subscription
    - Payment will be processed via Stripe (TODO: integrate Stripe)
    """
    db = get_firestore_client()

    # Get community
    community_doc = db.collection('communities').document(community_id).get()
    if not community_doc.exists:
        raise HTTPException(status_code=404, detail="Community not found")

    community_data = community_doc.to_dict()

    # Check if already a member
    member_doc = db.collection('communities').document(community_id).collection('members').document(current_user).get()
    if member_doc.exists:
        raise HTTPException(status_code=400, detail="Already a member of this community")

    # Get user data
    user_doc = db.collection('users').document(current_user).get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user_doc.to_dict()

    # Check premium community subscription requirement
    is_premium = community_data.get('monetization', {}).get('is_premium', False)
    if is_premium:
        user_subscription = user_data.get('subscription', {})
        if user_subscription.get('tier') == 'free' or user_subscription.get('status') != 'active':
            raise HTTPException(
                status_code=403,
                detail="Premium communities require an active Pro subscription"
            )
        # TODO: Process payment via Stripe

    # Add user as member
    member_ref = db.collection('communities').document(community_id).collection('members').document(current_user)
    member_data = {
        "user_id": current_user,
        "name": user_data.get('name', 'Unknown'),
        "photo_url": user_data.get('profile_photo_url'),
        "joined_at": firestore.SERVER_TIMESTAMP,
        "role": "member",
        "subscription_status": "active" if is_premium else None
    }
    member_ref.set(member_data)

    # Update community member count
    db.collection('communities').document(community_id).update({
        "member_count": firestore.Increment(1),
        "active_members_this_week": firestore.Increment(1)
    })

    # Add community to user's communities list
    user_communities = user_data.get('social', {}).get('communities', [])
    user_communities.append(community_id)
    db.collection('users').document(current_user).update({
        "social.communities": user_communities
    })

    # Send notification to founder
    from realtime_db_client import send_realtime_notification
    send_realtime_notification(community_data['founder_id'], {
        "type": "community_new_member",
        "title": "New Community Member",
        "body": f"{user_data.get('name')} joined {community_data['name']}",
        "data": {"community_id": community_id, "user_id": current_user}
    })

    return CommunityMember(
        user_id=current_user,
        name=user_data.get('name', 'Unknown'),
        photo_url=user_data.get('profile_photo_url'),
        joined_at=datetime.utcnow(),
        role="member",
        subscription_status="active" if is_premium else None
    )


@router.post("/{community_id}/leave", status_code=204)
async def leave_community(
    community_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Leave a community.

    Requires Firebase authentication.
    Founders cannot leave their own communities (must transfer ownership or delete).
    """
    db = get_firestore_client()

    # Get community
    community_doc = db.collection('communities').document(community_id).get()
    if not community_doc.exists:
        raise HTTPException(status_code=404, detail="Community not found")

    community_data = community_doc.to_dict()

    # Check if user is founder
    if community_data['founder_id'] == current_user:
        raise HTTPException(
            status_code=400,
            detail="Founders cannot leave their own communities. Please transfer ownership or delete the community."
        )

    # Check if member
    member_doc = db.collection('communities').document(community_id).collection('members').document(current_user).get()
    if not member_doc.exists:
        raise HTTPException(status_code=400, detail="Not a member of this community")

    # Remove member
    member_doc.reference.delete()

    # Update community member count
    db.collection('communities').document(community_id).update({
        "member_count": firestore.Increment(-1)
    })

    # Remove community from user's communities list
    user_doc = db.collection('users').document(current_user).get()
    if user_doc.exists:
        user_data = user_doc.to_dict()
        user_communities = user_data.get('social', {}).get('communities', [])
        if community_id in user_communities:
            user_communities.remove(community_id)
            db.collection('users').document(current_user).update({
                "social.communities": user_communities
            })

    return None


# ============================================================================
# Member Management
# ============================================================================

@router.get("/{community_id}/members", response_model=List[CommunityMember])
async def get_community_members(
    community_id: str,
    current_user: str = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=200)
):
    """
    Get community members.

    Requires Firebase authentication and community membership.
    """
    db = get_firestore_client()

    # Check if community exists
    community_doc = db.collection('communities').document(community_id).get()
    if not community_doc.exists:
        raise HTTPException(status_code=404, detail="Community not found")

    # Check if user is a member
    member_doc = db.collection('communities').document(community_id).collection('members').document(current_user).get()
    if not member_doc.exists:
        raise HTTPException(status_code=403, detail="Must be a member to view members list")

    # Get members
    members_ref = db.collection('communities').document(community_id).collection('members').limit(limit)
    members = members_ref.stream()

    result = []
    for member in members:
        member_data = member.to_dict()
        result.append(CommunityMember(
            user_id=member_data['user_id'],
            name=member_data.get('name', 'Unknown'),
            photo_url=member_data.get('photo_url'),
            joined_at=member_data.get('joined_at', datetime.utcnow()),
            role=member_data.get('role', 'member'),
            subscription_status=member_data.get('subscription_status')
        ))

    return result


@router.put("/{community_id}/members/{user_id}/role", response_model=CommunityMember)
async def update_member_role(
    community_id: str,
    user_id: str,
    role_update: MemberRoleUpdate,
    current_user: str = Depends(get_current_user)
):
    """
    Update a member's role.

    Requires Firebase authentication and admin/founder permissions.

    Roles:
    - admin: Can manage members, update community settings
    - moderator: Can moderate content, challenges
    - member: Standard member
    """
    db = get_firestore_client()

    # Get community
    community_doc = db.collection('communities').document(community_id).get()
    if not community_doc.exists:
        raise HTTPException(status_code=404, detail="Community not found")

    community_data = community_doc.to_dict()

    # Check if current user is admin or founder
    current_user_member_doc = db.collection('communities').document(community_id).collection('members').document(current_user).get()
    if not current_user_member_doc.exists:
        raise HTTPException(status_code=403, detail="Not a member of this community")

    current_user_role = current_user_member_doc.to_dict().get('role')
    if current_user_role != 'admin' and community_data['founder_id'] != current_user:
        raise HTTPException(status_code=403, detail="Only admins or founder can update member roles")

    # Cannot change founder's role
    if user_id == community_data['founder_id']:
        raise HTTPException(status_code=400, detail="Cannot change founder's role")

    # Get target member
    member_doc = db.collection('communities').document(community_id).collection('members').document(user_id).get()
    if not member_doc.exists:
        raise HTTPException(status_code=404, detail="Member not found")

    # Update role
    db.collection('communities').document(community_id).collection('members').document(user_id).update({
        "role": role_update.role
    })

    # Get updated member data
    updated_member = db.collection('communities').document(community_id).collection('members').document(user_id).get().to_dict()

    return CommunityMember(
        user_id=updated_member['user_id'],
        name=updated_member.get('name', 'Unknown'),
        photo_url=updated_member.get('photo_url'),
        joined_at=updated_member.get('joined_at', datetime.utcnow()),
        role=updated_member.get('role', 'member'),
        subscription_status=updated_member.get('subscription_status')
    )
