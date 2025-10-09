"""
ReddyGo Subscriptions Router

Multi-tier subscription system with Individual, Family, and Creator Pro plans.
Includes family subscriptions with unlimited members and founder revenue sharing.

All endpoints require Firebase authentication via Bearer token in Authorization header.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from firebase_client import get_firestore_client
from firebase_admin import firestore
from auth import get_current_user
import secrets
import string

router = APIRouter()


# ============================================================================
# Constants
# ============================================================================

SUBSCRIPTION_PRICES = {
    "pro_individual": 14.99,
    "pro_family": 14.99,
    "creator_pro": 24.99
}

FAMILY_FOUNDER_REVENUE_SHARE = 0.20  # 20% of member subscriptions

FOUNDER_MILESTONES = {
    3: {"merchandise": "3_member_shirt", "name": "Family Starter T-Shirt"},
    5: {"merchandise": "5_member_bottle", "name": "Family Growth Water Bottle"},
    10: {"merchandise": "10_member_logo", "name": "Custom Logo Unlock"},
    20: {"merchandise": "20_member_pack", "name": "Premium Merchandise Pack"},
    50: {"merchandise": "50_member_lifetime", "name": "Lifetime Pro Subscription"},
    100: {"merchandise": "100_member_verified", "name": "Verified Founder Badge"}
}


# ============================================================================
# Pydantic Models
# ============================================================================

class SubscriptionCreate(BaseModel):
    """Create individual subscription."""
    tier: str = Field(..., pattern="^(pro_individual|creator_pro)$")
    billing_cycle: str = Field("monthly", pattern="^(monthly|annual)$")
    payment_method_id: Optional[str] = None  # Stripe payment method


class SubscriptionResponse(BaseModel):
    """Subscription details."""
    id: str
    user_id: str
    tier: str
    status: str
    price: float
    billing_cycle: str
    created_at: datetime
    expires_at: datetime
    auto_renew: bool
    is_family_member: bool = False
    family_subscription_id: Optional[str] = None


class FamilySubscriptionCreate(BaseModel):
    """Create family subscription."""
    custom_name: str = Field(..., min_length=3, max_length=50)
    custom_logo_url: Optional[str] = None
    billing_cycle: str = Field("monthly", pattern="^(monthly|annual)$")
    payment_method_id: Optional[str] = None


class FamilyMemberResponse(BaseModel):
    """Family member details."""
    user_id: str
    name: str
    photo_url: Optional[str]
    role: str  # owner | member
    joined_at: datetime


class FamilySubscriptionResponse(BaseModel):
    """Family subscription details."""
    id: str
    owner_id: str
    owner_name: str
    status: str
    price: float
    created_at: datetime
    total_members: int
    active_members_this_week: int
    custom_name: str
    custom_logo_url: Optional[str]
    founder_benefits: Dict[str, Any]
    monthly_founder_revenue: float
    is_member: bool = False
    user_role: Optional[str] = None


class FamilyInviteCreate(BaseModel):
    """Generate family invite code."""
    max_uses: int = Field(10, ge=1, le=100)
    expires_in_days: int = Field(30, ge=1, le=365)


class FamilyInviteResponse(BaseModel):
    """Invite code details."""
    code: str
    uses: int
    max_uses: int
    expires_at: datetime
    is_valid: bool


class FounderBenefitsResponse(BaseModel):
    """Founder benefits tracking."""
    total_members: int
    monthly_revenue: float
    lifetime_revenue: float
    merchandise_claimed: List[str]
    available_merchandise: List[Dict[str, str]]
    custom_name: str
    custom_logo_url: Optional[str]
    next_milestone: Optional[Dict[str, Any]]


class FamilyCustomization(BaseModel):
    """Update family customization."""
    custom_name: Optional[str] = Field(None, min_length=3, max_length=50)
    custom_logo_url: Optional[str] = None


# ============================================================================
# Individual Subscriptions
# ============================================================================

@router.post("/create", response_model=SubscriptionResponse)
async def create_subscription(
    subscription: SubscriptionCreate,
    current_user: str = Depends(get_current_user)
):
    """
    Create a Pro Individual or Creator Pro subscription.

    Requires Firebase authentication.

    Validates:
    - User doesn't have an active subscription
    - User exists
    - Payment method is valid (TODO: Stripe integration)
    """
    db = get_firestore_client()

    # Get user
    user_doc = db.collection('users').document(current_user).get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user_doc.to_dict()

    # Check if user already has active subscription
    existing_sub = user_data.get('subscription', {})
    if existing_sub.get('status') == 'active':
        raise HTTPException(
            status_code=400,
            detail=f"Already have active {existing_sub.get('tier')} subscription"
        )

    # Creator Pro requires existing Pro subscription
    if subscription.tier == 'creator_pro':
        if existing_sub.get('tier') not in ['pro_individual', 'pro_family']:
            raise HTTPException(
                status_code=400,
                detail="Creator Pro requires an active Pro Individual or Pro Family subscription"
            )

    # TODO: Create Stripe customer and subscription
    # stripe_customer = create_stripe_customer(current_user, subscription.payment_method_id)
    # stripe_subscription = create_stripe_subscription(stripe_customer, subscription.tier)

    # Calculate expiration
    if subscription.billing_cycle == 'monthly':
        expires_at = datetime.utcnow() + timedelta(days=30)
    else:  # annual
        expires_at = datetime.utcnow() + timedelta(days=365)

    # Create subscription
    subscription_ref = db.collection('subscriptions').document()
    subscription_id = subscription_ref.id

    subscription_data = {
        "type": subscription.tier,
        "user_id": current_user,
        "status": "active",
        "price": SUBSCRIPTION_PRICES[subscription.tier],
        "billing_cycle": subscription.billing_cycle,
        "created_at": firestore.SERVER_TIMESTAMP,
        "expires_at": expires_at,
        "auto_renew": True,
        "stripe": {
            "customer_id": "cus_placeholder",  # TODO: Real Stripe customer ID
            "subscription_id": "sub_placeholder",  # TODO: Real Stripe subscription ID
            "payment_method_id": subscription.payment_method_id or "pm_placeholder"
        }
    }

    subscription_ref.set(subscription_data)

    # Update user subscription info
    db.collection('users').document(current_user).update({
        "subscription": {
            "tier": subscription.tier,
            "status": "active",
            "expires_at": expires_at
        }
    })

    return SubscriptionResponse(
        id=subscription_id,
        user_id=current_user,
        tier=subscription.tier,
        status="active",
        price=SUBSCRIPTION_PRICES[subscription.tier],
        billing_cycle=subscription.billing_cycle,
        created_at=datetime.utcnow(),
        expires_at=expires_at,
        auto_renew=True,
        is_family_member=False
    )


@router.get("/", response_model=SubscriptionResponse)
async def get_subscription(current_user: str = Depends(get_current_user)):
    """
    Get user's active subscription.

    Requires Firebase authentication.
    """
    db = get_firestore_client()

    # Get user's subscription
    user_doc = db.collection('users').document(current_user).get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user_doc.to_dict()
    subscription_info = user_data.get('subscription', {})

    if subscription_info.get('tier') == 'free':
        raise HTTPException(status_code=404, detail="No active subscription")

    # Check if family member
    is_family_member = subscription_info.get('family_subscription_id') is not None
    family_subscription_id = subscription_info.get('family_subscription_id')

    # Get subscription details
    if is_family_member:
        # User is a family member, get family subscription
        family_sub_doc = db.collection('family_subscriptions').document(family_subscription_id).get()
        if not family_sub_doc.exists:
            raise HTTPException(status_code=404, detail="Family subscription not found")

        family_sub_data = family_sub_doc.to_dict()

        return SubscriptionResponse(
            id=family_subscription_id,
            user_id=current_user,
            tier="pro_family",
            status=family_sub_data.get('status', 'active'),
            price=SUBSCRIPTION_PRICES['pro_family'],
            billing_cycle=family_sub_data.get('billing_cycle', 'monthly'),
            created_at=family_sub_data.get('created_at', datetime.utcnow()),
            expires_at=family_sub_data.get('expires_at', datetime.utcnow()),
            auto_renew=family_sub_data.get('auto_renew', True),
            is_family_member=True,
            family_subscription_id=family_subscription_id
        )
    else:
        # Individual subscription
        subscriptions = db.collection('subscriptions').where(
            'user_id', '==', current_user
        ).where('status', '==', 'active').limit(1).stream()

        subscription_list = list(subscriptions)
        if not subscription_list:
            raise HTTPException(status_code=404, detail="No active subscription found")

        subscription_doc = subscription_list[0]
        subscription_data = subscription_doc.to_dict()

        return SubscriptionResponse(
            id=subscription_doc.id,
            user_id=current_user,
            tier=subscription_data['type'],
            status=subscription_data['status'],
            price=subscription_data['price'],
            billing_cycle=subscription_data['billing_cycle'],
            created_at=subscription_data.get('created_at', datetime.utcnow()),
            expires_at=subscription_data.get('expires_at', datetime.utcnow()),
            auto_renew=subscription_data.get('auto_renew', True),
            is_family_member=False
        )


@router.post("/cancel", status_code=204)
async def cancel_subscription(current_user: str = Depends(get_current_user)):
    """
    Cancel user's subscription.

    Requires Firebase authentication.
    Subscription remains active until expiration date.
    """
    db = get_firestore_client()

    # Get user's subscription
    subscriptions = db.collection('subscriptions').where(
        'user_id', '==', current_user
    ).where('status', '==', 'active').limit(1).stream()

    subscription_list = list(subscriptions)
    if not subscription_list:
        raise HTTPException(status_code=404, detail="No active subscription to cancel")

    subscription_doc = subscription_list[0]
    subscription_id = subscription_doc.id
    subscription_data = subscription_doc.to_dict()

    # TODO: Cancel Stripe subscription
    # cancel_stripe_subscription(subscription_data['stripe']['subscription_id'])

    # Update subscription
    db.collection('subscriptions').document(subscription_id).update({
        "status": "canceled",
        "auto_renew": False,
        "canceled_at": firestore.SERVER_TIMESTAMP
    })

    # Update user subscription status
    db.collection('users').document(current_user).update({
        "subscription.status": "canceled"
    })

    return None


@router.post("/renew", response_model=SubscriptionResponse)
async def renew_subscription(current_user: str = Depends(get_current_user)):
    """
    Renew a canceled subscription.

    Requires Firebase authentication.
    """
    db = get_firestore_client()

    # Get user's subscription
    subscriptions = db.collection('subscriptions').where(
        'user_id', '==', current_user
    ).where('status', '==', 'canceled').limit(1).stream()

    subscription_list = list(subscriptions)
    if not subscription_list:
        raise HTTPException(status_code=404, detail="No canceled subscription to renew")

    subscription_doc = subscription_list[0]
    subscription_id = subscription_doc.id
    subscription_data = subscription_doc.to_dict()

    # TODO: Renew Stripe subscription
    # renew_stripe_subscription(subscription_data['stripe']['subscription_id'])

    # Calculate new expiration
    billing_cycle = subscription_data['billing_cycle']
    if billing_cycle == 'monthly':
        expires_at = datetime.utcnow() + timedelta(days=30)
    else:
        expires_at = datetime.utcnow() + timedelta(days=365)

    # Update subscription
    db.collection('subscriptions').document(subscription_id).update({
        "status": "active",
        "auto_renew": True,
        "expires_at": expires_at,
        "renewed_at": firestore.SERVER_TIMESTAMP
    })

    # Update user subscription status
    db.collection('users').document(current_user).update({
        "subscription.status": "active",
        "subscription.expires_at": expires_at
    })

    subscription_data['status'] = 'active'
    subscription_data['expires_at'] = expires_at

    return SubscriptionResponse(
        id=subscription_id,
        user_id=current_user,
        tier=subscription_data['type'],
        status="active",
        price=subscription_data['price'],
        billing_cycle=subscription_data['billing_cycle'],
        created_at=subscription_data.get('created_at', datetime.utcnow()),
        expires_at=expires_at,
        auto_renew=True,
        is_family_member=False
    )


# ============================================================================
# Family Subscriptions
# ============================================================================

def generate_invite_code(prefix: str = "FAM") -> str:
    """Generate unique family invite code."""
    random_part = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    return f"{prefix}_{random_part}"


def calculate_founder_revenue(member_count: int) -> float:
    """Calculate monthly founder revenue from family members."""
    if member_count <= 1:
        return 0.0
    # Founder doesn't pay themselves, so subtract 1
    return (member_count - 1) * SUBSCRIPTION_PRICES['pro_family'] * FAMILY_FOUNDER_REVENUE_SHARE


def get_next_milestone(current_members: int, claimed_milestones: List[str]) -> Optional[Dict[str, Any]]:
    """Get next available founder milestone."""
    for member_count in sorted(FOUNDER_MILESTONES.keys()):
        if member_count > current_members:
            milestone = FOUNDER_MILESTONES[member_count]
            if milestone['merchandise'] not in claimed_milestones:
                return {
                    "member_count": member_count,
                    "merchandise": milestone['merchandise'],
                    "name": milestone['name']
                }
    return None


@router.post("/family/create", response_model=FamilySubscriptionResponse)
async def create_family_subscription(
    family_sub: FamilySubscriptionCreate,
    current_user: str = Depends(get_current_user)
):
    """
    Create a Pro Family subscription.

    Requires Firebase authentication. User becomes the family founder.

    Benefits:
    - Unlimited family members
    - 20% revenue share from all member subscriptions
    - Founder milestones: free merchandise at 3, 5, 10, 20, 50, 100 members
    - Custom family name and logo
    """
    db = get_firestore_client()

    # Get user
    user_doc = db.collection('users').document(current_user).get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user_doc.to_dict()

    # Check if user already has active subscription
    existing_sub = user_data.get('subscription', {})
    if existing_sub.get('status') == 'active':
        raise HTTPException(
            status_code=400,
            detail=f"Already have active {existing_sub.get('tier')} subscription. Please cancel it first."
        )

    # TODO: Create Stripe customer and subscription
    # stripe_customer = create_stripe_customer(current_user, family_sub.payment_method_id)
    # stripe_subscription = create_stripe_subscription(stripe_customer, 'pro_family')

    # Calculate expiration
    if family_sub.billing_cycle == 'monthly':
        expires_at = datetime.utcnow() + timedelta(days=30)
    else:
        expires_at = datetime.utcnow() + timedelta(days=365)

    # Create family subscription
    family_sub_ref = db.collection('family_subscriptions').document()
    family_sub_id = family_sub_ref.id

    family_sub_data = {
        "subscription_id": family_sub_id,
        "owner_id": current_user,
        "owner_name": user_data.get('name', 'Unknown'),
        "plan": "pro_family",
        "status": "active",
        "price": SUBSCRIPTION_PRICES['pro_family'],
        "billing_cycle": family_sub.billing_cycle,
        "created_at": firestore.SERVER_TIMESTAMP,
        "expires_at": expires_at,
        "auto_renew": True,
        "members": [
            {
                "user_id": current_user,
                "name": user_data.get('name', 'Unknown'),
                "role": "owner",
                "joined_at": firestore.SERVER_TIMESTAMP
            }
        ],
        "founder_benefits": {
            "founder_badge": True,
            "revenue_share_percentage": 20,
            "lifetime_revenue_earned": 0.00,
            "merchandise_claimed": [],
            "custom_name": family_sub.custom_name,
            "custom_logo_url": family_sub.custom_logo_url
        },
        "stats": {
            "total_members": 1,
            "active_members_this_week": 1,
            "total_challenges_completed": 0,
            "total_distance_km": 0
        },
        "invite_codes": [],
        "stripe": {
            "customer_id": "cus_placeholder",
            "subscription_id": "sub_placeholder",
            "payment_method_id": family_sub.payment_method_id or "pm_placeholder"
        }
    }

    family_sub_ref.set(family_sub_data)

    # Update user subscription info
    db.collection('users').document(current_user).update({
        "subscription": {
            "tier": "pro_family",
            "status": "active",
            "expires_at": expires_at,
            "family_subscription_id": family_sub_id
        }
    })

    # Create founder rewards tracking
    db.collection('founder_rewards').document(current_user).set({
        "user_id": current_user,
        "family_founder": {
            "subscription_id": family_sub_id,
            "total_members": 1,
            "revenue_earned": 0.00,
            "merchandise_claimed": [],
            "milestones_achieved": []
        }
    }, merge=True)

    return FamilySubscriptionResponse(
        id=family_sub_id,
        owner_id=current_user,
        owner_name=user_data.get('name', 'Unknown'),
        status="active",
        price=SUBSCRIPTION_PRICES['pro_family'],
        created_at=datetime.utcnow(),
        total_members=1,
        active_members_this_week=1,
        custom_name=family_sub.custom_name,
        custom_logo_url=family_sub.custom_logo_url,
        founder_benefits=family_sub_data['founder_benefits'],
        monthly_founder_revenue=0.0,
        is_member=True,
        user_role="owner"
    )


@router.get("/family/{subscription_id}", response_model=FamilySubscriptionResponse)
async def get_family_subscription(
    subscription_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Get family subscription details.

    Requires Firebase authentication.
    """
    db = get_firestore_client()

    # Get family subscription
    family_sub_doc = db.collection('family_subscriptions').document(subscription_id).get()
    if not family_sub_doc.exists:
        raise HTTPException(status_code=404, detail="Family subscription not found")

    family_sub_data = family_sub_doc.to_dict()

    # Check if user is a member
    is_member = any(member['user_id'] == current_user for member in family_sub_data.get('members', []))
    user_role = next(
        (member['role'] for member in family_sub_data.get('members', []) if member['user_id'] == current_user),
        None
    )

    # Calculate monthly founder revenue
    total_members = family_sub_data.get('stats', {}).get('total_members', 1)
    monthly_revenue = calculate_founder_revenue(total_members)

    return FamilySubscriptionResponse(
        id=subscription_id,
        owner_id=family_sub_data['owner_id'],
        owner_name=family_sub_data.get('owner_name', 'Unknown'),
        status=family_sub_data['status'],
        price=family_sub_data['price'],
        created_at=family_sub_data.get('created_at', datetime.utcnow()),
        total_members=total_members,
        active_members_this_week=family_sub_data.get('stats', {}).get('active_members_this_week', 0),
        custom_name=family_sub_data.get('founder_benefits', {}).get('custom_name', 'Family'),
        custom_logo_url=family_sub_data.get('founder_benefits', {}).get('custom_logo_url'),
        founder_benefits=family_sub_data.get('founder_benefits', {}),
        monthly_founder_revenue=monthly_revenue,
        is_member=is_member,
        user_role=user_role
    )


@router.post("/family/{subscription_id}/generate-invite", response_model=FamilyInviteResponse)
async def generate_family_invite(
    subscription_id: str,
    invite: FamilyInviteCreate,
    current_user: str = Depends(get_current_user)
):
    """
    Generate a family invite code.

    Requires Firebase authentication and family ownership.
    """
    db = get_firestore_client()

    # Get family subscription
    family_sub_doc = db.collection('family_subscriptions').document(subscription_id).get()
    if not family_sub_doc.exists:
        raise HTTPException(status_code=404, detail="Family subscription not found")

    family_sub_data = family_sub_doc.to_dict()

    # Check if user is owner
    if family_sub_data['owner_id'] != current_user:
        raise HTTPException(status_code=403, detail="Only family owner can generate invite codes")

    # Generate unique invite code
    invite_code = generate_invite_code()
    expires_at = datetime.utcnow() + timedelta(days=invite.expires_in_days)

    invite_data = {
        "code": invite_code,
        "uses": 0,
        "max_uses": invite.max_uses,
        "expires_at": expires_at,
        "created_at": firestore.SERVER_TIMESTAMP
    }

    # Add invite code to family subscription
    db.collection('family_subscriptions').document(subscription_id).update({
        "invite_codes": firestore.ArrayUnion([invite_data])
    })

    return FamilyInviteResponse(
        code=invite_code,
        uses=0,
        max_uses=invite.max_uses,
        expires_at=expires_at,
        is_valid=True
    )


@router.post("/family/join/{invite_code}", response_model=FamilyMemberResponse)
async def join_family_subscription(
    invite_code: str,
    current_user: str = Depends(get_current_user)
):
    """
    Join a family subscription using an invite code.

    Requires Firebase authentication.

    Benefits:
    - Get Pro Family benefits without paying
    - Join unlimited family challenges
    - Contribute to family stats
    """
    db = get_firestore_client()

    # Find family subscription with this invite code
    family_subs = db.collection('family_subscriptions').where(
        'status', '==', 'active'
    ).stream()

    family_sub_doc = None
    invite_data = None
    for sub in family_subs:
        sub_data = sub.to_dict()
        for invite in sub_data.get('invite_codes', []):
            if invite['code'] == invite_code:
                family_sub_doc = sub
                family_sub_data = sub_data
                invite_data = invite
                break
        if family_sub_doc:
            break

    if not family_sub_doc:
        raise HTTPException(status_code=404, detail="Invalid invite code")

    # Validate invite
    if invite_data['uses'] >= invite_data['max_uses']:
        raise HTTPException(status_code=400, detail="Invite code has reached maximum uses")

    if datetime.utcnow() > invite_data['expires_at']:
        raise HTTPException(status_code=400, detail="Invite code has expired")

    # Check if user already has active subscription
    user_doc = db.collection('users').document(current_user).get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user_doc.to_dict()
    existing_sub = user_data.get('subscription', {})
    if existing_sub.get('status') == 'active':
        raise HTTPException(
            status_code=400,
            detail=f"Already have active {existing_sub.get('tier')} subscription. Please cancel it first."
        )

    # Add user to family
    new_member = {
        "user_id": current_user,
        "name": user_data.get('name', 'Unknown'),
        "role": "member",
        "joined_at": firestore.SERVER_TIMESTAMP
    }

    db.collection('family_subscriptions').document(family_sub_doc.id).update({
        "members": firestore.ArrayUnion([new_member]),
        "stats.total_members": firestore.Increment(1),
        "stats.active_members_this_week": firestore.Increment(1)
    })

    # Update invite code uses
    # Find and update the specific invite code in the array
    updated_invites = family_sub_data.get('invite_codes', [])
    for i, inv in enumerate(updated_invites):
        if inv['code'] == invite_code:
            updated_invites[i]['uses'] += 1
            break

    db.collection('family_subscriptions').document(family_sub_doc.id).update({
        "invite_codes": updated_invites
    })

    # Update user subscription
    db.collection('users').document(current_user).update({
        "subscription": {
            "tier": "pro_family",
            "status": "active",
            "expires_at": family_sub_data.get('expires_at', datetime.utcnow()),
            "family_subscription_id": family_sub_doc.id
        }
    })

    # Check for founder milestone achievement
    new_member_count = family_sub_data.get('stats', {}).get('total_members', 0) + 1
    if new_member_count in FOUNDER_MILESTONES:
        milestone = FOUNDER_MILESTONES[new_member_count]
        # Send notification to founder
        from realtime_db_client import send_realtime_notification
        send_realtime_notification(family_sub_data['owner_id'], {
            "type": "founder_milestone",
            "title": "Founder Milestone Achieved!",
            "body": f"Your family reached {new_member_count} members! Claim: {milestone['name']}",
            "data": {
                "milestone": new_member_count,
                "merchandise": milestone['merchandise']
            }
        })

    # Send notification to founder about new member
    from realtime_db_client import send_realtime_notification
    send_realtime_notification(family_sub_data['owner_id'], {
        "type": "family_new_member",
        "title": "New Family Member",
        "body": f"{user_data.get('name')} joined your family subscription!",
        "data": {"user_id": current_user}
    })

    return FamilyMemberResponse(
        user_id=current_user,
        name=user_data.get('name', 'Unknown'),
        photo_url=user_data.get('profile_photo_url'),
        role="member",
        joined_at=datetime.utcnow()
    )


@router.post("/family/{subscription_id}/leave", status_code=204)
async def leave_family_subscription(
    subscription_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Leave a family subscription.

    Requires Firebase authentication.
    Family owner cannot leave (must cancel subscription instead).
    """
    db = get_firestore_client()

    # Get family subscription
    family_sub_doc = db.collection('family_subscriptions').document(subscription_id).get()
    if not family_sub_doc.exists:
        raise HTTPException(status_code=404, detail="Family subscription not found")

    family_sub_data = family_sub_doc.to_dict()

    # Check if user is owner
    if family_sub_data['owner_id'] == current_user:
        raise HTTPException(
            status_code=400,
            detail="Family owner cannot leave. Please cancel the subscription or transfer ownership."
        )

    # Find and remove member
    members = family_sub_data.get('members', [])
    member_to_remove = None
    for member in members:
        if member['user_id'] == current_user:
            member_to_remove = member
            break

    if not member_to_remove:
        raise HTTPException(status_code=400, detail="Not a member of this family")

    # Remove member from family
    db.collection('family_subscriptions').document(subscription_id).update({
        "members": firestore.ArrayRemove([member_to_remove]),
        "stats.total_members": firestore.Increment(-1)
    })

    # Update user subscription to free
    db.collection('users').document(current_user).update({
        "subscription": {
            "tier": "free",
            "status": "active"
        }
    })

    return None


@router.get("/family/{subscription_id}/members", response_model=List[FamilyMemberResponse])
async def get_family_members(
    subscription_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Get family subscription members.

    Requires Firebase authentication and family membership.
    """
    db = get_firestore_client()

    # Get family subscription
    family_sub_doc = db.collection('family_subscriptions').document(subscription_id).get()
    if not family_sub_doc.exists:
        raise HTTPException(status_code=404, detail="Family subscription not found")

    family_sub_data = family_sub_doc.to_dict()

    # Check if user is a member
    is_member = any(member['user_id'] == current_user for member in family_sub_data.get('members', []))
    if not is_member:
        raise HTTPException(status_code=403, detail="Must be a family member to view members")

    # Get member details
    members = []
    for member in family_sub_data.get('members', []):
        # Get user photo from users collection
        user_doc = db.collection('users').document(member['user_id']).get()
        photo_url = None
        if user_doc.exists:
            photo_url = user_doc.to_dict().get('profile_photo_url')

        members.append(FamilyMemberResponse(
            user_id=member['user_id'],
            name=member.get('name', 'Unknown'),
            photo_url=photo_url,
            role=member.get('role', 'member'),
            joined_at=member.get('joined_at', datetime.utcnow())
        ))

    return members


# ============================================================================
# Founder Benefits
# ============================================================================

@router.get("/family/founder-benefits", response_model=FounderBenefitsResponse)
async def get_founder_benefits(current_user: str = Depends(get_current_user)):
    """
    Get founder benefits for family subscription.

    Requires Firebase authentication and family ownership.
    """
    db = get_firestore_client()

    # Find user's family subscription where they are owner
    family_subs = db.collection('family_subscriptions').where(
        'owner_id', '==', current_user
    ).where('status', '==', 'active').limit(1).stream()

    family_sub_list = list(family_subs)
    if not family_sub_list:
        raise HTTPException(status_code=404, detail="No family subscription found where you are the owner")

    family_sub = family_sub_list[0]
    family_sub_data = family_sub.to_dict()

    founder_benefits = family_sub_data.get('founder_benefits', {})
    total_members = family_sub_data.get('stats', {}).get('total_members', 1)
    monthly_revenue = calculate_founder_revenue(total_members)
    lifetime_revenue = founder_benefits.get('lifetime_revenue_earned', 0.0)
    claimed_merch = founder_benefits.get('merchandise_claimed', [])

    # Get available unclaimed merchandise
    available_merch = []
    for member_count in sorted(FOUNDER_MILESTONES.keys()):
        if member_count <= total_members:
            milestone = FOUNDER_MILESTONES[member_count]
            if milestone['merchandise'] not in claimed_merch:
                available_merch.append({
                    "milestone": member_count,
                    "merchandise": milestone['merchandise'],
                    "name": milestone['name']
                })

    # Get next milestone
    next_milestone = get_next_milestone(total_members, claimed_merch)

    return FounderBenefitsResponse(
        total_members=total_members,
        monthly_revenue=monthly_revenue,
        lifetime_revenue=lifetime_revenue,
        merchandise_claimed=claimed_merch,
        available_merchandise=available_merch,
        custom_name=founder_benefits.get('custom_name', 'Family'),
        custom_logo_url=founder_benefits.get('custom_logo_url'),
        next_milestone=next_milestone
    )


@router.post("/family/claim-merchandise/{merchandise_id}", status_code=204)
async def claim_founder_merchandise(
    merchandise_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Claim founder milestone merchandise.

    Requires Firebase authentication and family ownership.
    """
    db = get_firestore_client()

    # Find user's family subscription
    family_subs = db.collection('family_subscriptions').where(
        'owner_id', '==', current_user
    ).where('status', '==', 'active').limit(1).stream()

    family_sub_list = list(family_subs)
    if not family_sub_list:
        raise HTTPException(status_code=404, detail="No family subscription found")

    family_sub = family_sub_list[0]
    family_sub_data = family_sub.to_dict()

    founder_benefits = family_sub_data.get('founder_benefits', {})
    total_members = family_sub_data.get('stats', {}).get('total_members', 1)
    claimed_merch = founder_benefits.get('merchandise_claimed', [])

    # Check if merchandise is available
    merchandise_available = False
    for member_count in FOUNDER_MILESTONES.keys():
        milestone = FOUNDER_MILESTONES[member_count]
        if milestone['merchandise'] == merchandise_id and member_count <= total_members:
            merchandise_available = True
            break

    if not merchandise_available:
        raise HTTPException(status_code=400, detail="Merchandise not available or milestone not reached")

    if merchandise_id in claimed_merch:
        raise HTTPException(status_code=400, detail="Merchandise already claimed")

    # Claim merchandise
    db.collection('family_subscriptions').document(family_sub.id).update({
        "founder_benefits.merchandise_claimed": firestore.ArrayUnion([merchandise_id])
    })

    # Update founder rewards
    db.collection('founder_rewards').document(current_user).update({
        "family_founder.merchandise_claimed": firestore.ArrayUnion([merchandise_id])
    })

    return None


@router.put("/family/customize", response_model=FamilySubscriptionResponse)
async def customize_family(
    customization: FamilyCustomization,
    current_user: str = Depends(get_current_user)
):
    """
    Update family customization (name and logo).

    Requires Firebase authentication and family ownership.
    Custom logo unlocked at 10+ members milestone.
    """
    db = get_firestore_client()

    # Find user's family subscription
    family_subs = db.collection('family_subscriptions').where(
        'owner_id', '==', current_user
    ).where('status', '==', 'active').limit(1).stream()

    family_sub_list = list(family_subs)
    if not family_sub_list:
        raise HTTPException(status_code=404, detail="No family subscription found")

    family_sub = family_sub_list[0]
    family_sub_data = family_sub.to_dict()

    total_members = family_sub_data.get('stats', {}).get('total_members', 1)

    # Check if logo customization is unlocked
    if customization.custom_logo_url and total_members < 10:
        raise HTTPException(
            status_code=403,
            detail="Custom logo unlocked at 10+ members milestone"
        )

    # Build update dict
    update_dict = {}
    if customization.custom_name is not None:
        update_dict['founder_benefits.custom_name'] = customization.custom_name
    if customization.custom_logo_url is not None:
        update_dict['founder_benefits.custom_logo_url'] = customization.custom_logo_url

    if not update_dict:
        raise HTTPException(status_code=400, detail="No customization provided")

    # Update family subscription
    db.collection('family_subscriptions').document(family_sub.id).update(update_dict)

    # Get updated data
    updated_family = db.collection('family_subscriptions').document(family_sub.id).get().to_dict()
    founder_benefits = updated_family.get('founder_benefits', {})

    return FamilySubscriptionResponse(
        id=family_sub.id,
        owner_id=current_user,
        owner_name=updated_family.get('owner_name', 'Unknown'),
        status=updated_family['status'],
        price=updated_family['price'],
        created_at=updated_family.get('created_at', datetime.utcnow()),
        total_members=total_members,
        active_members_this_week=updated_family.get('stats', {}).get('active_members_this_week', 0),
        custom_name=founder_benefits.get('custom_name', 'Family'),
        custom_logo_url=founder_benefits.get('custom_logo_url'),
        founder_benefits=founder_benefits,
        monthly_founder_revenue=calculate_founder_revenue(total_members),
        is_member=True,
        user_role="owner"
    )
