"""
ReddyGo Creator Marketplace Router

Enables creators to sell digital fitness content including workout plans, meal plans,
training programs, and coaching packages.

Creator Pro subscribers receive 85% revenue share (vs 70% for regular creators).

All endpoints require Firebase authentication via Bearer token in Authorization header.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from firebase_client import get_firestore_client
from realtime_db_client import get_realtime_db
from firebase_admin import firestore
from auth import get_current_user
from enum import Enum

router = APIRouter()


# ============================================================================
# Enums and Constants
# ============================================================================

class ProductType(str, Enum):
    """Types of products creators can sell."""
    WORKOUT_PLAN = "workout_plan"
    MEAL_PLAN = "meal_plan"
    TRAINING_PROGRAM = "training_program"
    COACHING_PACKAGE = "coaching_package"
    DIGITAL_GUIDE = "digital_guide"
    VIDEO_COURSE = "video_course"


class ProductStatus(str, Enum):
    """Product listing status."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


# Revenue splits
CREATOR_PRO_REVENUE_SHARE = 0.85  # Creator Pro subscribers get 85%
CREATOR_STANDARD_REVENUE_SHARE = 0.70  # Standard creators get 70%
PLATFORM_FEE = 0.15  # Platform takes 15% (or 30% for standard)

# Bundle discount limits
MAX_BUNDLE_DISCOUNT = 0.30  # Max 30% discount for bundles


# ============================================================================
# Pydantic Models
# ============================================================================

class ProductCreate(BaseModel):
    """Create new marketplace product."""
    title: str = Field(..., min_length=5, max_length=100)
    description: str = Field(..., min_length=20, max_length=2000)
    product_type: ProductType
    price: float = Field(..., ge=0.99, le=999.99)
    tags: List[str] = Field(default=[], max_items=10)
    difficulty_level: Optional[str] = Field(None, pattern="^(beginner|intermediate|advanced)$")
    duration_weeks: Optional[int] = Field(None, ge=1, le=52)
    preview_images: List[str] = Field(default=[], max_items=5)
    preview_video_url: Optional[str] = None
    downloadable_files: List[Dict[str, str]] = Field(default=[])
    requirements: List[str] = Field(default=[])


class ProductUpdate(BaseModel):
    """Update existing product."""
    title: Optional[str] = Field(None, min_length=5, max_length=100)
    description: Optional[str] = Field(None, min_length=20, max_length=2000)
    price: Optional[float] = Field(None, ge=0.99, le=999.99)
    tags: Optional[List[str]] = Field(None, max_items=10)
    difficulty_level: Optional[str] = Field(None, pattern="^(beginner|intermediate|advanced)$")
    duration_weeks: Optional[int] = Field(None, ge=1, le=52)
    preview_images: Optional[List[str]] = Field(None, max_items=5)
    preview_video_url: Optional[str] = None
    downloadable_files: Optional[List[Dict[str, str]]] = None
    requirements: Optional[List[str]] = None
    status: Optional[ProductStatus] = None


class ProductResponse(BaseModel):
    """Product listing response."""
    id: str
    creator_id: str
    creator_name: str
    creator_verified: bool
    title: str
    description: str
    product_type: str
    price: float
    tags: List[str]
    difficulty_level: Optional[str]
    duration_weeks: Optional[int]
    preview_images: List[str]
    preview_video_url: Optional[str]
    rating: float
    total_ratings: int
    total_purchases: int
    status: str
    created_at: str
    updated_at: str


class PurchaseRequest(BaseModel):
    """Purchase product request."""
    product_id: str
    # TODO: Add payment_method_id for Stripe integration


class PurchaseResponse(BaseModel):
    """Purchase confirmation."""
    purchase_id: str
    product_id: str
    product_title: str
    price: float
    creator_revenue: float
    platform_fee: float
    purchased_at: str
    download_links: List[Dict[str, str]]


class BundleCreate(BaseModel):
    """Create product bundle."""
    title: str = Field(..., min_length=5, max_length=100)
    description: str = Field(..., min_length=20, max_length=1000)
    product_ids: List[str] = Field(..., min_items=2, max_items=10)
    discount_percentage: float = Field(..., ge=5, le=30)  # 5-30% discount


class BundleResponse(BaseModel):
    """Bundle information."""
    id: str
    creator_id: str
    creator_name: str
    title: str
    description: str
    products: List[Dict[str, Any]]
    original_price: float
    discount_percentage: float
    bundle_price: float
    savings: float
    total_purchases: int
    created_at: str


class ReviewCreate(BaseModel):
    """Create product review."""
    product_id: str
    rating: int = Field(..., ge=1, le=5)
    review_text: Optional[str] = Field(None, max_length=1000)
    helpful_features: List[str] = Field(default=[])


class ReviewResponse(BaseModel):
    """Product review."""
    id: str
    product_id: str
    user_id: str
    user_name: str
    user_avatar: Optional[str]
    rating: int
    review_text: Optional[str]
    helpful_features: List[str]
    verified_purchase: bool
    helpful_count: int
    created_at: str


class CreatorEarningsResponse(BaseModel):
    """Creator earnings summary."""
    total_revenue: float
    platform_fees: float
    net_earnings: float
    total_sales: int
    revenue_share_percentage: float
    has_creator_pro: bool
    monthly_breakdown: List[Dict[str, Any]]
    top_products: List[Dict[str, Any]]


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_revenue_split(price: float, has_creator_pro: bool) -> Dict[str, float]:
    """Calculate revenue split between creator and platform."""
    creator_share = CREATOR_PRO_REVENUE_SHARE if has_creator_pro else CREATOR_STANDARD_REVENUE_SHARE
    platform_share = 1.0 - creator_share

    return {
        "total_price": price,
        "creator_revenue": price * creator_share,
        "platform_fee": price * platform_share,
        "creator_share_percentage": creator_share * 100
    }


def is_creator(db, user_id: str) -> bool:
    """Check if user is an approved creator."""
    user_doc = db.collection('users').document(user_id).get()
    if not user_doc.exists:
        return False

    user_data = user_doc.to_dict()
    return user_data.get('is_creator', False)


def has_creator_pro_subscription(db, user_id: str) -> bool:
    """Check if user has active Creator Pro subscription."""
    subscription_query = db.collection('subscriptions').where(
        'user_id', '==', user_id
    ).where('tier', '==', 'creator_pro').where(
        'status', '==', 'active'
    ).limit(1).stream()

    return len(list(subscription_query)) > 0


def has_purchased_product(db, user_id: str, product_id: str) -> bool:
    """Check if user already purchased a product."""
    purchase_query = db.collection('purchases').where(
        'user_id', '==', user_id
    ).where('product_id', '==', product_id).limit(1).stream()

    return len(list(purchase_query)) > 0


def send_realtime_notification(realtime_db, user_id: str, notification: Dict[str, Any]):
    """Send real-time notification via Firebase Realtime Database."""
    try:
        realtime_db.child('notifications').child(user_id).push(notification)
    except Exception as e:
        print(f"Failed to send notification to {user_id}: {e}")


# ============================================================================
# Product Management Endpoints
# ============================================================================

@router.post("/products", response_model=ProductResponse)
async def create_product(
    product: ProductCreate,
    current_user: str = Depends(get_current_user)
):
    """
    Create new marketplace product (creators only).

    Requires user to be an approved creator.
    """
    db = get_firestore_client()

    # Verify user is a creator
    if not is_creator(db, current_user):
        raise HTTPException(403, "Only approved creators can create marketplace products")

    # Get creator info
    creator_doc = db.collection('users').document(current_user).get()
    creator_data = creator_doc.to_dict()

    # Create product
    product_data = {
        'creator_id': current_user,
        'creator_name': creator_data.get('name', 'Unknown'),
        'creator_verified': creator_data.get('verified', False),
        'title': product.title,
        'description': product.description,
        'product_type': product.product_type.value,
        'price': product.price,
        'tags': product.tags,
        'difficulty_level': product.difficulty_level,
        'duration_weeks': product.duration_weeks,
        'preview_images': product.preview_images,
        'preview_video_url': product.preview_video_url,
        'downloadable_files': product.downloadable_files,
        'requirements': product.requirements,
        'status': ProductStatus.DRAFT.value,
        'rating': 0.0,
        'total_ratings': 0,
        'total_purchases': 0,
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }

    product_ref = db.collection('marketplace_products').add(product_data)
    product_id = product_ref[1].id

    return ProductResponse(
        id=product_id,
        **product_data,
        created_at=product_data['created_at'].isoformat(),
        updated_at=product_data['updated_at'].isoformat()
    )


@router.get("/products/{product_id}", response_model=ProductResponse)
async def get_product(
    product_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get product details."""
    db = get_firestore_client()

    product_doc = db.collection('marketplace_products').document(product_id).get()
    if not product_doc.exists:
        raise HTTPException(404, "Product not found")

    product_data = product_doc.to_dict()

    # Only show draft/paused products to their creator
    if product_data['status'] in [ProductStatus.DRAFT.value, ProductStatus.PAUSED.value]:
        if product_data['creator_id'] != current_user:
            raise HTTPException(404, "Product not found")

    return ProductResponse(
        id=product_id,
        **product_data,
        created_at=product_data['created_at'].isoformat(),
        updated_at=product_data['updated_at'].isoformat()
    )


@router.put("/products/{product_id}", response_model=ProductResponse)
async def update_product(
    product_id: str,
    updates: ProductUpdate,
    current_user: str = Depends(get_current_user)
):
    """
    Update product (creator only).

    Only the creator can update their products.
    """
    db = get_firestore_client()

    product_ref = db.collection('marketplace_products').document(product_id)
    product_doc = product_ref.get()

    if not product_doc.exists:
        raise HTTPException(404, "Product not found")

    product_data = product_doc.to_dict()

    # Verify ownership
    if product_data['creator_id'] != current_user:
        raise HTTPException(403, "You can only update your own products")

    # Apply updates
    update_dict = {k: v for k, v in updates.dict().items() if v is not None}
    if update_dict:
        update_dict['updated_at'] = datetime.utcnow()
        product_ref.update(update_dict)

        # Get updated data
        updated_doc = product_ref.get()
        updated_data = updated_doc.to_dict()

        return ProductResponse(
            id=product_id,
            **updated_data,
            created_at=updated_data['created_at'].isoformat(),
            updated_at=updated_data['updated_at'].isoformat()
        )

    return ProductResponse(
        id=product_id,
        **product_data,
        created_at=product_data['created_at'].isoformat(),
        updated_at=product_data['updated_at'].isoformat()
    )


@router.delete("/products/{product_id}")
async def delete_product(
    product_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Delete product (creator only).

    Products with purchases cannot be deleted (only archived).
    """
    db = get_firestore_client()

    product_ref = db.collection('marketplace_products').document(product_id)
    product_doc = product_ref.get()

    if not product_doc.exists:
        raise HTTPException(404, "Product not found")

    product_data = product_doc.to_dict()

    # Verify ownership
    if product_data['creator_id'] != current_user:
        raise HTTPException(403, "You can only delete your own products")

    # Prevent deletion if has purchases
    if product_data.get('total_purchases', 0) > 0:
        raise HTTPException(400, "Cannot delete products with purchases. Archive instead.")

    product_ref.delete()

    return {"success": True, "message": "Product deleted successfully"}


@router.get("/products", response_model=List[ProductResponse])
async def list_products(
    product_type: Optional[ProductType] = None,
    difficulty_level: Optional[str] = Query(None, pattern="^(beginner|intermediate|advanced)$"),
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, le=999.99),
    creator_id: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = Query("created_at", pattern="^(created_at|price|rating|total_purchases)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    limit: int = Query(50, ge=1, le=100),
    current_user: str = Depends(get_current_user)
):
    """
    List and search marketplace products.

    Supports filtering by type, difficulty, price range, creator.
    """
    db = get_firestore_client()

    # Start with base query - only show active products
    query = db.collection('marketplace_products').where('status', '==', ProductStatus.ACTIVE.value)

    # Apply filters
    if product_type:
        query = query.where('product_type', '==', product_type.value)

    if difficulty_level:
        query = query.where('difficulty_level', '==', difficulty_level)

    if creator_id:
        query = query.where('creator_id', '==', creator_id)

    # Apply sorting
    direction = firestore.Query.DESCENDING if sort_order == "desc" else firestore.Query.ASCENDING
    query = query.order_by(sort_by, direction=direction)

    # Execute query
    products = []
    for product_doc in query.limit(limit).stream():
        product_data = product_doc.to_dict()

        # Apply price filter (Firestore doesn't support range queries with other filters)
        if min_price is not None and product_data['price'] < min_price:
            continue
        if max_price is not None and product_data['price'] > max_price:
            continue

        # Apply search filter
        if search:
            search_lower = search.lower()
            if search_lower not in product_data['title'].lower() and \
               search_lower not in product_data['description'].lower():
                continue

        products.append(ProductResponse(
            id=product_doc.id,
            **product_data,
            created_at=product_data['created_at'].isoformat(),
            updated_at=product_data['updated_at'].isoformat()
        ))

    return products


@router.get("/my-products", response_model=List[ProductResponse])
async def get_my_products(
    status: Optional[ProductStatus] = None,
    current_user: str = Depends(get_current_user)
):
    """
    Get all products created by current user (includes drafts and paused).

    Shows all products regardless of status.
    """
    db = get_firestore_client()

    # Query user's products
    query = db.collection('marketplace_products').where('creator_id', '==', current_user)

    if status:
        query = query.where('status', '==', status.value)

    query = query.order_by('created_at', direction=firestore.Query.DESCENDING)

    products = []
    for product_doc in query.stream():
        product_data = product_doc.to_dict()

        products.append(ProductResponse(
            id=product_doc.id,
            **product_data,
            created_at=product_data['created_at'].isoformat(),
            updated_at=product_data['updated_at'].isoformat()
        ))

    return products


# ============================================================================
# Purchase Endpoints
# ============================================================================

@router.post("/purchase", response_model=PurchaseResponse)
async def purchase_product(
    request: PurchaseRequest,
    current_user: str = Depends(get_current_user)
):
    """
    Purchase a marketplace product.

    TODO: Integrate with Stripe for payment processing.
    """
    db = get_firestore_client()
    realtime_db = get_realtime_db()

    # Get product
    product_ref = db.collection('marketplace_products').document(request.product_id)
    product_doc = product_ref.get()

    if not product_doc.exists:
        raise HTTPException(404, "Product not found")

    product_data = product_doc.to_dict()

    # Verify product is active
    if product_data['status'] != ProductStatus.ACTIVE.value:
        raise HTTPException(400, "Product is not available for purchase")

    # Prevent buying own product
    if product_data['creator_id'] == current_user:
        raise HTTPException(400, "Cannot purchase your own product")

    # Check if already purchased
    if has_purchased_product(db, current_user, request.product_id):
        raise HTTPException(400, "You already own this product")

    # Get buyer info
    buyer_doc = db.collection('users').document(current_user).get()
    if not buyer_doc.exists:
        raise HTTPException(404, "User not found")

    buyer_data = buyer_doc.to_dict()

    # Check if creator has Creator Pro for revenue split calculation
    creator_has_pro = has_creator_pro_subscription(db, product_data['creator_id'])

    # Calculate revenue split
    revenue_split = calculate_revenue_split(product_data['price'], creator_has_pro)

    # TODO: Process payment with Stripe
    # For now, simulate successful payment

    # Create purchase record using batch
    batch = db.batch()

    purchase_data = {
        'user_id': current_user,
        'user_name': buyer_data.get('name', 'Unknown'),
        'product_id': request.product_id,
        'product_title': product_data['title'],
        'creator_id': product_data['creator_id'],
        'price': product_data['price'],
        'creator_revenue': revenue_split['creator_revenue'],
        'platform_fee': revenue_split['platform_fee'],
        'purchased_at': datetime.utcnow(),
        'download_links': product_data.get('downloadable_files', []),
        'payment_status': 'completed'  # TODO: Update based on Stripe response
    }

    purchase_ref = db.collection('purchases').document()
    batch.set(purchase_ref, purchase_data)

    # Update product stats
    batch.update(product_ref, {
        'total_purchases': firestore.Increment(1),
        'updated_at': datetime.utcnow()
    })

    # Update creator earnings
    creator_ref = db.collection('users').document(product_data['creator_id'])
    batch.update(creator_ref, {
        'creator_stats.total_revenue': firestore.Increment(revenue_split['creator_revenue']),
        'creator_stats.total_sales': firestore.Increment(1)
    })

    # Commit all changes
    batch.commit()

    # Send notifications
    send_realtime_notification(realtime_db, current_user, {
        'type': 'purchase_success',
        'title': 'Purchase Successful!',
        'message': f'You purchased "{product_data["title"]}" for ${product_data["price"]:.2f}',
        'product_id': request.product_id,
        'timestamp': datetime.utcnow().isoformat()
    })

    send_realtime_notification(realtime_db, product_data['creator_id'], {
        'type': 'product_sold',
        'title': 'Product Sold!',
        'message': f'{buyer_data.get("name", "Someone")} purchased "{product_data["title"]}" - You earned ${revenue_split["creator_revenue"]:.2f}',
        'product_id': request.product_id,
        'revenue': revenue_split['creator_revenue'],
        'timestamp': datetime.utcnow().isoformat()
    })

    return PurchaseResponse(
        purchase_id=purchase_ref.id,
        product_id=request.product_id,
        product_title=product_data['title'],
        price=product_data['price'],
        creator_revenue=revenue_split['creator_revenue'],
        platform_fee=revenue_split['platform_fee'],
        purchased_at=purchase_data['purchased_at'].isoformat(),
        download_links=purchase_data['download_links']
    )


@router.get("/purchases", response_model=List[PurchaseResponse])
async def get_my_purchases(
    current_user: str = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=100)
):
    """Get all purchases made by current user."""
    db = get_firestore_client()

    purchases_query = db.collection('purchases').where(
        'user_id', '==', current_user
    ).order_by('purchased_at', direction=firestore.Query.DESCENDING).limit(limit)

    purchases = []
    for purchase_doc in purchases_query.stream():
        purchase_data = purchase_doc.to_dict()

        purchases.append(PurchaseResponse(
            purchase_id=purchase_doc.id,
            **purchase_data,
            purchased_at=purchase_data['purchased_at'].isoformat()
        ))

    return purchases


# ============================================================================
# Bundle Endpoints
# ============================================================================

@router.post("/bundles", response_model=BundleResponse)
async def create_bundle(
    bundle: BundleCreate,
    current_user: str = Depends(get_current_user)
):
    """
    Create product bundle with discount (creators only).

    All products must belong to the same creator.
    """
    db = get_firestore_client()

    # Verify user is a creator
    if not is_creator(db, current_user):
        raise HTTPException(403, "Only creators can create bundles")

    # Verify all products exist and belong to current user
    products = []
    original_price = 0.0

    for product_id in bundle.product_ids:
        product_doc = db.collection('marketplace_products').document(product_id).get()

        if not product_doc.exists:
            raise HTTPException(404, f"Product {product_id} not found")

        product_data = product_doc.to_dict()

        # Verify ownership
        if product_data['creator_id'] != current_user:
            raise HTTPException(403, "All products in bundle must be yours")

        # Verify product is active
        if product_data['status'] != ProductStatus.ACTIVE.value:
            raise HTTPException(400, f"Product {product_id} is not active")

        products.append({
            'id': product_id,
            'title': product_data['title'],
            'price': product_data['price'],
            'product_type': product_data['product_type']
        })

        original_price += product_data['price']

    # Calculate bundle price
    discount_amount = original_price * (bundle.discount_percentage / 100)
    bundle_price = original_price - discount_amount

    # Get creator info
    creator_doc = db.collection('users').document(current_user).get()
    creator_data = creator_doc.to_dict()

    # Create bundle
    bundle_data = {
        'creator_id': current_user,
        'creator_name': creator_data.get('name', 'Unknown'),
        'title': bundle.title,
        'description': bundle.description,
        'product_ids': bundle.product_ids,
        'products': products,
        'original_price': original_price,
        'discount_percentage': bundle.discount_percentage,
        'bundle_price': bundle_price,
        'savings': discount_amount,
        'total_purchases': 0,
        'created_at': datetime.utcnow()
    }

    bundle_ref = db.collection('marketplace_bundles').add(bundle_data)

    return BundleResponse(
        id=bundle_ref[1].id,
        **bundle_data,
        created_at=bundle_data['created_at'].isoformat()
    )


@router.get("/bundles/{bundle_id}", response_model=BundleResponse)
async def get_bundle(
    bundle_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get bundle details."""
    db = get_firestore_client()

    bundle_doc = db.collection('marketplace_bundles').document(bundle_id).get()
    if not bundle_doc.exists:
        raise HTTPException(404, "Bundle not found")

    bundle_data = bundle_doc.to_dict()

    return BundleResponse(
        id=bundle_id,
        **bundle_data,
        created_at=bundle_data['created_at'].isoformat()
    )


# ============================================================================
# Review Endpoints
# ============================================================================

@router.post("/reviews", response_model=ReviewResponse)
async def create_review(
    review: ReviewCreate,
    current_user: str = Depends(get_current_user)
):
    """
    Create product review (requires purchase).

    Users can only review products they've purchased.
    """
    db = get_firestore_client()

    # Verify purchase
    if not has_purchased_product(db, current_user, review.product_id):
        raise HTTPException(403, "You can only review products you've purchased")

    # Check if already reviewed
    existing_review_query = db.collection('product_reviews').where(
        'user_id', '==', current_user
    ).where('product_id', '==', review.product_id).limit(1).stream()

    if list(existing_review_query):
        raise HTTPException(400, "You already reviewed this product")

    # Get user info
    user_doc = db.collection('users').document(current_user).get()
    user_data = user_doc.to_dict()

    # Create review
    review_data = {
        'product_id': review.product_id,
        'user_id': current_user,
        'user_name': user_data.get('name', 'Unknown'),
        'user_avatar': user_data.get('avatar_url'),
        'rating': review.rating,
        'review_text': review.review_text,
        'helpful_features': review.helpful_features,
        'verified_purchase': True,
        'helpful_count': 0,
        'created_at': datetime.utcnow()
    }

    review_ref = db.collection('product_reviews').add(review_data)

    # Update product rating
    product_ref = db.collection('marketplace_products').document(review.product_id)
    product_doc = product_ref.get()

    if product_doc.exists:
        product_data = product_doc.to_dict()
        current_rating = product_data.get('rating', 0.0)
        total_ratings = product_data.get('total_ratings', 0)

        # Calculate new average rating
        new_total_ratings = total_ratings + 1
        new_rating = ((current_rating * total_ratings) + review.rating) / new_total_ratings

        product_ref.update({
            'rating': new_rating,
            'total_ratings': new_total_ratings
        })

    return ReviewResponse(
        id=review_ref[1].id,
        **review_data,
        created_at=review_data['created_at'].isoformat()
    )


@router.get("/products/{product_id}/reviews", response_model=List[ReviewResponse])
async def get_product_reviews(
    product_id: str,
    sort_by: str = Query("created_at", pattern="^(created_at|rating|helpful_count)$"),
    limit: int = Query(50, ge=1, le=100),
    current_user: str = Depends(get_current_user)
):
    """Get all reviews for a product."""
    db = get_firestore_client()

    reviews_query = db.collection('product_reviews').where(
        'product_id', '==', product_id
    ).order_by(sort_by, direction=firestore.Query.DESCENDING).limit(limit)

    reviews = []
    for review_doc in reviews_query.stream():
        review_data = review_doc.to_dict()

        reviews.append(ReviewResponse(
            id=review_doc.id,
            **review_data,
            created_at=review_data['created_at'].isoformat()
        ))

    return reviews


# ============================================================================
# Creator Earnings Endpoints
# ============================================================================

@router.get("/earnings", response_model=CreatorEarningsResponse)
async def get_creator_earnings(
    current_user: str = Depends(get_current_user),
    months: int = Query(12, ge=1, le=24)
):
    """
    Get creator earnings summary and breakdown.

    Shows total revenue, platform fees, and monthly breakdown.
    """
    db = get_firestore_client()

    # Verify user is a creator
    if not is_creator(db, current_user):
        raise HTTPException(403, "Only creators can view earnings")

    # Get user creator stats
    user_doc = db.collection('users').document(current_user).get()
    user_data = user_doc.to_dict()
    creator_stats = user_data.get('creator_stats', {})

    # Check Creator Pro status
    has_creator_pro = has_creator_pro_subscription(db, current_user)
    revenue_share = CREATOR_PRO_REVENUE_SHARE if has_creator_pro else CREATOR_STANDARD_REVENUE_SHARE

    # Get all purchases for this creator
    purchases_query = db.collection('purchases').where(
        'creator_id', '==', current_user
    ).order_by('purchased_at', direction=firestore.Query.DESCENDING).stream()

    # Calculate totals and monthly breakdown
    total_revenue = 0.0
    platform_fees = 0.0
    total_sales = 0
    monthly_data = {}
    product_sales = {}

    for purchase_doc in purchases_query:
        purchase_data = purchase_doc.to_dict()

        total_revenue += purchase_data.get('creator_revenue', 0.0)
        platform_fees += purchase_data.get('platform_fee', 0.0)
        total_sales += 1

        # Monthly breakdown
        month = purchase_data['purchased_at'].strftime('%Y-%m')
        if month not in monthly_data:
            monthly_data[month] = {'revenue': 0.0, 'sales': 0}

        monthly_data[month]['revenue'] += purchase_data.get('creator_revenue', 0.0)
        monthly_data[month]['sales'] += 1

        # Product sales
        product_id = purchase_data['product_id']
        if product_id not in product_sales:
            product_sales[product_id] = {
                'title': purchase_data['product_title'],
                'revenue': 0.0,
                'sales': 0
            }

        product_sales[product_id]['revenue'] += purchase_data.get('creator_revenue', 0.0)
        product_sales[product_id]['sales'] += 1

    # Format monthly breakdown
    monthly_breakdown = [
        {'month': month, 'revenue': data['revenue'], 'sales': data['sales']}
        for month, data in sorted(monthly_data.items(), reverse=True)[:months]
    ]

    # Get top products
    top_products = sorted(
        [{'product_id': pid, **data} for pid, data in product_sales.items()],
        key=lambda x: x['revenue'],
        reverse=True
    )[:10]

    return CreatorEarningsResponse(
        total_revenue=total_revenue,
        platform_fees=platform_fees,
        net_earnings=total_revenue,  # Already deducted in creator_revenue
        total_sales=total_sales,
        revenue_share_percentage=revenue_share * 100,
        has_creator_pro=has_creator_pro,
        monthly_breakdown=monthly_breakdown,
        top_products=top_products
    )
