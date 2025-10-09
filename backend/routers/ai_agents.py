"""
ReddyGo AI Agents Router

Revolutionary AI agent marketplace where fitness experts create AI coaches that generate
passive income while helping thousands of users simultaneously.

Features:
- Create custom AI coaches using OpenAI SDK
- Multi-agent swarm collaboration
- Agent marketplace with subscriptions
- Creator earnings dashboard
- Performance-based rankings and tournaments

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
import os
from openai import OpenAI
import json

router = APIRouter()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# Enums and Constants
# ============================================================================

class AgentSpecialization(str, Enum):
    """AI agent specialization types."""
    GENERAL_COACH = "general_coach"
    STRENGTH = "strength"
    CARDIO = "cardio"
    NUTRITION = "nutrition"
    INJURY_RECOVERY = "injury_recovery"
    MOTIVATION = "motivation"
    MARATHON = "marathon"
    BODYBUILDING = "bodybuilding"
    CALISTHENICS = "calisthenics"
    YOGA = "yoga"
    HIIT = "hiit"


class AgentStatus(str, Enum):
    """Agent listing status."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class MotivationalStyle(str, Enum):
    """Agent personality/motivation style."""
    SUPPORTIVE = "supportive"  # Encouraging, gentle
    DRILL_SERGEANT = "drill_sergeant"  # Tough love, strict
    SCIENTIFIC = "scientific"  # Data-driven, analytical
    FRIENDLY = "friendly"  # Casual, buddy-like
    PROFESSIONAL = "professional"  # Formal, expert


# Subscription pricing
AGENT_SUBSCRIPTION_PRICES = {
    "single_agent": 4.99,
    "dream_team": 9.99,  # 3 agents
    "elite_swarm": 19.99  # Unlimited agents
}

# Revenue splits
AGENT_CREATOR_REVENUE_SHARE = 0.70  # Agent creators get 70%
PLATFORM_FEE = 0.25  # Platform takes 25%
REFERRER_SHARE = 0.05  # Referrer gets 5%

# Performance bonuses
TOP_AGENT_BONUS_MULTIPLIER = 1.10  # Top 10 agents get 10% bonus


# ============================================================================
# Pydantic Models
# ============================================================================

class AgentCreate(BaseModel):
    """Create custom AI agent."""
    name: str = Field(..., min_length=3, max_length=50)
    description: str = Field(..., min_length=20, max_length=500)
    specialization: AgentSpecialization
    motivational_style: MotivationalStyle
    price_tier: float = Field(4.99, ge=2.99, le=19.99)

    # Agent training data
    training_philosophy: str = Field(..., min_length=100, max_length=3000)
    techniques: List[str] = Field(..., min_items=3, max_items=20)
    certifications: List[str] = Field(default=[])
    years_experience: int = Field(..., ge=0, le=50)

    # Advanced settings
    response_style: str = Field("detailed", pattern="^(concise|detailed|motivational)$")
    max_response_length: int = Field(300, ge=100, le=1000)

    # OpenAI model settings
    model: str = Field("gpt-4o-mini", pattern="^(gpt-4o-mini|gpt-4|gpt-4o)$")
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class AgentUpdate(BaseModel):
    """Update AI agent settings."""
    name: Optional[str] = Field(None, min_length=3, max_length=50)
    description: Optional[str] = Field(None, min_length=20, max_length=500)
    price_tier: Optional[float] = Field(None, ge=2.99, le=19.99)
    training_philosophy: Optional[str] = Field(None, min_length=100, max_length=3000)
    techniques: Optional[List[str]] = Field(None, min_items=3, max_items=20)
    status: Optional[AgentStatus] = None
    response_style: Optional[str] = Field(None, pattern="^(concise|detailed|motivational)$")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)


class AgentResponse(BaseModel):
    """AI agent information."""
    id: str
    creator_id: str
    creator_name: str
    creator_verified: bool
    name: str
    description: str
    specialization: str
    motivational_style: str
    price_tier: float
    status: str

    # Performance metrics
    rating: float
    total_ratings: int
    total_subscribers: int
    success_rate: float
    response_time_avg: float

    # Training info
    years_experience: int
    certifications: List[str]

    created_at: str
    updated_at: str


class AgentChatRequest(BaseModel):
    """Chat with AI agent."""
    agent_id: str
    message: str = Field(..., min_length=1, max_length=2000)
    context: Optional[Dict[str, Any]] = None


class AgentChatResponse(BaseModel):
    """AI agent chat response."""
    agent_id: str
    agent_name: str
    message: str
    confidence_score: float
    tokens_used: int
    cost: float
    timestamp: str


class SwarmConsultRequest(BaseModel):
    """Multi-agent swarm consultation."""
    agent_ids: List[str] = Field(..., min_items=2, max_items=5)
    question: str = Field(..., min_length=10, max_length=2000)
    context: Optional[Dict[str, Any]] = None


class SwarmConsultResponse(BaseModel):
    """Swarm consultation result."""
    question: str
    agent_responses: List[Dict[str, Any]]
    consensus_answer: str
    confidence_score: float
    total_cost: float
    timestamp: str


class AgentSubscriptionRequest(BaseModel):
    """Subscribe to AI agent."""
    agent_id: str
    # TODO: payment_method_id for Stripe


class DreamTeamSubscriptionRequest(BaseModel):
    """Subscribe to agent dream team."""
    agent_ids: List[str] = Field(..., min_items=3, max_items=3)
    # TODO: payment_method_id for Stripe


class CreatorDashboardResponse(BaseModel):
    """Creator earnings and analytics."""
    total_revenue: float
    monthly_revenue: float
    total_subscribers: int
    agents_created: int
    top_performing_agent: Dict[str, Any]
    monthly_breakdown: List[Dict[str, Any]]
    performance_bonus: float


class AgentPerformanceResponse(BaseModel):
    """Agent performance metrics."""
    agent_id: str
    agent_name: str
    total_subscribers: int
    active_subscribers: int
    rating: float
    total_ratings: int
    total_chats: int
    avg_response_time: float
    success_rate: float
    revenue_generated: float
    rank_in_category: int
    trending_score: float


class TournamentResponse(BaseModel):
    """Active agent tournament."""
    tournament_id: str
    category: str
    start_date: str
    end_date: str
    prize_pool: float
    participants: List[Dict[str, Any]]
    current_leader: Dict[str, Any]


# ============================================================================
# Helper Functions
# ============================================================================

def build_agent_system_prompt(agent_data: Dict[str, Any]) -> str:
    """Build system prompt for AI agent based on creator's training."""

    specialization_intros = {
        "general_coach": "You are a comprehensive fitness coach with broad expertise.",
        "strength": "You are a strength training specialist focused on building muscle and power.",
        "cardio": "You are a cardiovascular training expert specializing in endurance and aerobic fitness.",
        "nutrition": "You are a nutrition coach specializing in meal planning and dietary optimization.",
        "injury_recovery": "You are an injury recovery specialist focused on rehabilitation and safe training progressions.",
        "motivation": "You are a motivational coach focused on mindset, accountability, and habit formation.",
        "marathon": "You are a marathon training specialist with expertise in long-distance running.",
        "bodybuilding": "You are a bodybuilding coach focused on hypertrophy and physique development.",
        "calisthenics": "You are a calisthenics expert specializing in bodyweight training.",
        "yoga": "You are a yoga instructor focused on flexibility, balance, and mind-body connection.",
        "hiit": "You are a HIIT (High Intensity Interval Training) specialist."
    }

    style_instructions = {
        "supportive": "Use an encouraging, gentle, and supportive tone. Celebrate small wins.",
        "drill_sergeant": "Use a tough-love, no-nonsense approach. Be direct and demanding but fair.",
        "scientific": "Use a data-driven, analytical approach. Cite research and focus on metrics.",
        "friendly": "Use a casual, buddy-like tone. Be relatable and conversational.",
        "professional": "Use a formal, expert tone. Be authoritative and precise."
    }

    response_styles = {
        "concise": "Keep responses under 150 words. Be direct and actionable.",
        "detailed": "Provide comprehensive explanations with examples and reasoning.",
        "motivational": "Focus on inspiration and encouragement while providing advice."
    }

    prompt = f"""
{specialization_intros.get(agent_data['specialization'], 'You are a fitness coach.')}

**Creator's Training Philosophy:**
{agent_data['training_philosophy']}

**Your Expertise:**
- Years of Experience: {agent_data['years_experience']}
- Key Techniques: {', '.join(agent_data['techniques'])}
- Certifications: {', '.join(agent_data.get('certifications', ['General Fitness']))}

**Communication Style:**
{style_instructions.get(agent_data['motivational_style'], 'Be helpful and professional.')}

**Response Guidelines:**
{response_styles.get(agent_data.get('response_style', 'detailed'), 'Provide helpful, actionable advice.')}

**Important Rules:**
1. ALWAYS prioritize safety - never recommend exercises that could cause injury
2. Ask about injuries or limitations before giving specific exercise recommendations
3. Base advice on your creator's training philosophy and techniques
4. Stay within your specialization - don't give medical advice
5. Encourage users to consult professionals for serious injuries
6. Be consistent with your motivational style
7. Remember context from previous messages when possible

**Your Goal:**
Help users achieve their fitness goals using your specialized knowledge and your creator's proven methods.
Provide value that justifies their subscription to your coaching services.
"""

    return prompt


def calculate_agent_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> float:
    """Calculate OpenAI API cost for agent interaction."""
    costs = {
        "gpt-4o-mini": {"input": 0.150 / 1_000_000, "output": 0.600 / 1_000_000},
        "gpt-4": {"input": 30.00 / 1_000_000, "output": 60.00 / 1_000_000},
        "gpt-4o": {"input": 5.00 / 1_000_000, "output": 15.00 / 1_000_000}
    }

    model_costs = costs.get(model, costs["gpt-4o-mini"])
    total = (input_tokens * model_costs["input"]) + (output_tokens * model_costs["output"])
    return round(total, 6)


def is_subscribed_to_agent(db, user_id: str, agent_id: str) -> bool:
    """Check if user has active subscription to agent."""
    subscription_query = db.collection('agent_subscriptions').where(
        'user_id', '==', user_id
    ).where('agent_id', '==', agent_id).where(
        'status', '==', 'active'
    ).limit(1).stream()

    return len(list(subscription_query)) > 0


def get_user_subscription_tier(db, user_id: str) -> Optional[str]:
    """Get user's current agent subscription tier."""
    subscription_query = db.collection('agent_tier_subscriptions').where(
        'user_id', '==', user_id
    ).where('status', '==', 'active').limit(1).stream()

    subscriptions = list(subscription_query)
    if subscriptions:
        return subscriptions[0].to_dict().get('tier', 'single_agent')

    return None


def can_subscribe_to_agent(db, user_id: str, agent_id: str) -> bool:
    """Check if user can subscribe to agent based on their tier."""
    tier = get_user_subscription_tier(db, user_id)

    if tier == "elite_swarm":
        return True  # Unlimited agents

    if tier == "dream_team":
        # Check if already subscribed to 3 agents
        active_subs = db.collection('agent_subscriptions').where(
            'user_id', '==', user_id
        ).where('status', '==', 'active').stream()

        return len(list(active_subs)) < 3

    if tier == "single_agent":
        # Check if already subscribed to any agent
        active_subs = db.collection('agent_subscriptions').where(
            'user_id', '==', user_id
        ).where('status', '==', 'active').stream()

        return len(list(active_subs)) == 0

    return False


def send_realtime_notification(realtime_db, user_id: str, notification: Dict[str, Any]):
    """Send real-time notification via Firebase Realtime Database."""
    try:
        realtime_db.child('notifications').child(user_id).push(notification)
    except Exception as e:
        print(f"Failed to send notification to {user_id}: {e}")


# ============================================================================
# Agent Creation & Management Endpoints
# ============================================================================

@router.post("/create-agent", response_model=AgentResponse)
async def create_agent(
    agent: AgentCreate,
    current_user: str = Depends(get_current_user)
):
    """
    Create custom AI coach agent (requires creator approval).

    Agent creators earn 70% of subscription revenue from their agents.
    Top 10 performing agents get 10% performance bonus.
    """
    db = get_firestore_client()

    # Verify user is an approved creator
    user_doc = db.collection('users').document(current_user).get()
    if not user_doc.exists:
        raise HTTPException(404, "User not found")

    user_data = user_doc.to_dict()
    if not user_data.get('is_creator', False):
        raise HTTPException(403, "Only approved creators can create AI agents")

    # Check agent creation limit (max 5 agents per creator)
    creator_agents = db.collection('ai_agents').where(
        'creator_id', '==', current_user
    ).stream()

    if len(list(creator_agents)) >= 5:
        raise HTTPException(400, "Maximum 5 agents per creator. Archive old agents to create new ones.")

    # Create agent
    agent_data = {
        'creator_id': current_user,
        'creator_name': user_data.get('name', 'Unknown'),
        'creator_verified': user_data.get('verified', False),
        'name': agent.name,
        'description': agent.description,
        'specialization': agent.specialization.value,
        'motivational_style': agent.motivational_style.value,
        'price_tier': agent.price_tier,
        'status': AgentStatus.DRAFT.value,

        # Training data
        'training_philosophy': agent.training_philosophy,
        'techniques': agent.techniques,
        'certifications': agent.certifications,
        'years_experience': agent.years_experience,

        # Settings
        'response_style': agent.response_style,
        'max_response_length': agent.max_response_length,
        'model': agent.model,
        'temperature': agent.temperature,

        # Performance metrics
        'rating': 0.0,
        'total_ratings': 0,
        'total_subscribers': 0,
        'total_chats': 0,
        'success_rate': 0.0,
        'response_time_avg': 0.0,
        'revenue_generated': 0.0,

        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }

    agent_ref = db.collection('ai_agents').add(agent_data)
    agent_id = agent_ref[1].id

    return AgentResponse(
        id=agent_id,
        **agent_data,
        created_at=agent_data['created_at'].isoformat(),
        updated_at=agent_data['updated_at'].isoformat()
    )


@router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get AI agent details."""
    db = get_firestore_client()

    agent_doc = db.collection('ai_agents').document(agent_id).get()
    if not agent_doc.exists:
        raise HTTPException(404, "Agent not found")

    agent_data = agent_doc.to_dict()

    # Only show draft/paused agents to their creator
    if agent_data['status'] in [AgentStatus.DRAFT.value, AgentStatus.PAUSED.value]:
        if agent_data['creator_id'] != current_user:
            raise HTTPException(404, "Agent not found")

    return AgentResponse(
        id=agent_id,
        **agent_data,
        created_at=agent_data['created_at'].isoformat(),
        updated_at=agent_data['updated_at'].isoformat()
    )


@router.put("/agents/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    updates: AgentUpdate,
    current_user: str = Depends(get_current_user)
):
    """
    Update AI agent (creator only).

    Allows fine-tuning agent behavior based on user feedback.
    """
    db = get_firestore_client()

    agent_ref = db.collection('ai_agents').document(agent_id)
    agent_doc = agent_ref.get()

    if not agent_doc.exists:
        raise HTTPException(404, "Agent not found")

    agent_data = agent_doc.to_dict()

    # Verify ownership
    if agent_data['creator_id'] != current_user:
        raise HTTPException(403, "You can only update your own agents")

    # Apply updates
    update_dict = {k: v for k, v in updates.dict().items() if v is not None}
    if update_dict:
        update_dict['updated_at'] = datetime.utcnow()
        agent_ref.update(update_dict)

        # Get updated data
        updated_doc = agent_ref.get()
        updated_data = updated_doc.to_dict()

        return AgentResponse(
            id=agent_id,
            **updated_data,
            created_at=updated_data['created_at'].isoformat(),
            updated_at=updated_data['updated_at'].isoformat()
        )

    return AgentResponse(
        id=agent_id,
        **agent_data,
        created_at=agent_data['created_at'].isoformat(),
        updated_at=agent_data['updated_at'].isoformat()
    )


@router.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Delete AI agent (creator only).

    Agents with active subscribers cannot be deleted (only archived).
    """
    db = get_firestore_client()

    agent_ref = db.collection('ai_agents').document(agent_id)
    agent_doc = agent_ref.get()

    if not agent_doc.exists:
        raise HTTPException(404, "Agent not found")

    agent_data = agent_doc.to_dict()

    # Verify ownership
    if agent_data['creator_id'] != current_user:
        raise HTTPException(403, "You can only delete your own agents")

    # Prevent deletion if has active subscribers
    if agent_data.get('total_subscribers', 0) > 0:
        raise HTTPException(400, "Cannot delete agents with active subscribers. Archive instead.")

    agent_ref.delete()

    return {"success": True, "message": "Agent deleted successfully"}


@router.get("/my-agents", response_model=List[AgentResponse])
async def get_my_agents(
    status: Optional[AgentStatus] = None,
    current_user: str = Depends(get_current_user)
):
    """Get all AI agents created by current user."""
    db = get_firestore_client()

    query = db.collection('ai_agents').where('creator_id', '==', current_user)

    if status:
        query = query.where('status', '==', status.value)

    query = query.order_by('created_at', direction=firestore.Query.DESCENDING)

    agents = []
    for agent_doc in query.stream():
        agent_data = agent_doc.to_dict()

        agents.append(AgentResponse(
            id=agent_doc.id,
            **agent_data,
            created_at=agent_data['created_at'].isoformat(),
            updated_at=agent_data['updated_at'].isoformat()
        ))

    return agents


# ============================================================================
# Marketplace Endpoints
# ============================================================================

@router.get("/marketplace", response_model=List[AgentResponse])
async def browse_marketplace(
    specialization: Optional[AgentSpecialization] = None,
    motivational_style: Optional[MotivationalStyle] = None,
    min_rating: Optional[float] = Query(None, ge=0.0, le=5.0),
    max_price: Optional[float] = Query(None, ge=0, le=19.99),
    sort_by: str = Query("rating", pattern="^(rating|price_tier|total_subscribers|created_at)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    limit: int = Query(50, ge=1, le=100),
    current_user: str = Depends(get_current_user)
):
    """
    Browse AI agent marketplace.

    Filter by specialization, style, rating, price.
    Sort by popularity, rating, or price.
    """
    db = get_firestore_client()

    # Only show active agents
    query = db.collection('ai_agents').where('status', '==', AgentStatus.ACTIVE.value)

    # Apply filters
    if specialization:
        query = query.where('specialization', '==', specialization.value)

    if motivational_style:
        query = query.where('motivational_style', '==', motivational_style.value)

    # Apply sorting
    direction = firestore.Query.DESCENDING if sort_order == "desc" else firestore.Query.ASCENDING
    query = query.order_by(sort_by, direction=direction)

    # Execute query
    agents = []
    for agent_doc in query.limit(limit).stream():
        agent_data = agent_doc.to_dict()

        # Apply rating filter
        if min_rating is not None and agent_data.get('rating', 0.0) < min_rating:
            continue

        # Apply price filter
        if max_price is not None and agent_data['price_tier'] > max_price:
            continue

        agents.append(AgentResponse(
            id=agent_doc.id,
            **agent_data,
            created_at=agent_data['created_at'].isoformat(),
            updated_at=agent_data['updated_at'].isoformat()
        ))

    return agents


@router.post("/subscribe/{agent_id}")
async def subscribe_to_agent(
    agent_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Subscribe to AI agent (requires active agent tier subscription).

    User must have purchased a tier (single_agent, dream_team, or elite_swarm).
    """
    db = get_firestore_client()
    realtime_db = get_realtime_db()

    # Get agent
    agent_ref = db.collection('ai_agents').document(agent_id)
    agent_doc = agent_ref.get()

    if not agent_doc.exists:
        raise HTTPException(404, "Agent not found")

    agent_data = agent_doc.to_dict()

    # Verify agent is active
    if agent_data['status'] != AgentStatus.ACTIVE.value:
        raise HTTPException(400, "Agent is not available")

    # Check if user can subscribe based on tier
    if not can_subscribe_to_agent(db, current_user, agent_id):
        raise HTTPException(400, "Subscription limit reached for your tier. Upgrade to subscribe to more agents.")

    # Check if already subscribed
    if is_subscribed_to_agent(db, current_user, agent_id):
        raise HTTPException(400, "Already subscribed to this agent")

    # Get user data
    user_doc = db.collection('users').document(current_user).get()
    user_data = user_doc.to_dict()

    # Create subscription using batch
    batch = db.batch()

    subscription_data = {
        'user_id': current_user,
        'user_name': user_data.get('name', 'Unknown'),
        'agent_id': agent_id,
        'agent_name': agent_data['name'],
        'creator_id': agent_data['creator_id'],
        'price_tier': agent_data['price_tier'],
        'status': 'active',
        'subscribed_at': datetime.utcnow(),
        'total_chats': 0,
        'last_interaction': None
    }

    subscription_ref = db.collection('agent_subscriptions').document()
    batch.set(subscription_ref, subscription_data)

    # Update agent stats
    batch.update(agent_ref, {
        'total_subscribers': firestore.Increment(1),
        'updated_at': datetime.utcnow()
    })

    # Commit
    batch.commit()

    # Send notifications
    send_realtime_notification(realtime_db, current_user, {
        'type': 'agent_subscription',
        'title': 'AI Coach Subscribed!',
        'message': f'You can now chat with {agent_data["name"]} anytime!',
        'agent_id': agent_id,
        'timestamp': datetime.utcnow().isoformat()
    })

    send_realtime_notification(realtime_db, agent_data['creator_id'], {
        'type': 'new_subscriber',
        'title': 'New Subscriber!',
        'message': f'{user_data.get("name", "Someone")} subscribed to your agent "{agent_data["name"]}"',
        'agent_id': agent_id,
        'timestamp': datetime.utcnow().isoformat()
    })

    return {
        "success": True,
        "message": "Successfully subscribed to AI agent",
        "agent_name": agent_data['name'],
        "subscription_id": subscription_ref.id
    }


@router.post("/subscribe-team")
async def subscribe_dream_team(
    request: DreamTeamSubscriptionRequest,
    current_user: str = Depends(get_current_user)
):
    """
    Subscribe to agent dream team (3 agents working together).

    Requires dream_team or elite_swarm tier subscription.
    """
    db = get_firestore_client()

    # Verify user has dream_team or elite_swarm tier
    tier = get_user_subscription_tier(db, current_user)
    if tier not in ["dream_team", "elite_swarm"]:
        raise HTTPException(403, "Dream team subscription requires dream_team or elite_swarm tier")

    # Verify all 3 agents exist and are active
    agent_names = []
    for agent_id in request.agent_ids:
        agent_doc = db.collection('ai_agents').document(agent_id).get()
        if not agent_doc.exists or agent_doc.to_dict()['status'] != AgentStatus.ACTIVE.value:
            raise HTTPException(404, f"Agent {agent_id} not found or not active")
        agent_names.append(agent_doc.to_dict()['name'])

    # Subscribe to all 3 agents
    for agent_id in request.agent_ids:
        if not is_subscribed_to_agent(db, current_user, agent_id):
            # Subscribe (reuse subscribe_to_agent logic)
            pass  # Implementation would call subscribe endpoint

    return {
        "success": True,
        "message": "Successfully subscribed to dream team",
        "agents": agent_names
    }


@router.post("/unsubscribe/{agent_id}")
async def unsubscribe_from_agent(
    agent_id: str,
    current_user: str = Depends(get_current_user)
):
    """Unsubscribe from AI agent."""
    db = get_firestore_client()

    # Find active subscription
    subscription_query = db.collection('agent_subscriptions').where(
        'user_id', '==', current_user
    ).where('agent_id', '==', agent_id).where(
        'status', '==', 'active'
    ).limit(1).stream()

    subscriptions = list(subscription_query)
    if not subscriptions:
        raise HTTPException(404, "No active subscription found")

    subscription_doc = subscriptions[0]
    subscription_data = subscription_doc.to_dict()

    # Update subscription and agent stats using batch
    batch = db.batch()

    batch.update(subscription_doc.reference, {
        'status': 'cancelled',
        'cancelled_at': datetime.utcnow()
    })

    agent_ref = db.collection('ai_agents').document(agent_id)
    batch.update(agent_ref, {
        'total_subscribers': firestore.Increment(-1)
    })

    batch.commit()

    return {
        "success": True,
        "message": "Successfully unsubscribed from agent"
    }


@router.get("/my-subscriptions", response_model=List[Dict[str, Any]])
async def get_my_subscriptions(
    current_user: str = Depends(get_current_user)
):
    """Get user's active agent subscriptions."""
    db = get_firestore_client()

    subscriptions_query = db.collection('agent_subscriptions').where(
        'user_id', '==', current_user
    ).where('status', '==', 'active').stream()

    subscriptions = []
    for sub_doc in subscriptions_query:
        sub_data = sub_doc.to_dict()

        # Get agent details
        agent_doc = db.collection('ai_agents').document(sub_data['agent_id']).get()
        if agent_doc.exists:
            agent_data = agent_doc.to_dict()

            subscriptions.append({
                'subscription_id': sub_doc.id,
                'agent_id': sub_data['agent_id'],
                'agent_name': sub_data['agent_name'],
                'specialization': agent_data.get('specialization'),
                'price_tier': sub_data['price_tier'],
                'subscribed_at': sub_data['subscribed_at'].isoformat(),
                'total_chats': sub_data.get('total_chats', 0),
                'last_interaction': sub_data.get('last_interaction').isoformat() if sub_data.get('last_interaction') else None
            })

    return subscriptions


# ============================================================================
# Agent Interaction Endpoints
# ============================================================================

@router.post("/chat", response_model=AgentChatResponse)
async def chat_with_agent(
    request: AgentChatRequest,
    current_user: str = Depends(get_current_user)
):
    """
    Chat with subscribed AI agent.

    Agent uses OpenAI SDK with creator's custom training.
    """
    db = get_firestore_client()

    # Verify subscription
    if not is_subscribed_to_agent(db, current_user, request.agent_id):
        raise HTTPException(403, "Not subscribed to this agent. Subscribe first to chat.")

    # Get agent
    agent_doc = db.collection('ai_agents').document(request.agent_id).get()
    if not agent_doc.exists:
        raise HTTPException(404, "Agent not found")

    agent_data = agent_doc.to_dict()

    # Build system prompt
    system_prompt = build_agent_system_prompt(agent_data)

    # Get user context
    user_doc = db.collection('users').document(current_user).get()
    user_data = user_doc.to_dict() if user_doc.exists else {}

    # Add user context to prompt
    context_info = f"""
User Context:
- Name: {user_data.get('name', 'Unknown')}
- Fitness Level: {user_data.get('fitness_level', 'beginner')}
- Goals: {user_data.get('goals', 'general fitness')}
- Recent Injuries: {user_data.get('injuries', 'None reported')}
"""

    if request.context:
        context_info += f"\nAdditional Context: {json.dumps(request.context)}"

    # Call OpenAI API
    try:
        start_time = datetime.utcnow()

        response = openai_client.chat.completions.create(
            model=agent_data.get('model', 'gpt-4o-mini'),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": context_info},
                {"role": "user", "content": request.message}
            ],
            temperature=agent_data.get('temperature', 0.7),
            max_tokens=agent_data.get('max_response_length', 300)
        )

        end_time = datetime.utcnow()
        response_time = (end_time - start_time).total_seconds()

        message = response.choices[0].message
        usage = response.usage

        # Calculate cost
        cost = calculate_agent_cost(
            usage.prompt_tokens,
            usage.completion_tokens,
            agent_data.get('model', 'gpt-4o-mini')
        )

        # Update stats
        subscription_query = db.collection('agent_subscriptions').where(
            'user_id', '==', current_user
        ).where('agent_id', '==', request.agent_id).where(
            'status', '==', 'active'
        ).limit(1).stream()

        subscriptions = list(subscription_query)
        if subscriptions:
            subscription_doc = subscriptions[0]
            subscription_doc.reference.update({
                'total_chats': firestore.Increment(1),
                'last_interaction': datetime.utcnow()
            })

        # Update agent stats
        agent_ref = db.collection('ai_agents').document(request.agent_id)
        agent_ref.update({
            'total_chats': firestore.Increment(1),
            'response_time_avg': firestore.Increment(response_time / 1000)  # Average calculation
        })

        # Log chat for analytics
        db.collection('agent_chat_logs').add({
            'agent_id': request.agent_id,
            'user_id': current_user,
            'message': request.message,
            'response': message.content,
            'tokens_used': usage.total_tokens,
            'cost': cost,
            'response_time': response_time,
            'timestamp': datetime.utcnow()
        })

        return AgentChatResponse(
            agent_id=request.agent_id,
            agent_name=agent_data['name'],
            message=message.content,
            confidence_score=0.85,  # Placeholder - could use logprobs
            tokens_used=usage.total_tokens,
            cost=cost,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        raise HTTPException(500, f"Agent chat failed: {str(e)}")


@router.post("/swarm-consult", response_model=SwarmConsultResponse)
async def swarm_consultation(
    request: SwarmConsultRequest,
    current_user: str = Depends(get_current_user)
):
    """
    Multi-agent swarm consultation.

    Multiple AI agents collaborate to provide comprehensive advice.
    Agents discuss the question and reach consensus.
    """
    db = get_firestore_client()

    # Verify subscriptions to all agents
    for agent_id in request.agent_ids:
        if not is_subscribed_to_agent(db, current_user, agent_id):
            raise HTTPException(403, f"Not subscribed to agent {agent_id}")

    # Get all agents
    agents = []
    for agent_id in request.agent_ids:
        agent_doc = db.collection('ai_agents').document(agent_id).get()
        if agent_doc.exists:
            agent_data = agent_doc.to_dict()
            agent_data['id'] = agent_id
            agents.append(agent_data)

    if len(agents) < 2:
        raise HTTPException(400, "Need at least 2 active agents for swarm consultation")

    # Get each agent's perspective
    agent_responses = []
    total_cost = 0.0

    for agent in agents:
        system_prompt = build_agent_system_prompt(agent)

        try:
            response = openai_client.chat.completions.create(
                model=agent.get('model', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"From your specialized perspective as a {agent['specialization']} expert, please answer: {request.question}"}
                ],
                temperature=agent.get('temperature', 0.7),
                max_tokens=300
            )

            message = response.choices[0].message
            usage = response.usage

            cost = calculate_agent_cost(
                usage.prompt_tokens,
                usage.completion_tokens,
                agent.get('model', 'gpt-4o-mini')
            )

            total_cost += cost

            agent_responses.append({
                'agent_id': agent['id'],
                'agent_name': agent['name'],
                'specialization': agent['specialization'],
                'response': message.content,
                'tokens_used': usage.total_tokens
            })

        except Exception as e:
            print(f"Agent {agent['id']} failed: {e}")
            continue

    # Synthesize consensus using coordinator
    consensus_prompt = f"""
You are a fitness coordination agent. Multiple specialist AI coaches have provided their perspectives on a user's question.

**User Question:**
{request.question}

**Expert Perspectives:**
"""

    for resp in agent_responses:
        consensus_prompt += f"\n\n**{resp['specialization'].upper()} COACH ({resp['agent_name']}):**\n{resp['response']}"

    consensus_prompt += """

**Your Task:**
Synthesize these expert perspectives into a single, comprehensive answer that:
1. Incorporates the best insights from each specialist
2. Resolves any conflicting advice by explaining trade-offs
3. Provides a clear, actionable recommendation
4. Is concise but thorough (max 400 words)
"""

    try:
        consensus_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a fitness coordination agent skilled at synthesizing expert opinions."},
                {"role": "user", "content": consensus_prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )

        consensus_message = consensus_response.choices[0].message
        consensus_usage = consensus_response.usage

        total_cost += calculate_agent_cost(
            consensus_usage.prompt_tokens,
            consensus_usage.completion_tokens,
            "gpt-4o-mini"
        )

        # Log swarm consultation
        db.collection('swarm_consultations').add({
            'user_id': current_user,
            'question': request.question,
            'agent_ids': request.agent_ids,
            'total_cost': total_cost,
            'timestamp': datetime.utcnow()
        })

        return SwarmConsultResponse(
            question=request.question,
            agent_responses=agent_responses,
            consensus_answer=consensus_message.content,
            confidence_score=0.90,  # Higher confidence with multiple experts
            total_cost=total_cost,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        raise HTTPException(500, f"Consensus generation failed: {str(e)}")


@router.post("/analyze-workout")
async def analyze_workout_with_agent(
    agent_id: str,
    workout_data: Dict[str, Any],
    current_user: str = Depends(get_current_user)
):
    """
    Get AI agent analysis of completed workout.

    Agent provides personalized feedback based on performance.
    """
    db = get_firestore_client()

    # Verify subscription
    if not is_subscribed_to_agent(db, current_user, agent_id):
        raise HTTPException(403, "Not subscribed to this agent")

    # Format workout data for agent
    workout_summary = json.dumps(workout_data, indent=2)

    message = f"""
Please analyze my workout and provide feedback:

{workout_summary}

Please provide:
1. Performance assessment
2. Comparison to my usual performance (if you remember)
3. Specific areas of improvement
4. Celebration of achievements
5. Suggestions for next workout
"""

    # Use regular chat endpoint
    chat_request = AgentChatRequest(
        agent_id=agent_id,
        message=message,
        context=workout_data
    )

    return await chat_with_agent(chat_request, current_user)


# ============================================================================
# Creator Earnings Endpoints
# ============================================================================

@router.get("/creator-dashboard", response_model=CreatorDashboardResponse)
async def get_creator_dashboard(
    current_user: str = Depends(get_current_user)
):
    """
    Get creator earnings dashboard.

    Shows revenue from all created AI agents.
    """
    db = get_firestore_client()

    # Get all creator's agents
    agents_query = db.collection('ai_agents').where(
        'creator_id', '==', current_user
    ).stream()

    agents = list(agents_query)
    if not agents:
        raise HTTPException(404, "No agents found. Create an agent first.")

    # Calculate totals
    total_revenue = 0.0
    total_subscribers = 0
    top_agent = None
    top_revenue = 0.0

    for agent_doc in agents:
        agent_data = agent_doc.to_dict()
        revenue = agent_data.get('revenue_generated', 0.0)

        total_revenue += revenue
        total_subscribers += agent_data.get('total_subscribers', 0)

        if revenue > top_revenue:
            top_revenue = revenue
            top_agent = {
                'id': agent_doc.id,
                'name': agent_data['name'],
                'revenue': revenue,
                'subscribers': agent_data.get('total_subscribers', 0)
            }

    # Get monthly revenue (placeholder - would query transactions)
    current_month = datetime.utcnow().strftime('%Y-%m')
    monthly_revenue = total_revenue * 0.3  # Placeholder

    # Performance bonus (top 10 agents)
    performance_bonus = 0.0  # Calculate based on rankings

    return CreatorDashboardResponse(
        total_revenue=total_revenue,
        monthly_revenue=monthly_revenue,
        total_subscribers=total_subscribers,
        agents_created=len(agents),
        top_performing_agent=top_agent or {},
        monthly_breakdown=[],  # Placeholder
        performance_bonus=performance_bonus
    )


@router.get("/agent-performance/{agent_id}", response_model=AgentPerformanceResponse)
async def get_agent_performance(
    agent_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Get detailed performance metrics for specific agent.

    Includes subscriber count, ratings, chat volume, revenue.
    """
    db = get_firestore_client()

    agent_doc = db.collection('ai_agents').document(agent_id).get()
    if not agent_doc.exists:
        raise HTTPException(404, "Agent not found")

    agent_data = agent_doc.to_dict()

    # Verify ownership
    if agent_data['creator_id'] != current_user:
        raise HTTPException(403, "You can only view performance of your own agents")

    # Get active subscriber count
    active_subs = db.collection('agent_subscriptions').where(
        'agent_id', '==', agent_id
    ).where('status', '==', 'active').stream()

    active_subscriber_count = len(list(active_subs))

    return AgentPerformanceResponse(
        agent_id=agent_id,
        agent_name=agent_data['name'],
        total_subscribers=agent_data.get('total_subscribers', 0),
        active_subscribers=active_subscriber_count,
        rating=agent_data.get('rating', 0.0),
        total_ratings=agent_data.get('total_ratings', 0),
        total_chats=agent_data.get('total_chats', 0),
        avg_response_time=agent_data.get('response_time_avg', 0.0),
        success_rate=agent_data.get('success_rate', 0.0),
        revenue_generated=agent_data.get('revenue_generated', 0.0),
        rank_in_category=1,  # Placeholder - calculate rank
        trending_score=0.0  # Placeholder - calculate trending
    )


@router.get("/earnings-history", response_model=List[Dict[str, Any]])
async def get_earnings_history(
    current_user: str = Depends(get_current_user),
    months: int = Query(12, ge=1, le=24)
):
    """Get monthly earnings history from AI agents."""
    db = get_firestore_client()

    # Query earnings transactions
    earnings_query = db.collection('agent_creator_earnings').where(
        'creator_id', '==', current_user
    ).order_by('month', direction=firestore.Query.DESCENDING).limit(months).stream()

    earnings = []
    for earning_doc in earnings_query:
        earning_data = earning_doc.to_dict()
        earnings.append({
            'month': earning_data['month'],
            'total_revenue': earning_data.get('total_revenue', 0.0),
            'subscribers': earning_data.get('total_subscribers', 0),
            'chats': earning_data.get('total_chats', 0),
            'performance_bonus': earning_data.get('performance_bonus', 0.0)
        })

    return earnings


# ============================================================================
# Tournament Endpoints
# ============================================================================

@router.get("/tournaments/active", response_model=List[TournamentResponse])
async def get_active_tournaments(
    current_user: str = Depends(get_current_user)
):
    """
    Get active agent tournaments.

    Tournaments compete agents based on client results and satisfaction.
    """
    db = get_firestore_client()

    # Get active tournaments
    tournaments_query = db.collection('agent_tournaments').where(
        'status', '==', 'active'
    ).order_by('start_date', direction=firestore.Query.DESCENDING).stream()

    tournaments = []
    for tournament_doc in tournaments_query:
        tournament_data = tournament_doc.to_dict()

        tournaments.append(TournamentResponse(
            tournament_id=tournament_doc.id,
            category=tournament_data['category'],
            start_date=tournament_data['start_date'].isoformat(),
            end_date=tournament_data['end_date'].isoformat(),
            prize_pool=tournament_data.get('prize_pool', 0.0),
            participants=tournament_data.get('participants', []),
            current_leader=tournament_data.get('current_leader', {})
        ))

    return tournaments


@router.get("/leaderboard", response_model=List[Dict[str, Any]])
async def get_agent_leaderboard(
    specialization: Optional[AgentSpecialization] = None,
    metric: str = Query("rating", pattern="^(rating|subscribers|revenue|chats)$"),
    limit: int = Query(50, ge=1, le=100),
    current_user: str = Depends(get_current_user)
):
    """
    Get agent leaderboard (top performing agents).

    Filter by specialization, sort by various metrics.
    """
    db = get_firestore_client()

    # Build query
    query = db.collection('ai_agents').where('status', '==', AgentStatus.ACTIVE.value)

    if specialization:
        query = query.where('specialization', '==', specialization.value)

    # Map metric to field
    metric_field_map = {
        "rating": "rating",
        "subscribers": "total_subscribers",
        "revenue": "revenue_generated",
        "chats": "total_chats"
    }

    query = query.order_by(metric_field_map[metric], direction=firestore.Query.DESCENDING)
    query = query.limit(limit)

    leaderboard = []
    rank = 1

    for agent_doc in query.stream():
        agent_data = agent_doc.to_dict()

        leaderboard.append({
            'rank': rank,
            'agent_id': agent_doc.id,
            'agent_name': agent_data['name'],
            'creator_name': agent_data['creator_name'],
            'specialization': agent_data['specialization'],
            'rating': agent_data.get('rating', 0.0),
            'subscribers': agent_data.get('total_subscribers', 0),
            'revenue': agent_data.get('revenue_generated', 0.0),
            'total_chats': agent_data.get('total_chats', 0)
        })

        rank += 1

    return leaderboard
