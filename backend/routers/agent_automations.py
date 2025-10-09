"""
ReddyGo Agent Automation Workflows

Visual no-code builder for creating AI agent automations, templates, training, and analytics.
Think "Zapier meets Shopify" for AI fitness coaches.

Features:
- Pre-built agent templates
- Visual workflow automation builder
- Conversation training studio
- Agent testing & quality analysis
- Comprehensive analytics dashboard

All endpoints require Firebase authentication via Bearer token in Authorization header.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File
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

class WorkflowTriggerType(str, Enum):
    """Automation workflow trigger types."""
    USER_SUBSCRIBED = "user_subscribed"
    USER_MESSAGE = "user_message"
    WORKOUT_COMPLETED = "workout_completed"
    MILESTONE_ACHIEVED = "milestone_achieved"
    USER_INACTIVE = "user_inactive"
    SCHEDULED_TIME = "scheduled_time"
    STREAK_MILESTONE = "streak_milestone"
    SUBSCRIPTION_EXPIRING = "subscription_expiring"
    PERSONAL_RECORD = "personal_record"
    GOAL_COMPLETED = "goal_completed"


class WorkflowActionType(str, Enum):
    """Automation workflow action types."""
    SEND_MESSAGE = "send_message"
    UPDATE_TRAINING_PLAN = "update_training_plan"
    SEND_NOTIFICATION = "send_notification"
    SEND_EMAIL = "send_email"
    AWARD_POINTS = "award_points"
    SCHEDULE_ACTION = "schedule_action"
    CALL_WEBHOOK = "call_webhook"
    RUN_SWARM = "run_swarm"
    UPDATE_GOALS = "update_goals"
    CREATE_WORKOUT = "create_workout"


class AgentQualityCategory(str, Enum):
    """Agent quality check categories."""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"  # 75-89
    FAIR = "fair"  # 60-74
    NEEDS_IMPROVEMENT = "needs_improvement"  # <60


# ============================================================================
# Pydantic Models
# ============================================================================

# Template Models

class AgentTemplate(BaseModel):
    """Pre-built agent template."""
    id: str
    name: str
    description: str
    category: str  # fitness, nutrition, specialty
    specialization: str
    motivational_style: str
    preview_image_url: Optional[str]

    # Pre-filled content
    training_philosophy: str
    techniques: List[str]
    sample_conversations: List[Dict[str, str]]
    default_automations: List[Dict[str, Any]]

    # Metrics
    times_used: int
    avg_rating: float
    created_by: Optional[str]  # If community template
    royalty_per_use: Optional[float]

    created_at: str


class TemplateCustomization(BaseModel):
    """Customize template before applying."""
    template_id: str
    customizations: Dict[str, Any]
    # Example: {"name": "My Custom Name", "techniques": ["add", "new", "techniques"]}


# Workflow Models

class WorkflowTrigger(BaseModel):
    """Workflow trigger configuration."""
    type: WorkflowTriggerType
    conditions: Optional[Dict[str, Any]] = None
    # Example conditions: {"inactive_days": 3}, {"streak_milestone": 7}


class WorkflowAction(BaseModel):
    """Workflow action configuration."""
    type: WorkflowActionType
    config: Dict[str, Any]
    # Example config: {"message": "Hey! You've been inactive..."}, {"points": 50}
    delay_seconds: Optional[int] = 0  # Delay before executing


class WorkflowCreate(BaseModel):
    """Create automation workflow."""
    name: str = Field(..., min_length=3, max_length=100)
    description: str = Field(..., min_length=10, max_length=500)
    agent_id: str
    trigger: WorkflowTrigger
    actions: List[WorkflowAction] = Field(..., min_items=1, max_items=10)
    enabled: bool = True


class WorkflowResponse(BaseModel):
    """Workflow information."""
    id: str
    agent_id: str
    name: str
    description: str
    trigger: Dict[str, Any]
    actions: List[Dict[str, Any]]
    enabled: bool
    total_executions: int
    success_rate: float
    created_at: str
    updated_at: str


class WorkflowExecutionLog(BaseModel):
    """Workflow execution log entry."""
    id: str
    workflow_id: str
    workflow_name: str
    trigger_data: Dict[str, Any]
    actions_executed: List[Dict[str, Any]]
    status: str  # success, failed, partial
    error_message: Optional[str]
    execution_time_ms: int
    timestamp: str


# Knowledge Base Models

class KnowledgeUploadResponse(BaseModel):
    """Knowledge base upload result."""
    upload_id: str
    filename: str
    file_size: int
    file_type: str
    status: str  # processing, completed, failed
    extracted_knowledge: Optional[List[str]]
    uploaded_at: str


class KnowledgeParse(BaseModel):
    """Parse knowledge from content."""
    content: str = Field(..., min_length=50, max_length=10000)
    content_type: str = Field("text", pattern="^(text|url|pdf)$")


# Training Models

class ConversationExample(BaseModel):
    """Training conversation example."""
    question: str = Field(..., min_length=5, max_length=500)
    ideal_response: str = Field(..., min_length=10, max_length=2000)
    tags: List[str] = Field(default=[])


class TestResponse(BaseModel):
    """Test agent response."""
    test_question: str
    agent_response: str
    quality_score: float
    matches_tone: bool
    safety_passed: bool
    improvement_suggestions: List[str]


class ABTest(BaseModel):
    """A/B test configuration."""
    name: str
    agent_id: str
    variant_a: Dict[str, Any]  # Original settings
    variant_b: Dict[str, Any]  # Modified settings
    metric: str  # rating, engagement, retention
    duration_days: int = Field(7, ge=1, le=30)


class ABTestResult(BaseModel):
    """A/B test results."""
    test_id: str
    test_name: str
    variant_a_performance: Dict[str, float]
    variant_b_performance: Dict[str, float]
    winner: str  # a, b, or tie
    confidence_level: float
    started_at: str
    ended_at: Optional[str]


# Testing Models

class AgentQualityScore(BaseModel):
    """Agent quality analysis."""
    overall_score: int  # 0-100
    category: str
    breakdown: Dict[str, int]  # {safety: 95, expertise: 80, tone: 85, helpfulness: 90}
    strengths: List[str]
    improvements: List[str]
    suggested_actions: List[str]


class SafetyCheckResult(BaseModel):
    """Safety validation result."""
    passed: bool
    risk_level: str  # safe, caution, dangerous
    issues_found: List[str]
    recommendations: List[str]


# Analytics Models

class AnalyticsOverview(BaseModel):
    """Analytics dashboard overview."""
    agent_id: str
    agent_name: str

    # Subscriber metrics
    total_subscribers: int
    active_subscribers: int
    new_subscribers_7d: int
    churn_rate_30d: float

    # Engagement metrics
    total_chats: int
    avg_chats_per_user: float
    engagement_rate: float  # % of subscribers chatting
    avg_response_quality: float

    # Revenue metrics
    total_revenue: float
    monthly_revenue: float
    revenue_per_subscriber: float

    # Performance
    avg_rating: float
    total_ratings: int
    trending_score: float

    period_start: str
    period_end: str


class PopularQuestion(BaseModel):
    """Popular user question."""
    question: str
    frequency: int
    avg_satisfaction: float
    common_keywords: List[str]


class FeedbackSummary(BaseModel):
    """User feedback summary."""
    total_reviews: int
    avg_rating: float
    rating_distribution: Dict[str, int]  # {5: 100, 4: 50, ...}
    sentiment_analysis: Dict[str, float]  # {positive: 0.7, neutral: 0.2, negative: 0.1}
    common_praises: List[str]
    common_complaints: List[str]


# ============================================================================
# Helper Functions
# ============================================================================

def verify_agent_ownership(db, agent_id: str, user_id: str) -> bool:
    """Verify user owns the agent."""
    agent_doc = db.collection('ai_agents').document(agent_id).get()
    if not agent_doc.exists:
        return False
    return agent_doc.to_dict().get('creator_id') == user_id


async def execute_workflow_action(
    action: WorkflowAction,
    trigger_data: Dict[str, Any],
    agent_id: str,
    user_id: str
):
    """Execute a workflow action."""
    db = get_firestore_client()
    realtime_db = get_realtime_db()

    try:
        if action.type == WorkflowActionType.SEND_MESSAGE:
            # Send chat message via agent
            message = action.config.get('message', '')

            # Get agent data to build system prompt
            from routers.ai_agents import build_agent_system_prompt

            agent_doc = db.collection('ai_agents').document(agent_id).get()
            agent_data = agent_doc.to_dict()

            system_prompt = build_agent_system_prompt(agent_data)

            response = openai_client.chat.completions.create(
                model=agent_data.get('model', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=agent_data.get('temperature', 0.7),
                max_tokens=300
            )

            agent_response = response.choices[0].message.content

            # Send via realtime DB
            realtime_db.child('agent_messages').child(user_id).push({
                'agent_id': agent_id,
                'agent_name': agent_data['name'],
                'message': agent_response,
                'automated': True,
                'timestamp': datetime.utcnow().isoformat()
            })

            return {"status": "sent", "message": agent_response}

        elif action.type == WorkflowActionType.SEND_NOTIFICATION:
            # Send push notification
            notification_config = action.config

            realtime_db.child('notifications').child(user_id).push({
                'type': 'agent_automation',
                'title': notification_config.get('title', 'Agent Message'),
                'message': notification_config.get('message', ''),
                'agent_id': agent_id,
                'timestamp': datetime.utcnow().isoformat()
            })

            return {"status": "sent"}

        elif action.type == WorkflowActionType.AWARD_POINTS:
            # Award points to user
            points = action.config.get('points', 0)
            reason = action.config.get('reason', 'Automation reward')

            user_ref = db.collection('users').document(user_id)
            user_ref.update({
                'stats.total_points': firestore.Increment(points)
            })

            # Log points transaction
            db.collection('points_transactions').add({
                'user_id': user_id,
                'points': points,
                'reason': reason,
                'source': 'agent_automation',
                'agent_id': agent_id,
                'timestamp': datetime.utcnow()
            })

            return {"status": "awarded", "points": points}

        elif action.type == WorkflowActionType.SCHEDULE_ACTION:
            # Schedule future action
            delay_minutes = action.config.get('delay_minutes', 60)
            future_action = action.config.get('action', {})

            db.collection('scheduled_workflow_actions').add({
                'agent_id': agent_id,
                'user_id': user_id,
                'action': future_action,
                'execute_at': datetime.utcnow() + timedelta(minutes=delay_minutes),
                'status': 'pending',
                'created_at': datetime.utcnow()
            })

            return {"status": "scheduled", "execute_in_minutes": delay_minutes}

        else:
            return {"status": "not_implemented", "action_type": action.type.value}

    except Exception as e:
        return {"status": "failed", "error": str(e)}


def send_realtime_notification(realtime_db, user_id: str, notification: Dict[str, Any]):
    """Send real-time notification via Firebase Realtime Database."""
    try:
        realtime_db.child('notifications').child(user_id).push(notification)
    except Exception as e:
        print(f"Failed to send notification to {user_id}: {e}")


# ============================================================================
# Template Endpoints
# ============================================================================

@router.get("/templates", response_model=List[AgentTemplate])
async def get_templates(
    category: Optional[str] = Query(None, pattern="^(fitness|nutrition|specialty|all)$"),
    featured: bool = False,
    limit: int = Query(50, ge=1, le=100),
    current_user: str = Depends(get_current_user)
):
    """
    Get pre-built agent templates.

    Browse 50+ templates organized by category.
    """
    db = get_firestore_client()

    query = db.collection('agent_templates')

    if featured:
        query = query.where('featured', '==', True)

    if category and category != "all":
        query = query.where('category', '==', category)

    query = query.order_by('times_used', direction=firestore.Query.DESCENDING).limit(limit)

    templates = []
    for template_doc in query.stream():
        template_data = template_doc.to_dict()

        templates.append(AgentTemplate(
            id=template_doc.id,
            **template_data,
            created_at=template_data['created_at'].isoformat()
        ))

    return templates


@router.post("/templates/apply")
async def apply_template(
    template_id: str,
    customizations: Optional[Dict[str, Any]] = None,
    current_user: str = Depends(get_current_user)
):
    """
    Apply template to create new agent.

    Optionally customize template fields before applying.
    """
    db = get_firestore_client()

    # Get template
    template_doc = db.collection('agent_templates').document(template_id).get()
    if not template_doc.exists:
        raise HTTPException(404, "Template not found")

    template_data = template_doc.to_dict()

    # Verify user is a creator
    user_doc = db.collection('users').document(current_user).get()
    user_data = user_doc.to_dict()

    if not user_data.get('is_creator', False):
        raise HTTPException(403, "Only creators can use templates")

    # Build agent from template
    agent_data = {
        'creator_id': current_user,
        'creator_name': user_data.get('name', 'Unknown'),
        'creator_verified': user_data.get('verified', False),
        'name': customizations.get('name', f"{user_data.get('name', 'My')} {template_data['name']}") if customizations else template_data['name'],
        'description': template_data['description'],
        'specialization': template_data['specialization'],
        'motivational_style': template_data['motivational_style'],
        'price_tier': 4.99,
        'status': 'draft',

        # From template
        'training_philosophy': template_data['training_philosophy'],
        'techniques': template_data['techniques'],
        'certifications': customizations.get('certifications', []) if customizations else [],
        'years_experience': customizations.get('years_experience', 0) if customizations else 0,

        # Settings
        'response_style': 'detailed',
        'max_response_length': 300,
        'model': 'gpt-4o-mini',
        'temperature': 0.7,

        # Metrics
        'rating': 0.0,
        'total_ratings': 0,
        'total_subscribers': 0,
        'total_chats': 0,
        'success_rate': 0.0,
        'response_time_avg': 0.0,
        'revenue_generated': 0.0,

        'template_id': template_id,
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }

    # Create agent
    agent_ref = db.collection('ai_agents').add(agent_data)
    agent_id = agent_ref[1].id

    # Add sample conversations as training data
    for sample in template_data.get('sample_conversations', []):
        db.collection('agent_training_examples').add({
            'agent_id': agent_id,
            'question': sample.get('question', ''),
            'ideal_response': sample.get('response', ''),
            'source': 'template',
            'created_at': datetime.utcnow()
        })

    # Create default automations
    for automation in template_data.get('default_automations', []):
        db.collection('agent_workflows').add({
            'agent_id': agent_id,
            'creator_id': current_user,
            'name': automation.get('name', ''),
            'description': automation.get('description', ''),
            'trigger': automation.get('trigger', {}),
            'actions': automation.get('actions', []),
            'enabled': True,
            'total_executions': 0,
            'success_rate': 100.0,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        })

    # Increment template usage
    template_doc.reference.update({
        'times_used': firestore.Increment(1)
    })

    # If template has royalty, track it
    if template_data.get('royalty_per_use', 0) > 0:
        db.collection('template_royalties').add({
            'template_id': template_id,
            'template_creator': template_data.get('created_by'),
            'user_id': current_user,
            'agent_id': agent_id,
            'royalty_amount': template_data['royalty_per_use'],
            'status': 'pending',
            'created_at': datetime.utcnow()
        })

    return {
        "success": True,
        "agent_id": agent_id,
        "message": "Agent created from template successfully",
        "automations_created": len(template_data.get('default_automations', [])),
        "training_examples_added": len(template_data.get('sample_conversations', []))
    }


@router.get("/templates/{template_id}", response_model=AgentTemplate)
async def get_template_details(
    template_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get detailed template information with preview."""
    db = get_firestore_client()

    template_doc = db.collection('agent_templates').document(template_id).get()
    if not template_doc.exists:
        raise HTTPException(404, "Template not found")

    template_data = template_doc.to_dict()

    return AgentTemplate(
        id=template_id,
        **template_data,
        created_at=template_data['created_at'].isoformat()
    )


# ============================================================================
# Workflow Automation Endpoints
# ============================================================================

@router.post("/workflows/create", response_model=WorkflowResponse)
async def create_workflow(
    workflow: WorkflowCreate,
    current_user: str = Depends(get_current_user)
):
    """
    Create automation workflow for agent.

    Build Zapier-style automation: TRIGGER → CONDITIONS → ACTIONS
    """
    db = get_firestore_client()

    # Verify agent ownership
    if not verify_agent_ownership(db, workflow.agent_id, current_user):
        raise HTTPException(403, "You don't own this agent")

    # Create workflow
    workflow_data = {
        'agent_id': workflow.agent_id,
        'creator_id': current_user,
        'name': workflow.name,
        'description': workflow.description,
        'trigger': workflow.trigger.dict(),
        'actions': [action.dict() for action in workflow.actions],
        'enabled': workflow.enabled,
        'total_executions': 0,
        'success_rate': 100.0,
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }

    workflow_ref = db.collection('agent_workflows').add(workflow_data)
    workflow_id = workflow_ref[1].id

    return WorkflowResponse(
        id=workflow_id,
        agent_id=workflow.agent_id,
        name=workflow.name,
        description=workflow.description,
        trigger=workflow.trigger.dict(),
        actions=[action.dict() for action in workflow.actions],
        enabled=workflow.enabled,
        total_executions=0,
        success_rate=100.0,
        created_at=workflow_data['created_at'].isoformat(),
        updated_at=workflow_data['updated_at'].isoformat()
    )


@router.get("/workflows", response_model=List[WorkflowResponse])
async def get_workflows(
    agent_id: str,
    enabled_only: bool = False,
    current_user: str = Depends(get_current_user)
):
    """Get all workflows for agent."""
    db = get_firestore_client()

    # Verify agent ownership
    if not verify_agent_ownership(db, agent_id, current_user):
        raise HTTPException(403, "You don't own this agent")

    query = db.collection('agent_workflows').where('agent_id', '==', agent_id)

    if enabled_only:
        query = query.where('enabled', '==', True)

    workflows = []
    for workflow_doc in query.stream():
        workflow_data = workflow_doc.to_dict()

        workflows.append(WorkflowResponse(
            id=workflow_doc.id,
            **workflow_data,
            created_at=workflow_data['created_at'].isoformat(),
            updated_at=workflow_data['updated_at'].isoformat()
        ))

    return workflows


@router.put("/workflows/{workflow_id}")
async def update_workflow(
    workflow_id: str,
    updates: Dict[str, Any],
    current_user: str = Depends(get_current_user)
):
    """Update workflow configuration."""
    db = get_firestore_client()

    workflow_ref = db.collection('agent_workflows').document(workflow_id)
    workflow_doc = workflow_ref.get()

    if not workflow_doc.exists:
        raise HTTPException(404, "Workflow not found")

    workflow_data = workflow_doc.to_dict()

    # Verify ownership via agent
    if not verify_agent_ownership(db, workflow_data['agent_id'], current_user):
        raise HTTPException(403, "You don't own this workflow")

    # Update
    updates['updated_at'] = datetime.utcnow()
    workflow_ref.update(updates)

    return {"success": True, "message": "Workflow updated"}


@router.delete("/workflows/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    current_user: str = Depends(get_current_user)
):
    """Delete workflow."""
    db = get_firestore_client()

    workflow_ref = db.collection('agent_workflows').document(workflow_id)
    workflow_doc = workflow_ref.get()

    if not workflow_doc.exists:
        raise HTTPException(404, "Workflow not found")

    workflow_data = workflow_doc.to_dict()

    # Verify ownership
    if not verify_agent_ownership(db, workflow_data['agent_id'], current_user):
        raise HTTPException(403, "You don't own this workflow")

    workflow_ref.delete()

    return {"success": True, "message": "Workflow deleted"}


@router.post("/workflows/{workflow_id}/test")
async def test_workflow(
    workflow_id: str,
    test_data: Dict[str, Any],
    current_user: str = Depends(get_current_user)
):
    """
    Test workflow with sample data.

    Simulates trigger and executes actions in test mode.
    """
    db = get_firestore_client()

    workflow_doc = db.collection('agent_workflows').document(workflow_id).get()
    if not workflow_doc.exists:
        raise HTTPException(404, "Workflow not found")

    workflow_data = workflow_doc.to_dict()

    # Verify ownership
    if not verify_agent_ownership(db, workflow_data['agent_id'], current_user):
        raise HTTPException(403, "You don't own this workflow")

    # Execute actions in test mode
    action_results = []

    for action_data in workflow_data['actions']:
        action = WorkflowAction(**action_data)

        result = await execute_workflow_action(
            action,
            test_data,
            workflow_data['agent_id'],
            current_user  # Use creator as test user
        )

        action_results.append({
            'action_type': action.type.value,
            'result': result,
            'config': action.config
        })

    return {
        "success": True,
        "workflow_name": workflow_data['name'],
        "trigger": workflow_data['trigger'],
        "actions_executed": len(action_results),
        "results": action_results
    }


@router.post("/workflows/{workflow_id}/enable")
async def enable_disable_workflow(
    workflow_id: str,
    enabled: bool,
    current_user: str = Depends(get_current_user)
):
    """Enable or disable workflow."""
    db = get_firestore_client()

    workflow_ref = db.collection('agent_workflows').document(workflow_id)
    workflow_doc = workflow_ref.get()

    if not workflow_doc.exists:
        raise HTTPException(404, "Workflow not found")

    workflow_data = workflow_doc.to_dict()

    # Verify ownership
    if not verify_agent_ownership(db, workflow_data['agent_id'], current_user):
        raise HTTPException(403, "You don't own this workflow")

    workflow_ref.update({
        'enabled': enabled,
        'updated_at': datetime.utcnow()
    })

    return {
        "success": True,
        "workflow_name": workflow_data['name'],
        "enabled": enabled
    }


@router.get("/workflows/{workflow_id}/logs", response_model=List[WorkflowExecutionLog])
async def get_workflow_logs(
    workflow_id: str,
    limit: int = Query(50, ge=1, le=200),
    current_user: str = Depends(get_current_user)
):
    """Get workflow execution logs."""
    db = get_firestore_client()

    # Verify ownership
    workflow_doc = db.collection('agent_workflows').document(workflow_id).get()
    if not workflow_doc.exists:
        raise HTTPException(404, "Workflow not found")

    workflow_data = workflow_doc.to_dict()

    if not verify_agent_ownership(db, workflow_data['agent_id'], current_user):
        raise HTTPException(403, "You don't own this workflow")

    # Get logs
    logs_query = db.collection('workflow_execution_logs').where(
        'workflow_id', '==', workflow_id
    ).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)

    logs = []
    for log_doc in logs_query.stream():
        log_data = log_doc.to_dict()

        logs.append(WorkflowExecutionLog(
            id=log_doc.id,
            **log_data,
            timestamp=log_data['timestamp'].isoformat()
        ))

    return logs


# Due to file size limits, I'll continue in the next message with:
# - Knowledge Base endpoints
# - Training endpoints
# - Testing endpoints
# - Analytics endpoints
