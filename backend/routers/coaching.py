"""
ReddyGo Coaching Router

AI coaching endpoints powered by Coach Agent.
Provides personalized fitness advice with memory integration.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from agents.coach import CoachAgent

router = APIRouter()

# Initialize Coach Agent
coach_agent = CoachAgent()


# Request/Response Models
class CoachingRequest(BaseModel):
    """Request for AI coaching advice."""
    user_id: str
    query: str
    context: Optional[Dict[str, Any]] = None


class WorkoutAnalysisRequest(BaseModel):
    """Request for workout analysis."""
    user_id: str
    workout_data: Dict[str, Any]
    sensor_data: Optional[Dict[str, Any]] = None


class CoachingResponse(BaseModel):
    """AI coach response."""
    advice: str
    memories_used: list
    cost_estimate: float
    agent_name: str


@router.post("/ask", response_model=CoachingResponse)
async def ask_coach(request: CoachingRequest):
    """
    Ask the AI coach a question.

    Get personalized fitness advice based on user's history and goals.

    Example queries:
    - "What's a good warm-up for sprinting?"
    - "How can I improve my 5K time?"
    - "Is it safe to run with knee pain?"
    """
    try:
        # Run coach agent
        result = await coach_agent.run(
            user_message=request.query,
            context={"user_id": request.user_id, **(request.context or {})}
        )

        if result.get("status") == "failed":
            raise HTTPException(status_code=500, detail=result.get("error"))

        # Extract response
        advice = result.get("content", "")
        usage = result.get("usage", {})

        # Calculate cost
        cost = coach_agent.get_cost_estimate(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0)
        )

        return CoachingResponse(
            advice=advice,
            memories_used=[],  # TODO: Extract from tool calls
            cost_estimate=cost,
            agent_name="Coach"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coaching request failed: {str(e)}")


@router.post("/analyze-workout")
async def analyze_workout(request: WorkoutAnalysisRequest):
    """
    Analyze completed workout and provide feedback.

    Returns:
    - Performance analysis
    - Comparison to past workouts
    - Improvement suggestions
    - Achievement recognition
    """
    try:
        # Construct analysis prompt
        prompt = f"""
Analyze this workout for user {request.user_id}:

Workout Data:
{request.workout_data}

Sensor Data:
{request.sensor_data or 'Not provided'}

Please provide:
1. Performance analysis (how did they do?)
2. Comparison to their past workouts
3. Specific improvements to suggest
4. Recognition of achievements or milestones
"""

        result = await coach_agent.run(
            user_message=prompt,
            context={"user_id": request.user_id, "type": "workout_analysis"}
        )

        if result.get("status") == "failed":
            raise HTTPException(status_code=500, detail=result.get("error"))

        return {
            "analysis": result.get("content"),
            "achievements": [],  # TODO: Extract from response
            "improvements": [],
            "cost": coach_agent.get_cost_estimate(
                result.get("usage", {}).get("prompt_tokens", 0),
                result.get("usage", {}).get("completion_tokens", 0)
            )
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workout analysis failed: {str(e)}")


@router.get("/plan/{user_id}")
async def get_workout_plan(user_id: str, duration_days: int = 7):
    """
    Generate personalized workout plan.

    Args:
        user_id: User ID
        duration_days: Plan duration (default: 7 days)

    Returns:
        Weekly workout plan based on user's fitness level and goals
    """
    try:
        prompt = f"""
Create a {duration_days}-day workout plan for user {user_id}.

Requirements:
1. Check their fitness level and injury history
2. Consider their past workout patterns
3. Progressive difficulty (build on previous days)
4. Mix of challenge types (sprints, endurance, strength)
5. Rest days included
6. Specific, actionable workouts

Format as a day-by-day plan.
"""

        result = await coach_agent.run(
            user_message=prompt,
            context={"user_id": user_id, "type": "workout_plan"}
        )

        if result.get("status") == "failed":
            raise HTTPException(status_code=500, detail=result.get("error"))

        return {
            "plan": result.get("content"),
            "duration_days": duration_days,
            "user_id": user_id,
            "generated_at": "utcnow()",  # Would be actual datetime
            "cost": coach_agent.get_cost_estimate(
                result.get("usage", {}).get("prompt_tokens", 0),
                result.get("usage", {}).get("completion_tokens", 0)
            )
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan generation failed: {str(e)}")


@router.post("/feedback")
async def submit_feedback(user_id: str, feedback_text: str):
    """
    Submit feedback to update user memory.

    Example:
    - "I felt knee pain during squats today"
    - "I want to focus on building endurance"
    - "I prefer outdoor workouts"

    This will be stored in user memory for future coaching sessions.
    """
    try:
        # Update memory prompt
        prompt = f"""
User feedback: "{feedback_text}"

Please update the user's memory with this information.
Use the update_user_memory tool to store important details.
"""

        result = await coach_agent.run(
            user_message=prompt,
            context={"user_id": user_id}
        )

        return {
            "memory_updated": True,
            "feedback": feedback_text,
            "user_id": user_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")
