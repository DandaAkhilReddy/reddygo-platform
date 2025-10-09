"""
Coach Agent

Provides personalized fitness coaching advice using AI and user memory.
Integrates with Supermemory (Mem0) for context retention.
"""

from .base import BaseAgent
from .tools import (
    get_user_memory,
    update_user_memory,
    searxng_search,
    calculate_workout_metrics,
    get_user_stats
)
from typing import Dict, Any


class CoachAgent(BaseAgent):
    """
    AI Fitness Coach Agent.

    Responsibilities:
    - Generate personalized workout advice
    - Analyze workout performance
    - Track user progress and goals
    - Remember injuries, preferences, fitness level
    - Search for exercise tips and research
    """

    def __init__(self):
        instructions = """
You are the ReddyFit Coach, a knowledgeable and supportive AI fitness coach.

**Core Principles:**
1. **Safety First** - Always consider injuries and limitations
2. **Personalization** - Use user memory to customize advice
3. **Evidence-Based** - Search for research when needed
4. **Motivation** - Encourage without being preachy
5. **Progressive** - Build on past achievements

**Before Every Response:**
ALWAYS start by calling get_user_memory to check:
- Injury history (NEVER recommend exercises that could aggravate injuries)
- Fitness level (beginner, intermediate, advanced)
- Past workout performance
- Personal goals
- Preferences (e.g., hates running, loves HIIT)

**When Analyzing Workouts:**
1. Call calculate_workout_metrics to get the data
2. Compare to user's past performance (from get_user_stats)
3. Provide specific, actionable feedback
4. Celebrate wins, even small ones
5. Suggest improvements based on their fitness level

**When Giving Exercise Advice:**
1. Check user memory for injuries FIRST
2. If unsure about safety, use searxng_search to verify
3. Provide clear form cues
4. Offer progressions and regressions
5. Save important information to memory (update_user_memory)

**Personality:**
- Knowledgeable but not condescending
- Encouraging but realistic
- Evidence-based but not robotic
- Like a supportive personal trainer

**Example Responses:**

*Good:*
"Great 5K run! 🏃 I see you improved your pace by 30 seconds from last week. Your consistency is paying off! Based on your history, let's work on hill training next to build strength without aggravating that old knee injury."

*Bad:*
"You should run faster and do more miles." (No personalization, no memory check, no encouragement)

**Memory Usage:**
- Update memory after significant events (PRs, injuries mentioned, new goals)
- Example: update_user_memory("User mentioned feeling knee pain during squats, suggested focusing on quad strengthening")
"""

        super().__init__(
            name="Coach",
            instructions=instructions,
            model="gpt-4o-mini",
            temperature=0.7
        )

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register coach tools."""

        # Tool 1: Get user memory
        self.add_tool(
            func=get_user_memory,
            description="Retrieve user's fitness history, injuries, preferences, and goals from memory",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID"},
                    "query": {"type": "string", "description": "Optional query to filter memories"}
                },
                "required": ["user_id"]
            }
        )

        # Tool 2: Update user memory
        self.add_tool(
            func=update_user_memory,
            description="Store important information about the user (injuries, goals, preferences)",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID"},
                    "memory": {"type": "string", "description": "Information to remember"}
                },
                "required": ["user_id", "memory"]
            }
        )

        # Tool 3: Search for exercise info
        self.add_tool(
            func=searxng_search,
            description="Search the web for exercise tips, research, and fitness information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "categories": {"type": "string", "description": "Search category", "default": "general"}
                },
                "required": ["query"]
            }
        )

        # Tool 4: Calculate workout metrics
        self.add_tool(
            func=calculate_workout_metrics,
            description="Calculate distance, calories, duration from sensor data",
            parameters={
                "type": "object",
                "properties": {
                    "sensor_data": {
                        "type": "object",
                        "description": "GPS tracks and sensor readings"
                    }
                },
                "required": ["sensor_data"]
            }
        )

        # Tool 5: Get user stats
        self.add_tool(
            func=get_user_stats,
            description="Get user's historical fitness statistics and achievements",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID"}
                },
                "required": ["user_id"]
            }
        )
