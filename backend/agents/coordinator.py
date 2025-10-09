"""
Coordinator Agent

Orchestrates challenge creation, participant matching, and initial setup.
Handles challenge lifecycle management.
"""

from .base import BaseAgent
from .tools import (
    query_nearby_users,
    check_blocked_zones,
    create_challenge_instance,
    send_push_notification
)
from typing import Dict, Any


class CoordinatorAgent(BaseAgent):
    """
    Challenge Coordinator Agent.

    Responsibilities:
    - Validate location (not in blocked zone)
    - Find nearby users for challenges
    - Create challenge instance
    - Send notifications to participants
    - Hand off to validation monitoring
    """

    def __init__(self):
        instructions = """
You are the Challenge Coordinator Agent for ReddyGo, a geo-fitness platform.

Your job is to help users create location-based fitness challenges with nearby participants.

When a user wants to start a challenge:

1. **Validate Location Safety**
   - Check if the location is in a blocked zone (schools, hospitals, high-crime areas)
   - If blocked, politely decline and suggest the user move to a safer area
   - Use the check_blocked_zones tool

2. **Find Nearby Users**
   - Query for users within 200m radius who are available
   - Use the query_nearby_users tool
   - If no users found, offer to create a solo challenge

3. **Create Challenge**
   - Generate an engaging challenge title based on the type
   - Set appropriate duration (10-60 minutes depending on challenge)
   - Use the create_challenge_instance tool

4. **Notify Participants**
   - Send push notifications to matched users
   - Use friendly, motivating language
   - Example: "Sarah started a 50-pushup challenge nearby! Join now? 💪"
   - Use the send_push_notification tool

5. **Respond to User**
   - Confirm challenge creation
   - Tell them how many people were notified
   - Give them encouragement

**Personality:**
- Energetic and motivating
- Safety-conscious
- Concise and action-oriented

**Example Response:**
"✅ Challenge created! I found 3 users nearby and sent them invites. Your 50-pushup challenge starts now - let's go! 💪"
"""

        super().__init__(
            name="Coordinator",
            instructions=instructions,
            model="gpt-4o-mini",
            temperature=0.7
        )

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register coordinator tools."""

        # Tool 1: Query nearby users
        self.add_tool(
            func=query_nearby_users,
            description="Find users within a radius of a location for challenge invites",
            parameters={
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "Latitude"},
                    "lon": {"type": "number", "description": "Longitude"},
                    "radius_meters": {"type": "integer", "description": "Search radius in meters", "default": 200}
                },
                "required": ["lat", "lon"]
            }
        )

        # Tool 2: Check blocked zones
        self.add_tool(
            func=check_blocked_zones,
            description="Check if a location is in a restricted safety zone",
            parameters={
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "Latitude"},
                    "lon": {"type": "number", "description": "Longitude"}
                },
                "required": ["lat", "lon"]
            }
        )

        # Tool 3: Create challenge
        self.add_tool(
            func=create_challenge_instance,
            description="Create a new fitness challenge in the database",
            parameters={
                "type": "object",
                "properties": {
                    "creator_id": {"type": "string", "description": "User ID of creator"},
                    "title": {"type": "string", "description": "Challenge title"},
                    "challenge_type": {"type": "string", "enum": ["race", "zone_control", "scavenger_hunt"]},
                    "zone_center": {
                        "type": "object",
                        "properties": {
                            "lat": {"type": "number"},
                            "lon": {"type": "number"}
                        }
                    },
                    "zone_radius_meters": {"type": "integer", "description": "Challenge zone radius"},
                    "rules": {"type": "object", "description": "Challenge rules"}
                },
                "required": ["creator_id", "title", "challenge_type", "zone_center", "zone_radius_meters", "rules"]
            }
        )

        # Tool 4: Send notifications
        self.add_tool(
            func=send_push_notification,
            description="Send push notification to a user",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID"},
                    "title": {"type": "string", "description": "Notification title"},
                    "body": {"type": "string", "description": "Notification body"},
                    "data": {"type": "object", "description": "Extra data"}
                },
                "required": ["user_id", "title", "body"]
            }
        )
