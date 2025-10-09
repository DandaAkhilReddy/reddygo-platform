"""
Safety Agent

Ensures user safety by checking restricted zones, weather conditions,
and time-based restrictions for challenges.
"""

from .base import BaseAgent
from .tools import check_blocked_zones, check_weather
from typing import Dict, Any


class SafetyAgent(BaseAgent):
    """
    Safety Monitoring Agent.

    Responsibilities:
    - Check restricted zones (schools, hospitals, high-crime areas)
    - Validate weather safety
    - Time-based restrictions (no late-night challenges in residential)
    - Emergency alert detection
    - Risk assessment for challenges
    """

    def __init__(self):
        instructions = """
You are the Safety Agent for ReddyGo, responsible for keeping users safe during outdoor fitness challenges.

**Your Priorities (in order):**

1. **USER SAFETY** - Always err on the side of caution
2. **Legal compliance** - Avoid restricted zones
3. **Community respect** - Don't disturb neighborhoods at night
4. **Positive experience** - Guide users, don't just block them

**Restricted Zones (NEVER allow challenges):**
- Schools (within 200m during school hours 7am-4pm)
- Hospitals (within 100m)
- Government buildings (within 100m)
- Private property
- High-crime areas (based on local data)
- Areas with active emergencies

**Weather Safety Rules:**

Automatically BLOCK challenges when:
- Severe thunderstorm warning
- Tornado warning
- Flash flood warning
- Extreme heat (>100°F / 38°C)
- Extreme cold (<0°F / -18°C)
- Air quality index >150 (unhealthy)

WARN but allow when:
- Rain (light to moderate)
- Wind (15-25 mph)
- Hot (90-100°F / 32-38°C)
- Cold (32-45°F / 0-7°C)

**Time-Based Restrictions:**

- Residential areas: No challenges 10pm-7am (noise complaints)
- Parks: Respect posted hours (usually dawn to dusk)
- Urban areas: 24/7 allowed (but weather still applies)

**Your Response Format:**

When BLOCKING a challenge:
```json
{
  "is_safe": false,
  "reason": "Clear explanation",
  "suggestion": "Alternative action",
  "severity": "high"  // high, medium, low
}
```

When ALLOWING with WARNING:
```json
{
  "is_safe": true,
  "warnings": ["List of cautions"],
  "recommendations": ["Safety tips"]
}
```

**Tone:**
- Authoritative but not preachy
- Helpful (offer alternatives)
- Clear and concise
- Empathetic to user disappointment

**Example Responses:**

*Blocking (School Zone):*
"⚠️ Can't start challenges near schools during school hours (7am-4pm) for student safety. The park 3 blocks east is open - try there instead!"

*Blocking (Severe Weather):*
"⛈️ Thunderstorm warning in your area. Stay safe indoors! Weather should clear in 2 hours."

*Warning (Light Rain):*
"🌧️ Light rain detected. Challenge allowed but watch for slippery surfaces. Stay visible to traffic!"

*Emergency Alert:*
"🚨 STOP: Emergency reported in this area. Please move to safety immediately. Challenge cancelled."
"""

        super().__init__(
            name="Safety",
            instructions=instructions,
            model="gpt-4o-mini",
            temperature=0.3  # Lower temperature for consistent safety decisions
        )

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register safety agent tools."""

        # Tool 1: Check blocked zones
        self.add_tool(
            func=check_blocked_zones,
            description="Check if location is in a restricted safety zone",
            parameters={
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "Latitude"},
                    "lon": {"type": "number", "description": "Longitude"}
                },
                "required": ["lat", "lon"]
            }
        )

        # Tool 2: Check weather
        self.add_tool(
            func=check_weather,
            description="Get current weather conditions and safety assessment",
            parameters={
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "Latitude"},
                    "lon": {"type": "number", "description": "Longitude"}
                },
                "required": ["lat", "lon"]
            }
        )

        # Tool 3: Check time restrictions
        self.add_tool(
            func=self._check_time_restrictions,
            description="Validate if current time is appropriate for challenges at this location",
            parameters={
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "zone_type": {"type": "string", "description": "Zone type: residential, park, urban"}
                },
                "required": ["lat", "lon", "zone_type"]
            }
        )

        # Tool 4: Send safety alert
        self.add_tool(
            func=self._send_safety_alert,
            description="Send emergency safety alert to users in an area",
            parameters={
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "radius_meters": {"type": "integer"},
                    "alert_message": {"type": "string"},
                    "severity": {"type": "string", "enum": ["info", "warning", "critical"]}
                },
                "required": ["lat", "lon", "alert_message", "severity"]
            }
        )

    def _check_time_restrictions(self, lat: float, lon: float, zone_type: str) -> Dict[str, Any]:
        """Check time-based restrictions (placeholder)."""
        from datetime import datetime

        current_hour = datetime.now().hour

        # Residential quiet hours: 10pm - 7am
        if zone_type == "residential" and (current_hour >= 22 or current_hour < 7):
            return {
                "is_allowed": False,
                "reason": "Quiet hours in residential area (10pm-7am)",
                "restriction_type": "time_based"
            }

        # Parks: Usually dawn to dusk (6am - 9pm)
        if zone_type == "park" and (current_hour < 6 or current_hour >= 21):
            return {
                "is_allowed": False,
                "reason": "Park closed (hours: 6am-9pm)",
                "restriction_type": "time_based"
            }

        return {
            "is_allowed": True,
            "warnings": []
        }

    def _send_safety_alert(self, lat: float, lon: float, radius_meters: int, alert_message: str, severity: str) -> Dict[str, Any]:
        """Send safety alert to users (placeholder)."""
        # In production: Query users in radius, send push notifications
        print(f"🚨 SAFETY ALERT ({severity}): {alert_message} - Area: ({lat}, {lon}) radius {radius_meters}m")

        return {
            "alert_sent": True,
            "users_notified": 0,  # Placeholder
            "timestamp": "utcnow()"
        }
