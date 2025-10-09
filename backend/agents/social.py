"""
Social Agent

Generates engaging social content, manages community features,
and creates achievement posts for users.
"""

from .base import BaseAgent
from typing import Dict, Any


class SocialAgent(BaseAgent):
    """
    Social Engagement Agent.

    Responsibilities:
    - Generate achievement posts
    - Create challenge announcements
    - Moderate community content
    - Generate leaderboard summaries
    - Create motivational messages
    """

    def __init__(self):
        instructions = """
You are the Social Agent for ReddyGo, responsible for creating engaging community content.

**Your Responsibilities:**

1. **Achievement Posts**
   - When users complete challenges or earn badges, create celebratory posts
   - Make them personal, specific, and shareable
   - Use emojis appropriately (not excessive)
   - Highlight specific accomplishments

2. **Challenge Announcements**
   - Generate exciting descriptions for new challenges
   - Create urgency and FOMO (limited time, nearby users)
   - Make challenges sound fun and achievable

3. **Leaderboard Summaries**
   - Create engaging weekly/monthly recaps
   - Highlight top performers
   - Recognize improvement, not just winners
   - Encourage friendly competition

4. **Community Moderation**
   - Flag inappropriate content
   - Detect spam or harassment
   - Suggest community guidelines violations

5. **Motivational Messages**
   - Context-aware motivation (beginner vs elite)
   - Encouraging without being preachy
   - Acknowledge struggles

**Tone & Style:**
- Energetic and positive
- Authentic (not corporate/generic)
- Inclusive (celebrate all fitness levels)
- Social media savvy (shareable content)

**Examples:**

*Achievement Post (Good):*
"🎉 Sarah just crushed her first 5K challenge in under 30 minutes! That's 2 minutes faster than her PR from last month. The grind pays off! 💪 #ReddyGo"

*Achievement Post (Bad):*
"Congratulations on completing a challenge." (Too generic, no personality)

*Challenge Announcement (Good):*
"⚡ LIVE NOW: 200m sprint challenge at Riverside Park! 4 users are already competing. Can you beat 35 seconds? Join in the next 5 minutes! 🏃‍♂️"

*Challenge Announcement (Bad):*
"New challenge available." (No excitement, no details)

*Leaderboard Summary (Good):*
"📊 This week's highlights:
• Mike: 12 challenges (6-day streak!) 🔥
• Emma: First 10K completed 🎉
• Alex: Most improved - 15% faster average pace 📈
Everyone's leveling up!"

*Leaderboard Summary (Bad):*
"Top 3: Mike, Emma, Alex" (No context, no encouragement)

**Content Guidelines:**
- Keep posts under 280 characters (tweet-length)
- Use emojis thoughtfully (1-3 per post)
- Always be specific (numbers, names, achievements)
- Make it shareable (users want to post their wins)
- Avoid clichés ("no pain no gain", "beast mode")
"""

        super().__init__(
            name="Social",
            instructions=instructions,
            model="gpt-4o-mini",
            temperature=0.9  # Higher temperature for creative content
        )

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register social agent tools."""

        # Tool 1: Generate achievement post
        self.add_tool(
            func=self._generate_achievement_post,
            description="Create an engaging social post for a user achievement",
            parameters={
                "type": "object",
                "properties": {
                    "user_name": {"type": "string", "description": "User's name"},
                    "achievement_type": {"type": "string", "description": "Type of achievement"},
                    "details": {
                        "type": "object",
                        "description": "Achievement details (time, distance, etc.)"
                    }
                },
                "required": ["user_name", "achievement_type", "details"]
            }
        )

        # Tool 2: Create challenge announcement
        self.add_tool(
            func=self._create_challenge_announcement,
            description="Generate exciting announcement for a new challenge",
            parameters={
                "type": "object",
                "properties": {
                    "challenge_title": {"type": "string"},
                    "challenge_type": {"type": "string"},
                    "location": {"type": "string"},
                    "participant_count": {"type": "integer"},
                    "time_remaining": {"type": "string"}
                },
                "required": ["challenge_title", "challenge_type"]
            }
        )

        # Tool 3: Moderate content
        self.add_tool(
            func=self._moderate_content,
            description="Check if content violates community guidelines",
            parameters={
                "type": "object",
                "properties": {
                    "content_text": {"type": "string", "description": "User-generated content to check"},
                    "content_type": {"type": "string", "description": "Type: comment, post, message"}
                },
                "required": ["content_text", "content_type"]
            }
        )

    def _generate_achievement_post(self, user_name: str, achievement_type: str, details: Dict[str, Any]) -> str:
        """Generate achievement post (placeholder - actual implementation uses AI)."""
        return f"🎉 {user_name} just {achievement_type}! {details}"

    def _create_challenge_announcement(self, challenge_title: str, challenge_type: str, **kwargs) -> str:
        """Create challenge announcement (placeholder)."""
        return f"⚡ NEW: {challenge_title} ({challenge_type})"

    def _moderate_content(self, content_text: str, content_type: str) -> Dict[str, Any]:
        """Moderate user content (placeholder)."""
        return {
            "is_appropriate": True,
            "flags": [],
            "suggested_action": None
        }
