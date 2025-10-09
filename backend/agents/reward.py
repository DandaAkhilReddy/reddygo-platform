"""
Reward Agent

Manages usage-based reward system with exponential multipliers for consistent users.
Handles points, badges, streaks, and tier progression.
"""

from .base import BaseAgent
from .tools import get_user_stats, update_user_stats
from typing import Dict, Any


# Reward Tiers with Exponential Multipliers
USAGE_TIERS = {
    "casual": {
        "threshold_min": 1,
        "threshold_max": 3,
        "description": "1-3 challenges/week",
        "multiplier": 1.0,
        "perks": ["Basic badges", "Leaderboard access"]
    },
    "active": {
        "threshold_min": 4,
        "threshold_max": 7,
        "description": "4-7 challenges/week",
        "multiplier": 1.5,
        "perks": ["Priority matching", "Custom challenges", "Exclusive badges"]
    },
    "dedicated": {
        "threshold_min": 8,
        "threshold_max": 14,
        "description": "8-14 challenges/week",
        "multiplier": 2.0,
        "perks": ["Ad-free experience", "Early features", "Pro badge"]
    },
    "elite": {
        "threshold_min": 15,
        "threshold_max": 999,
        "description": "15+ challenges/week",
        "multiplier": 3.0,
        "perks": ["VIP status", "Profile customization", "Merch discounts", "Priority support"]
    }
}

# Streak Bonuses
STREAK_BONUSES = {
    7: {"points": 100, "badge": "Week Warrior"},
    14: {"points": 250, "badge": "Two Week Champion"},
    30: {"points": 500, "badge": "Monthly Master"},
    60: {"points": 1200, "badge": "Consistency King"},
    90: {"points": 2000, "badge": "Quarter Champion"}
}


class RewardAgent(BaseAgent):
    """
    Usage-Based Reward Agent.

    Responsibilities:
    - Calculate points with usage multipliers
    - Track streaks (daily, weekly, monthly)
    - Award badges and achievements
    - Manage tier progression
    - Referral rewards
    - Gamification logic
    """

    def __init__(self):
        instructions = """
You are the Reward Agent for ReddyGo, managing the usage-based reward system.

**Core Philosophy:**
MORE USAGE = MORE REWARDS (exponential scaling to build habits)

**Tier System:**

🥉 **Casual** (1-3 challenges/week)
- 1.0x points multiplier
- Basic badges
- Leaderboard access

🥈 **Active** (4-7 challenges/week)
- 1.5x points multiplier
- Priority matching (find challenges faster)
- Custom challenges
- Exclusive badges

🥇 **Dedicated** (8-14 challenges/week)
- 2.0x points multiplier
- Ad-free experience
- Early access to new features
- Pro badge

💎 **Elite** (15+ challenges/week)
- 3.0x points multiplier (TRIPLE REWARDS!)
- VIP status
- Profile customization
- Merch discounts (10-30% off)
- Priority support

**Point Calculation:**

Base points per challenge: 100
Win bonus: +50
Personal record: +25
Help teammate: +15

Total points = (base + bonuses) × tier multiplier

Example:
- Casual user completes challenge: 100 × 1.0 = 100 points
- Elite user completes challenge: 100 × 3.0 = 300 points
- Elite user wins challenge: (100 + 50) × 3.0 = 450 points!

**Streak Bonuses (Addictive by Design):**

7 days: +100 points + "Week Warrior" badge
14 days: +250 points + "Two Week Champion" badge
30 days: +500 points + "Monthly Master" badge
60 days: +1,200 points + "Consistency King" badge
90 days: +2,000 points + "Quarter Champion" badge

**Referral Rewards:**

Invite friend (both get): 200 points
Friend completes 5 challenges: +500 bonus
Friend subscribes Pro: +1,000 points

**Badge System:**

Distance badges: 5K, 10K, Half Marathon, Marathon
Challenge types: Speed Demon (100 races), Zone Master (100 zone control)
Social: Team Player (50 team challenges), Motivator (100 encouragements)
Special: Early Adopter, Beta Tester, Community Leader

**Your Job:**

When a user completes an action:
1. Call get_user_stats to get weekly challenge count
2. Determine their tier based on USAGE_TIERS
3. Calculate points with multiplier
4. Check for streak bonuses
5. Check for new badges earned
6. Update stats with update_user_stats
7. Return engaging summary

**Tone:**
- Celebratory (gamify the experience)
- Specific (tell them exact points earned)
- Forward-looking (how many more for next tier)
- Addictive (make them want more)

**Example Response:**

"🎉 Challenge complete! You earned 450 points (150 base × 3.0x Elite multiplier)!

💎 Current status:
• Points: 12,450 (+450)
• Tier: Elite (15 challenges this week)
• Streak: 23 days 🔥 (7 more for 1,200 bonus!)

Keep crushing it! You're in the top 5% of users. 💪"
"""

        super().__init__(
            name="Reward",
            instructions=instructions,
            model="gpt-4o-mini",
            temperature=0.7
        )

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register reward agent tools."""

        # Tool 1: Get user stats
        self.add_tool(
            func=get_user_stats,
            description="Get user's current stats (points, tier, streak, etc.)",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID"}
                },
                "required": ["user_id"]
            }
        )

        # Tool 2: Update user stats
        self.add_tool(
            func=update_user_stats,
            description="Update user's stats after reward calculation",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID"},
                    "stats_update": {
                        "type": "object",
                        "description": "Stats to update (points, tier, badges, etc.)"
                    }
                },
                "required": ["user_id", "stats_update"]
            }
        )

        # Tool 3: Calculate usage rewards
        self.add_tool(
            func=self._calculate_usage_rewards,
            description="Calculate rewards based on usage patterns",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "action": {"type": "string", "description": "Action type: challenge_complete, win, pr, etc."},
                    "action_details": {"type": "object", "description": "Additional details"}
                },
                "required": ["user_id", "action"]
            }
        )

        # Tool 4: Award badge
        self.add_tool(
            func=self._award_badge,
            description="Award a badge to user",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "badge_id": {"type": "string"},
                    "badge_name": {"type": "string"}
                },
                "required": ["user_id", "badge_id", "badge_name"]
            }
        )

        # Tool 5: Calculate challenge points
        self.add_tool(
            func=self._calculate_challenge_points,
            description="Calculate points for a completed challenge",
            parameters={
                "type": "object",
                "properties": {
                    "challenge_result": {
                        "type": "object",
                        "description": "Challenge outcome (completed, won, pr, etc.)"
                    },
                    "weekly_challenge_count": {"type": "integer"}
                },
                "required": ["challenge_result", "weekly_challenge_count"]
            }
        )

    def _calculate_usage_rewards(self, user_id: str, action: str, action_details: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate rewards based on usage (actual implementation)."""
        stats = get_user_stats(user_id)

        weekly_challenges = stats.get("challenges_this_week", 0)
        current_streak = stats.get("current_streak", 0)

        # Determine tier
        tier = self._determine_tier(weekly_challenges)
        multiplier = USAGE_TIERS[tier]["multiplier"]

        # Base points
        base_points = 100

        # Bonuses
        bonuses = 0
        if action == "challenge_win":
            bonuses += 50
        if action_details and action_details.get("personal_record"):
            bonuses += 25

        # Apply multiplier
        total_points = int((base_points + bonuses) * multiplier)

        # Streak bonus
        streak_bonus = STREAK_BONUSES.get(current_streak, {}).get("points", 0)

        return {
            "points_earned": total_points,
            "streak_bonus": streak_bonus,
            "tier": tier,
            "multiplier": multiplier,
            "breakdown": {
                "base": base_points,
                "bonuses": bonuses,
                "multiplied": total_points
            },
            "next_tier_info": self._get_next_tier_info(tier, weekly_challenges)
        }

    def _determine_tier(self, weekly_challenges: int) -> str:
        """Determine user tier based on weekly challenge count."""
        for tier_name, tier_data in USAGE_TIERS.items():
            if tier_data["threshold_min"] <= weekly_challenges <= tier_data["threshold_max"]:
                return tier_name
        return "casual"

    def _get_next_tier_info(self, current_tier: str, weekly_challenges: int) -> Dict[str, Any]:
        """Get info about next tier progression."""
        tier_order = ["casual", "active", "dedicated", "elite"]

        if current_tier == "elite":
            return {
                "at_max_tier": True,
                "message": "You're at the highest tier! 🏆"
            }

        current_index = tier_order.index(current_tier)
        next_tier = tier_order[current_index + 1]
        next_tier_data = USAGE_TIERS[next_tier]

        challenges_needed = next_tier_data["threshold_min"] - weekly_challenges

        return {
            "next_tier": next_tier,
            "challenges_needed": challenges_needed,
            "next_multiplier": next_tier_data["multiplier"],
            "message": f"{challenges_needed} more challenge{'' if challenges_needed == 1 else 's'} this week to reach {next_tier.title()} tier!"
        }

    def _award_badge(self, user_id: str, badge_id: str, badge_name: str) -> Dict[str, Any]:
        """Award badge to user (placeholder)."""
        print(f"🏅 Badge awarded to {user_id}: {badge_name}")
        return {"badge_awarded": True, "badge_name": badge_name}

    def _calculate_challenge_points(self, challenge_result: Dict[str, Any], weekly_challenge_count: int) -> Dict[str, Any]:
        """Calculate points for challenge (placeholder)."""
        tier = self._determine_tier(weekly_challenge_count)
        multiplier = USAGE_TIERS[tier]["multiplier"]

        base = 100
        total = int(base * multiplier)

        return {
            "points": total,
            "tier": tier,
            "multiplier": multiplier
        }
