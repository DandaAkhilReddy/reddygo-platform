"""
ReddyGo Rewards Router

Usage-based reward system with exponential multipliers.
Manages points, badges, streaks, and tier progression.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from agents.reward import RewardAgent, USAGE_TIERS, STREAK_BONUSES
from database import get_supabase_client

router = APIRouter()

# Initialize Reward Agent
reward_agent = RewardAgent()


# Request/Response Models
class RewardStatus(BaseModel):
    """Current reward status for user."""
    user_id: str
    points_balance: int
    lifetime_points: int
    current_tier: str
    tier_multiplier: float
    current_streak: int
    longest_streak: int
    badges: List[Dict[str, Any]]
    perks: List[str]
    next_tier_info: Dict[str, Any]


class RewardHistory(BaseModel):
    """Single reward history entry."""
    action_type: str
    points_earned: int
    multiplier_applied: float
    timestamp: str


class LeaderboardEntry(BaseModel):
    """Leaderboard entry."""
    user_id: str
    user_name: str
    points: int
    tier: str
    rank: int


@router.get("/{user_id}", response_model=RewardStatus)
async def get_reward_status(user_id: str):
    """
    Get user's current reward status.

    Returns:
    - Points balance
    - Current tier & multiplier
    - Streak information
    - Badges earned
    - Available perks
    - Progress to next tier
    """
    supabase = get_supabase_client()

    # Get reward data
    result = supabase.table("users").select("stats").eq("id", user_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="User not found")

    stats = result.data[0].get("stats", {})

    # Get tier info
    weekly_challenges = stats.get("challenges_this_week", 0)
    tier = _determine_tier(weekly_challenges)
    tier_data = USAGE_TIERS[tier]

    # Calculate next tier info
    next_tier_info = _get_next_tier_info(tier, weekly_challenges)

    return RewardStatus(
        user_id=user_id,
        points_balance=stats.get("points_balance", 0),
        lifetime_points=stats.get("lifetime_points", 0),
        current_tier=tier,
        tier_multiplier=tier_data["multiplier"],
        current_streak=stats.get("current_streak", 0),
        longest_streak=stats.get("longest_streak", 0),
        badges=stats.get("badges", []),
        perks=tier_data["perks"],
        next_tier_info=next_tier_info
    )


@router.get("/{user_id}/history", response_model=List[RewardHistory])
async def get_reward_history(user_id: str, limit: int = 50):
    """
    Get user's reward history.

    Shows recent point-earning actions with multipliers applied.
    """
    # TODO: Implement reward_history table query
    # For now, return placeholder

    return [
        {
            "action_type": "challenge_complete",
            "points_earned": 150,
            "multiplier_applied": 1.5,
            "timestamp": "2025-10-08T12:00:00Z"
        }
    ]


@router.get("/leaderboard/top", response_model=List[LeaderboardEntry])
async def get_leaderboard(limit: int = 100, tier: Optional[str] = None):
    """
    Get top users by points.

    Args:
        limit: Number of users to return (default: 100)
        tier: Filter by tier (casual, active, dedicated, elite)

    Returns:
        Ranked list of top users
    """
    supabase = get_supabase_client()

    # Get users with stats
    query = supabase.table("users").select("id, name, stats")

    if tier:
        # TODO: Add tier filtering
        pass

    result = query.limit(limit).execute()

    if not result.data:
        return []

    # Sort by points
    users = result.data
    sorted_users = sorted(
        users,
        key=lambda u: u.get("stats", {}).get("points_balance", 0),
        reverse=True
    )

    # Create leaderboard entries
    leaderboard = []
    for rank, user in enumerate(sorted_users, start=1):
        stats = user.get("stats", {})
        weekly_challenges = stats.get("challenges_this_week", 0)
        tier = _determine_tier(weekly_challenges)

        leaderboard.append(LeaderboardEntry(
            user_id=user["id"],
            user_name=user.get("name", "Unknown"),
            points=stats.get("points_balance", 0),
            tier=tier,
            rank=rank
        ))

    return leaderboard


@router.post("/redeem")
async def redeem_points(user_id: str, perk_id: str, points_cost: int):
    """
    Redeem points for a perk.

    Args:
        user_id: User ID
        perk_id: Perk identifier
        points_cost: Points required

    Returns:
        Success status and updated balance
    """
    supabase = get_supabase_client()

    # Get current points
    result = supabase.table("users").select("stats").eq("id", user_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="User not found")

    stats = result.data[0].get("stats", {})
    current_points = stats.get("points_balance", 0)

    # Check if enough points
    if current_points < points_cost:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient points. Need {points_cost}, have {current_points}"
        )

    # Deduct points
    new_balance = current_points - points_cost
    stats["points_balance"] = new_balance

    # Update database
    supabase.table("users").update({"stats": stats}).eq("id", user_id).execute()

    return {
        "success": True,
        "perk_id": perk_id,
        "points_spent": points_cost,
        "new_balance": new_balance
    }


@router.post("/claim-streak")
async def claim_streak_bonus(user_id: str):
    """
    Claim daily streak bonus.

    Checks current streak and awards bonus points if milestone reached.
    """
    supabase = get_supabase_client()

    # Get stats
    result = supabase.table("users").select("stats").eq("id", user_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="User not found")

    stats = result.data[0].get("stats", {})
    current_streak = stats.get("current_streak", 0)

    # Check for streak milestone
    streak_bonus = STREAK_BONUSES.get(current_streak)

    if not streak_bonus:
        return {
            "bonus_available": False,
            "current_streak": current_streak,
            "next_milestone": _get_next_streak_milestone(current_streak)
        }

    # Award bonus
    bonus_points = streak_bonus["points"]
    badge = streak_bonus["badge"]

    stats["points_balance"] = stats.get("points_balance", 0) + bonus_points

    if "badges" not in stats:
        stats["badges"] = []

    stats["badges"].append({
        "name": badge,
        "earned_at": "utcnow()",
        "streak_length": current_streak
    })

    # Update database
    supabase.table("users").update({"stats": stats}).eq("id", user_id).execute()

    return {
        "bonus_claimed": True,
        "points_earned": bonus_points,
        "badge_earned": badge,
        "new_balance": stats["points_balance"],
        "current_streak": current_streak
    }


# Helper functions
def _determine_tier(weekly_challenges: int) -> str:
    """Determine tier based on weekly challenge count."""
    for tier_name, tier_data in USAGE_TIERS.items():
        if tier_data["threshold_min"] <= weekly_challenges <= tier_data["threshold_max"]:
            return tier_name
    return "casual"


def _get_next_tier_info(current_tier: str, weekly_challenges: int) -> Dict[str, Any]:
    """Get info about next tier."""
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
        "message": f"{challenges_needed} more challenge{'s' if challenges_needed != 1 else ''} this week to reach {next_tier.title()}!"
    }


def _get_next_streak_milestone(current_streak: int) -> int:
    """Get next streak milestone."""
    milestones = sorted(STREAK_BONUSES.keys())
    for milestone in milestones:
        if milestone > current_streak:
            return milestone
    return milestones[-1]  # Already at max
