"""
ReddyGo AI Agents Package

6-agent specialized system using OpenAI Agents SDK:
- Coordinator: Challenge orchestration
- Validation: GPS/sensor validation with AI
- Coach: Personalized training advice
- Social: Community engagement
- Safety: Restricted zone detection
- Reward: Achievement tracking
"""

from .coordinator import CoordinatorAgent
from .coach import CoachAgent
from .validation import ValidationAgent
from .social import SocialAgent
from .safety import SafetyAgent
from .reward import RewardAgent

__all__ = [
    'CoordinatorAgent',
    'CoachAgent',
    'ValidationAgent',
    'SocialAgent',
    'SafetyAgent',
    'RewardAgent'
]
