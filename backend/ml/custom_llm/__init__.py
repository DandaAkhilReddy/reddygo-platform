"""
Custom LLM - GPT-style Transformer from Scratch

Day 1-2 learning project: Build a complete language model from scratch.
"""

from .model import GPTModel, create_small_gpt, create_medium_gpt
from .tokenizer import CharacterTokenizer, FitnessDatasetBuilder
from .transformer import (
    SelfAttention,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    create_causal_mask
)

__all__ = [
    'GPTModel',
    'create_small_gpt',
    'create_medium_gpt',
    'CharacterTokenizer',
    'FitnessDatasetBuilder',
    'SelfAttention',
    'MultiHeadAttention',
    'FeedForward',
    'TransformerBlock',
    'create_causal_mask',
]
