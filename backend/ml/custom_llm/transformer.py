"""
Day 1: Transformer Components from Scratch
Build self-attention, multi-head attention, and transformer blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SelfAttention(nn.Module):
    """
    Single-head self-attention mechanism

    This is the core of transformer models. It allows each token to attend to
    all other tokens in the sequence, learning relationships between words.

    Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Linear projections for Query, Key, Value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Scaling factor for stability
        self.scale = math.sqrt(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional mask for causal attention (batch_size, seq_len, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # TODO Day 1: Implement self-attention
        # 1. Project input to Q, K, V
        # 2. Compute attention scores: QK^T / sqrt(d_k)
        # 3. Apply mask (if provided) for causal attention
        # 4. Apply softmax to get attention weights
        # 5. Multiply by V to get output

        batch_size, seq_len, _ = x.shape

        # Step 1: Linear projections
        Q = self.query(x)  # (batch, seq_len, embed_dim)
        K = self.key(x)    # (batch, seq_len, embed_dim)
        V = self.value(x)  # (batch, seq_len, embed_dim)

        # Step 2: Compute attention scores
        # Q @ K^T = (batch, seq_len, embed_dim) @ (batch, embed_dim, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, seq_len, seq_len)

        # Step 3: Apply mask for causal (autoregressive) attention
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 4: Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)

        # Step 5: Apply attention to values
        output = torch.matmul(attn_weights, V)  # (batch, seq_len, embed_dim)

        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention allows the model to attend to different aspects
    of the input simultaneously (e.g., syntax, semantics, context).

    Each head learns different attention patterns.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projections for all heads (more efficient than separate layers)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        # TODO Day 1: Implement multi-head attention
        # 1. Project to Q, K, V and split into multiple heads
        # 2. Compute attention for each head in parallel
        # 3. Concatenate heads
        # 4. Apply output projection

        batch_size, seq_len, embed_dim = x.shape

        # Step 1: Project and split into heads
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Step 2: Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, num_heads, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)

        # Step 3: Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, head_dim)

        # Step 4: Concatenate heads
        attn_output = attn_output.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)

        # Step 5: Output projection
        output = self.out_proj(attn_output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network

    Applies two linear transformations with GELU activation in between.
    This allows the model to process each position independently.
    """

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        # TODO Day 1: Implement feed-forward network
        # 1. First linear layer + GELU activation
        # 2. Dropout
        # 3. Second linear layer
        # 4. Dropout

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Complete transformer block with:
    1. Multi-head self-attention
    2. Layer normalization
    3. Feed-forward network
    4. Residual connections

    This is the building block repeated N times in transformer models.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # Layer normalization (applied before attention - Pre-LN architecture)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            mask: Causal mask for autoregressive generation

        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        # TODO Day 1: Implement transformer block
        # Architecture: Pre-LN (layer norm before attention/FF)
        # 1. x = x + Attention(LayerNorm(x))
        # 2. x = x + FeedForward(LayerNorm(x))

        # Self-attention with residual connection
        attn_output = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_output)

        # Feed-forward with residual connection
        ff_output = self.ff(self.ln2(x))
        x = x + ff_output

        return x


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal (lower triangular) mask for autoregressive generation.

    This ensures that position i can only attend to positions <= i,
    preventing the model from "cheating" by looking at future tokens.

    Args:
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        Causal mask of shape (seq_len, seq_len)

    Example:
        seq_len = 4
        Returns:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


if __name__ == "__main__":
    # Test the components
    print("Testing Transformer Components...")

    batch_size = 2
    seq_len = 10
    embed_dim = 128
    num_heads = 4
    ff_dim = 512

    # Create random input
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Create causal mask
    mask = create_causal_mask(seq_len, x.device)

    # Test SelfAttention
    print("\n1. Testing SelfAttention...")
    self_attn = SelfAttention(embed_dim)
    out = self_attn(x, mask)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == x.shape, "SelfAttention output shape mismatch!"
    print("   ✓ SelfAttention working!")

    # Test MultiHeadAttention
    print("\n2. Testing MultiHeadAttention...")
    mha = MultiHeadAttention(embed_dim, num_heads)
    out = mha(x, mask)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == x.shape, "MultiHeadAttention output shape mismatch!"
    print("   ✓ MultiHeadAttention working!")

    # Test FeedForward
    print("\n3. Testing FeedForward...")
    ff = FeedForward(embed_dim, ff_dim)
    out = ff(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == x.shape, "FeedForward output shape mismatch!"
    print("   ✓ FeedForward working!")

    # Test TransformerBlock
    print("\n4. Testing TransformerBlock...")
    block = TransformerBlock(embed_dim, num_heads, ff_dim)
    out = block(x, mask)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == x.shape, "TransformerBlock output shape mismatch!"
    print("   ✓ TransformerBlock working!")

    print("\n✅ All components working correctly!")
    print("\nNext: Build complete GPT model in model.py")
