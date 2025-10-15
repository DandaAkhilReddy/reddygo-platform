"""
Day 1: GPT-style Language Model from Scratch
Decoder-only transformer for autoregressive text generation
"""

import torch
import torch.nn as nn
from typing import Optional
from .transformer import TransformerBlock, create_causal_mask


class GPTModel(nn.Module):
    """
    GPT-style decoder-only transformer model

    Architecture:
    1. Token embedding
    2. Positional embedding
    3. N transformer blocks
    4. Layer normalization
    5. Linear projection to vocabulary

    This is a simplified version of GPT-2/GPT-3 architecture.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension (model dimension)
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            ff_dim: Feed-forward dimension (usually 4 * embed_dim)
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token embeddings (learn representation for each token)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional embeddings (learn position information)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Dropout for embeddings
        self.dropout = nn.Dropout(dropout)

        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.ln_f = nn.LayerNorm(embed_dim)

        # Output projection to vocabulary (language modeling head)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie weights between input and output embeddings (common practice)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Xavier/Kaiming initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            input_ids: Token indices (batch_size, seq_len)
            targets: Target token indices for training (batch_size, seq_len)

        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss (if targets provided)
        """
        batch_size, seq_len = input_ids.shape

        # Check sequence length
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max {self.max_seq_len}"

        # TODO Day 1: Implement forward pass
        # 1. Token embeddings
        # 2. Positional embeddings
        # 3. Add embeddings and apply dropout
        # 4. Pass through transformer blocks
        # 5. Final layer norm
        # 6. Project to vocabulary
        # 7. Compute loss if targets provided

        # Step 1: Get token embeddings
        token_emb = self.token_embedding(input_ids)  # (batch, seq_len, embed_dim)

        # Step 2: Get positional embeddings
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)  # (1, seq_len)
        pos_emb = self.position_embedding(positions)  # (1, seq_len, embed_dim)

        # Step 3: Combine embeddings
        x = self.dropout(token_emb + pos_emb)  # (batch, seq_len, embed_dim)

        # Step 4: Create causal mask
        mask = create_causal_mask(seq_len, input_ids.device)

        # Step 5: Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)

        # Step 6: Final layer normalization
        x = self.ln_f(x)

        # Step 7: Project to vocabulary
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        # Step 8: Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1  # Ignore padding tokens
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively

        Args:
            input_ids: Prompt token indices (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (only sample from top k tokens)

        Returns:
            Generated token indices (batch_size, seq_len + max_new_tokens)
        """
        # TODO Day 1: Implement text generation
        # 1. For each new token:
        #    a. Get logits for current sequence
        #    b. Take logits for last position
        #    c. Apply temperature scaling
        #    d. Apply top-k filtering (optional)
        #    e. Sample next token
        #    f. Append to sequence
        # 2. Return generated sequence

        for _ in range(max_new_tokens):
            # Crop input to max sequence length
            input_ids_cropped = input_ids[:, -self.max_seq_len:]

            # Forward pass
            logits, _ = self(input_ids_cropped)

            # Get logits for last position
            logits = logits[:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Convert to probabilities
            probs = nn.functional.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_small_gpt(vocab_size: int = 5000) -> GPTModel:
    """Create a small GPT model for experimentation (similar to GPT-2 Small)"""
    return GPTModel(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        ff_dim=1024,
        max_seq_len=512,
        dropout=0.1
    )


def create_medium_gpt(vocab_size: int = 10000) -> GPTModel:
    """Create a medium GPT model (similar to GPT-2 Medium)"""
    return GPTModel(
        vocab_size=vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=12,
        ff_dim=2048,
        max_seq_len=1024,
        dropout=0.1
    )


if __name__ == "__main__":
    # Test the model
    print("Testing GPT Model...")

    vocab_size = 5000
    batch_size = 2
    seq_len = 32

    # Create model
    model = create_small_gpt(vocab_size)
    print(f"\n✓ Model created")
    print(f"  Parameters: {model.count_parameters():,}")

    # Create random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Test forward pass
    print(f"\n✓ Testing forward pass...")
    logits, loss = model(input_ids, targets)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    # Test generation
    print(f"\n✓ Testing generation...")
    prompt = torch.randint(0, vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"  Prompt shape: {prompt.shape}")
    print(f"  Generated shape: {generated.shape}")
    print(f"  Generated {generated.shape[1] - prompt.shape[1]} new tokens")

    print("\n✅ Model working correctly!")
    print("\nNext Steps:")
    print("  1. Create tokenizer (tokenizer.py)")
    print("  2. Prepare training data")
    print("  3. Train the model (train.py)")
