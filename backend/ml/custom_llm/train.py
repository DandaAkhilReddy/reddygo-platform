"""
Day 1-2: Training Loop for Custom LLM
Train our GPT-style model from scratch
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import os
from tqdm import tqdm

from .model import GPTModel, create_small_gpt
from .tokenizer import CharacterTokenizer, FitnessDatasetBuilder


class TextDataset(Dataset):
    """
    Dataset for language modeling

    Converts text into input-target pairs for next-token prediction.
    """

    def __init__(self, texts: List[str], tokenizer: CharacterTokenizer, max_length: int = 128):
        """
        Args:
            texts: List of training texts
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Encode all texts
        self.examples = []
        for text in texts:
            encoded = tokenizer.encode(text, add_special_tokens=True)

            # Split into chunks if too long
            for i in range(0, len(encoded) - 1, max_length):
                chunk = encoded[i:i + max_length + 1]  # +1 for target
                if len(chunk) > 1:  # Need at least input + target
                    self.examples.append(chunk)

        print(f"✓ Dataset created: {len(self.examples)} training examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get training example

        Returns:
            input_ids: Input token sequence (length - 1)
            target_ids: Target token sequence (length - 1)
        """
        tokens = self.examples[idx]

        # Input is all tokens except the last
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)

        # Target is all tokens except the first (shifted by 1)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)

        return input_ids, target_ids


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate batch with padding

    Args:
        batch: List of (input_ids, target_ids) tuples

    Returns:
        Padded input_ids and target_ids tensors
    """
    input_ids = [item[0] for item in batch]
    target_ids = [item[1] for item in batch]

    # Pad sequences to same length
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids = nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=-1)  # -1 to ignore in loss

    return input_ids, target_ids


class Trainer:
    """
    Trainer for GPT model
    """

    def __init__(
        self,
        model: GPTModel,
        train_loader: DataLoader,
        learning_rate: float = 3e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device

        # Optimizer (AdamW with weight decay)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )

        # Learning rate scheduler (cosine annealing)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 10  # 10 epochs
        )

        print(f"✓ Trainer initialized")
        print(f"  Device: {device}")
        print(f"  Optimizer: AdamW (lr={learning_rate})")
        print(f"  Model parameters: {model.count_parameters():,}")

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch

        Args:
            epoch: Current epoch number

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            # Move to device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Forward pass
            logits, loss = self.model(input_ids, target_ids)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()
            self.scheduler.step()

            # Track loss
            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, filepath: str, epoch: int, loss: float) -> None:
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'embed_dim': self.model.embed_dim,
                'max_seq_len': self.model.max_seq_len
            }
        }

        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint saved to {filepath}")

    @staticmethod
    def load_checkpoint(filepath: str, model: GPTModel, device: str) -> Tuple[GPTModel, int]:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']

        print(f"✓ Checkpoint loaded from {filepath}")
        print(f"  Epoch: {epoch}")
        print(f"  Loss: {checkpoint['loss']:.4f}")

        return model, epoch


def generate_sample(
    model: GPTModel,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    device: str = "cpu"
) -> str:
    """
    Generate text from prompt

    Args:
        model: Trained GPT model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to use

    Returns:
        Generated text
    """
    model.eval()

    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=50
        )

    # Decode
    generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

    return generated_text


def main():
    """Main training loop"""
    print("=" * 70)
    print("Training Custom GPT Model on Fitness Data")
    print("=" * 70)
    print()

    # Configuration
    VOCAB_SIZE = 500
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-4
    MAX_SEQ_LENGTH = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Configuration:")
    print(f"  Vocab Size: {VOCAB_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Max Sequence Length: {MAX_SEQ_LENGTH}")
    print(f"  Device: {DEVICE}")
    print()

    # Step 1: Create corpus
    print("Step 1: Creating training corpus...")
    corpus = FitnessDatasetBuilder.create_fitness_corpus()
    print()

    # Step 2: Build tokenizer
    print("Step 2: Building tokenizer...")
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(corpus)
    tokenizer.save("backend/ml/custom_llm/tokenizer.json")
    print()

    # Step 3: Create dataset
    print("Step 3: Creating dataset...")
    dataset = TextDataset(corpus, tokenizer, max_length=MAX_SEQ_LENGTH)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch
    )
    print()

    # Step 4: Create model
    print("Step 4: Creating model...")
    model = create_small_gpt(vocab_size=tokenizer.vocab_size)
    print(f"  Model has {model.count_parameters():,} parameters")
    print()

    # Step 5: Create trainer
    print("Step 5: Initializing trainer...")
    trainer = Trainer(model, train_loader, learning_rate=LEARNING_RATE, device=DEVICE)
    print()

    # Step 6: Train
    print("Step 6: Training...")
    print("=" * 70)

    best_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = trainer.train_epoch(epoch)

        print(f"Epoch {epoch}/{NUM_EPOCHS} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint every 10 epochs or if best
        if epoch % 10 == 0 or avg_loss < best_loss:
            trainer.save_checkpoint(
                f"backend/ml/custom_llm/checkpoints/model_epoch_{epoch}.pt",
                epoch,
                avg_loss
            )
            if avg_loss < best_loss:
                best_loss = avg_loss
                trainer.save_checkpoint(
                    "backend/ml/custom_llm/checkpoints/best_model.pt",
                    epoch,
                    avg_loss
                )

        # Generate sample every 10 epochs
        if epoch % 10 == 0:
            print("\n" + "=" * 70)
            print("Sample Generation:")
            print("=" * 70)

            prompts = [
                "Running is",
                "Complete a",
                "Your average pace"
            ]

            for prompt in prompts:
                generated = generate_sample(
                    model, tokenizer, prompt,
                    max_tokens=50, temperature=0.8, device=DEVICE
                )
                print(f"\nPrompt: '{prompt}'")
                print(f"Generated: '{generated}'")

            print("=" * 70)
            print()

    print("\n✅ Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: backend/ml/custom_llm/checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
