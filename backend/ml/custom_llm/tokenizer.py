"""
Day 1: Simple Character-Level Tokenizer
Build a basic tokenizer for training our custom LLM
"""

from typing import List, Dict
import json
import os


class CharacterTokenizer:
    """
    Simple character-level tokenizer

    This is the simplest tokenizer - each character is a token.
    Good for learning, but not production-ready (use BPE/WordPiece for that).
    """

    def __init__(self):
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"  # Beginning of sequence
        self.eos_token = "<EOS>"  # End of sequence

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from training texts

        Args:
            texts: List of training texts
        """
        # Collect all unique characters
        chars = set()
        for text in texts:
            chars.update(text)

        # Sort for reproducibility
        chars = sorted(list(chars))

        # Add special tokens first
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

        # Build mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(special_tokens + chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

        print(f"✓ Vocabulary built: {self.vocab_size} tokens")
        print(f"  Special tokens: {special_tokens}")
        print(f"  Character tokens: {len(chars)}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token indices

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token indices
        """
        # Convert each character to index
        indices = []

        if add_special_tokens:
            indices.append(self.char_to_idx[self.bos_token])

        for char in text:
            idx = self.char_to_idx.get(char, self.char_to_idx[self.unk_token])
            indices.append(idx)

        if add_special_tokens:
            indices.append(self.char_to_idx[self.eos_token])

        return indices

    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token indices to text

        Args:
            indices: List of token indices
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        special_token_set = {
            self.char_to_idx[self.pad_token],
            self.char_to_idx[self.unk_token],
            self.char_to_idx[self.bos_token],
            self.char_to_idx[self.eos_token]
        }

        chars = []
        for idx in indices:
            if skip_special_tokens and idx in special_token_set:
                continue
            chars.append(self.idx_to_char.get(idx, self.unk_token))

        return ''.join(chars)

    def save(self, filepath: str) -> None:
        """Save tokenizer to file"""
        data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()},  # JSON requires string keys
            'vocab_size': self.vocab_size,
            'special_tokens': {
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'bos_token': self.bos_token,
                'eos_token': self.eos_token
            }
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Tokenizer saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'CharacterTokenizer':
        """Load tokenizer from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.char_to_idx = data['char_to_idx']
        tokenizer.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
        tokenizer.vocab_size = data['vocab_size']

        special_tokens = data['special_tokens']
        tokenizer.pad_token = special_tokens['pad_token']
        tokenizer.unk_token = special_tokens['unk_token']
        tokenizer.bos_token = special_tokens['bos_token']
        tokenizer.eos_token = special_tokens['eos_token']

        print(f"✓ Tokenizer loaded from {filepath}")
        return tokenizer


class FitnessDatasetBuilder:
    """
    Build training dataset from fitness-related text

    This creates a simple dataset for our custom LLM to learn fitness language.
    """

    @staticmethod
    def create_fitness_corpus() -> List[str]:
        """Create a simple fitness text corpus for training"""

        fitness_texts = [
            # Running tips
            "Running is a great way to improve cardiovascular health and burn calories.",
            "Start with a warm-up jog before increasing your pace.",
            "Maintain good posture while running: keep your back straight and head up.",
            "Remember to stay hydrated during long runs.",
            "Cool down with light stretching after your run.",

            # Challenge descriptions
            "Complete a 5K run in under 30 minutes.",
            "Sprint challenge: Run 100 meters as fast as you can.",
            "Endurance test: Maintain a steady pace for 45 minutes.",
            "Hill workout: Find a steep incline and run uphill for maximum intensity.",
            "Interval training: Alternate between sprinting and jogging.",

            # Coaching advice
            "Focus on your breathing rhythm while running.",
            "Gradually increase your distance to avoid injury.",
            "Listen to your body and take rest days when needed.",
            "Proper running shoes are essential for injury prevention.",
            "Track your progress to stay motivated.",

            # Performance metrics
            "Your average pace is 6 minutes per kilometer.",
            "Great job! You've improved your 5K time by 2 minutes.",
            "Your heart rate reached 160 BPM during the sprint.",
            "You burned approximately 400 calories during this workout.",
            "Your cadence is 180 steps per minute, which is optimal.",

            # Motivational messages
            "Every run makes you stronger, keep going!",
            "Push through the discomfort, you're building endurance.",
            "Consistency is key to improving your running performance.",
            "You're doing great! Keep up the good work.",
            "Remember why you started, and keep moving forward.",

            # GPS/Location context
            "You're running through the park, maintain your pace.",
            "Approaching the 2 kilometer mark, keep it up!",
            "You've entered a hilly section, adjust your effort accordingly.",
            "Great job completing the lap around the track!",
            "You're nearing your starting point, finish strong!",
        ]

        return fitness_texts

    @staticmethod
    def save_corpus(texts: List[str], filepath: str) -> None:
        """Save corpus to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        print(f"✓ Corpus saved to {filepath} ({len(texts)} texts)")


if __name__ == "__main__":
    print("Testing Character Tokenizer...\n")

    # Create fitness corpus
    corpus = FitnessDatasetBuilder.create_fitness_corpus()
    print(f"✓ Created fitness corpus: {len(corpus)} texts\n")

    # Build tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(corpus)
    print()

    # Test encoding/decoding
    test_text = "Running is fun!"
    print(f"Original text: '{test_text}'")

    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")

    decoded = tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'")
    print()

    # Verify round-trip
    assert decoded == test_text, "Encoding/decoding mismatch!"
    print("✅ Round-trip encoding/decoding successful!\n")

    # Save tokenizer
    save_path = "backend/ml/custom_llm/tokenizer.json"
    tokenizer.save(save_path)
    print()

    # Test loading
    loaded_tokenizer = CharacterTokenizer.load(save_path)
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
    print("✅ Tokenizer save/load working!\n")

    # Save corpus
    corpus_path = "backend/ml/custom_llm/fitness_corpus.txt"
    FitnessDatasetBuilder.save_corpus(corpus, corpus_path)
    print()

    print("✅ All tokenizer tests passed!")
    print("\nNext: Build training loop in train.py")
