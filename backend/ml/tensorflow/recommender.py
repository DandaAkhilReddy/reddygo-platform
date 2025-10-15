"""
ReddyGo - TensorFlow Challenge Recommender
Collaborative filtering model to recommend challenges based on user preferences

Learning Focus:
- TensorFlow/Keras API
- Collaborative filtering
- Embedding layers
- Model deployment with TF Serving
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import List, Dict, Tuple


class ChallengeRecommender(keras.Model):
    """
    Neural Collaborative Filtering model for challenge recommendations

    Architecture:
    - User Embedding: Learn user preference vectors
    - Challenge Embedding: Learn challenge feature vectors
    - MLP: Combine embeddings for prediction

    Example:
        Input: user_id=123, challenge_id=456
        Output: Probability user will enjoy this challenge (0-1)
    """

    def __init__(self, num_users: int, num_challenges: int, embedding_size: int = 50):
        """
        Args:
            num_users: Total number of users in the system
            num_challenges: Total number of challenges
            embedding_size: Dimension of embedding vectors
        """
        super(ChallengeRecommender, self).__init__()

        self.embedding_size = embedding_size

        # User embedding layer
        self.user_embedding = layers.Embedding(
            input_dim=num_users + 1,  # +1 for unknown users
            output_dim=embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name='user_embedding'
        )

        # Challenge embedding layer
        self.challenge_embedding = layers.Embedding(
            input_dim=num_challenges + 1,  # +1 for unknown challenges
            output_dim=embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name='challenge_embedding'
        )

        # MLP layers to combine embeddings
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(64, activation='relu')
        self.dropout2 = layers.Dropout(0.2)
        self.dense3 = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        """
        Forward pass

        Args:
            inputs: Tuple of (user_ids, challenge_ids)
                   Each is a tensor of shape (batch_size,)
            training: Whether in training mode

        Returns:
            Prediction scores of shape (batch_size, 1)
        """
        user_ids, challenge_ids = inputs

        # Get embeddings
        user_vector = self.user_embedding(user_ids)  # (batch_size, embedding_size)
        challenge_vector = self.challenge_embedding(challenge_ids)  # (batch_size, embedding_size)

        # Concatenate embeddings
        concat = layers.concatenate([user_vector, challenge_vector])  # (batch_size, 2 * embedding_size)

        # MLP
        x = self.dense1(concat)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        output = self.output_layer(x)

        return output


class RecommenderSystem:
    """
    High-level interface for challenge recommendation
    """

    def __init__(self, model_path: str = "models/recommender"):
        self.model_path = model_path
        self.model = None
        self.user_id_map = {}  # Map user IDs to integers
        self.challenge_id_map = {}  # Map challenge IDs to integers

    def build_model(self, num_users: int, num_challenges: int, embedding_size: int = 50):
        """Initialize a new model"""
        self.model = ChallengeRecommender(
            num_users=num_users,
            num_challenges=num_challenges,
            embedding_size=embedding_size
        )

        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        print(f"Model built with {num_users} users and {num_challenges} challenges")

    def train(self,
              user_ids: np.ndarray,
              challenge_ids: np.ndarray,
              labels: np.ndarray,
              epochs: int = 50,
              batch_size: int = 128,
              validation_split: float = 0.2):
        """
        Train the recommender model

        Args:
            user_ids: Array of user IDs (integers)
            challenge_ids: Array of challenge IDs (integers)
            labels: Binary labels (1 = user liked/completed, 0 = didn't like)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]

        # Train
        history = self.model.fit(
            x=[user_ids, challenge_ids],
            y=labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        print("Training complete!")
        return history

    def recommend_challenges(self, user_id: int, candidate_challenges: List[int], top_k: int = 10) -> List[Dict]:
        """
        Recommend top-k challenges for a user

        Args:
            user_id: User ID (integer)
            candidate_challenges: List of challenge IDs to rank
            top_k: Number of top recommendations to return

        Returns:
            List of dicts: [{"challenge_id": int, "score": float}, ...]
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

        # Prepare inputs
        n_candidates = len(candidate_challenges)
        user_ids = np.full(n_candidates, user_id)
        challenge_ids = np.array(candidate_challenges)

        # Predict scores
        scores = self.model.predict([user_ids, challenge_ids], verbose=0)
        scores = scores.flatten()

        # Sort by score
        top_indices = np.argsort(scores)[::-1][:top_k]

        recommendations = [
            {
                "challenge_id": int(candidate_challenges[idx]),
                "score": float(scores[idx]),
                "rank": i + 1
            }
            for i, idx in enumerate(top_indices)
        ]

        return recommendations

    def get_similar_challenges(self, challenge_id: int, top_k: int = 5) -> List[Dict]:
        """
        Find challenges similar to a given challenge using embeddings

        Args:
            challenge_id: Challenge ID
            top_k: Number of similar challenges to return

        Returns:
            List of similar challenge IDs with similarity scores
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

        # Get challenge embedding
        challenge_embedding = self.model.challenge_embedding(np.array([challenge_id]))
        challenge_embedding = challenge_embedding.numpy()[0]  # Shape: (embedding_size,)

        # Get all challenge embeddings
        all_challenge_ids = np.arange(self.model.challenge_embedding.input_dim)
        all_embeddings = self.model.challenge_embedding(all_challenge_ids).numpy()

        # Compute cosine similarity
        similarities = np.dot(all_embeddings, challenge_embedding) / (
            np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(challenge_embedding)
        )

        # Get top-k (excluding itself)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]

        similar_challenges = [
            {
                "challenge_id": int(idx),
                "similarity": float(similarities[idx])
            }
            for idx in top_indices
        ]

        return similar_challenges

    def save_model(self):
        """Save model to disk"""
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load model from disk"""
        self.model = keras.models.load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")


# Example usage
if __name__ == "__main__":
    """
    Example: Train recommender to suggest challenges
    """
    # Simulated data
    num_users = 1000
    num_challenges = 200
    num_interactions = 10000

    # Generate random interactions
    user_ids = np.random.randint(0, num_users, size=num_interactions)
    challenge_ids = np.random.randint(0, num_challenges, size=num_interactions)
    labels = np.random.randint(0, 2, size=num_interactions)  # 0 or 1

    # Initialize recommender
    recommender = RecommenderSystem()
    recommender.build_model(num_users=num_users, num_challenges=num_challenges, embedding_size=50)

    # Train
    recommender.train(
        user_ids=user_ids,
        challenge_ids=challenge_ids,
        labels=labels,
        epochs=20,
        batch_size=128
    )

    # Save model
    recommender.save_model()

    # Recommend challenges for a user
    test_user_id = 42
    candidate_challenges = list(range(200))  # All challenges
    recommendations = recommender.recommend_challenges(test_user_id, candidate_challenges, top_k=10)

    print(f"\nTop 10 challenge recommendations for user {test_user_id}:")
    for rec in recommendations:
        print(f"  Rank {rec['rank']}: Challenge {rec['challenge_id']} (score: {rec['score']:.4f})")

    # Find similar challenges
    similar = recommender.get_similar_challenges(challenge_id=10, top_k=5)
    print(f"\nChallenges similar to challenge 10:")
    for sim in similar:
        print(f"  Challenge {sim['challenge_id']} (similarity: {sim['similarity']:.4f})")
