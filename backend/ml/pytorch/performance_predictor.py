"""
ReddyGo - PyTorch Performance Predictor
LSTM model to predict future workout performance based on historical data

Learning Focus:
- PyTorch LSTM architecture
- Time series prediction
- Custom training loops
- Model saving/loading
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import numpy as np


class PerformancePredictorLSTM(nn.Module):
    """
    LSTM Neural Network for predicting workout performance

    Architecture:
    - Input: Sequence of past workouts (distance, duration, pace, heart_rate, etc.)
    - Hidden: 2-layer LSTM with 50 units each
    - Output: Predicted next performance metrics

    Example:
        Input: Last 10 runs → Output: Predicted pace for next 5km
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 50, num_layers: int = 2, output_size: int = 3):
        """
        Args:
            input_size: Number of features per workout (distance, duration, pace, HR, etc.)
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            output_size: Number of prediction metrics (predicted_time, predicted_pace, confidence)
        """
        super(PerformancePredictorLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input shape: (batch, seq_len, features)
            dropout=0.2 if num_layers > 1 else 0
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 25)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(25, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
               Example: (32, 10, 10) = 32 users, last 10 workouts, 10 features each

        Returns:
            Output tensor of shape (batch_size, output_size)
            Example: (32, 3) = 32 predictions with 3 metrics each
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use only the last time step's output
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Fully connected layers
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class PerformancePredictor:
    """
    High-level interface for the performance prediction model
    Handles training, inference, and model persistence
    """

    def __init__(self, model_path: str = "models/performance_predictor.pth"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None  # For normalizing input data

    def build_model(self, input_size: int = 10, hidden_size: int = 50):
        """Initialize a new model"""
        self.model = PerformancePredictorLSTM(
            input_size=input_size,
            hidden_size=hidden_size
        ).to(self.device)
        print(f"Model initialized on {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")

    def train(self,
              train_data: List[Tuple[np.ndarray, np.ndarray]],
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001):
        """
        Train the model on workout history

        Args:
            train_data: List of (input_sequence, target) pairs
                       input_sequence: shape (seq_len, features) - past workouts
                       target: shape (output_size,) - next performance
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() first.")

        # Prepare data
        X_train = torch.stack([torch.FloatTensor(x) for x, _ in train_data])
        y_train = torch.stack([torch.FloatTensor(y) for _, y in train_data])

        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        print("Training complete!")

    def predict(self, workout_history: np.ndarray) -> dict:
        """
        Predict next performance based on workout history

        Args:
            workout_history: Shape (seq_len, features) - recent workouts
                            Example: Last 10 runs with [distance, duration, pace, HR, ...]

        Returns:
            Dict with predictions:
            {
                "predicted_time": float,  # Predicted time in minutes
                "predicted_pace": float,   # Predicted pace in min/km
                "confidence": float        # Confidence score 0-1
            }
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call build_model() or load_model() first.")

        self.model.eval()
        with torch.no_grad():
            # Prepare input
            x = torch.FloatTensor(workout_history).unsqueeze(0).to(self.device)  # Add batch dimension

            # Predict
            output = self.model(x)
            output = output.cpu().numpy()[0]  # Remove batch dimension

            return {
                "predicted_time": float(output[0]),
                "predicted_pace": float(output[1]),
                "confidence": float(output[2])
            }

    def save_model(self):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.model.lstm.input_size,
            'hidden_size': self.model.hidden_size,
        }, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load model from disk"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model = PerformancePredictorLSTM(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {self.model_path}")


# Example usage
if __name__ == "__main__":
    """
    Example: Train model to predict 5km running time
    """
    # Simulated training data (10 past workouts → predict next)
    # Features: [distance_km, duration_min, pace_min_per_km, avg_hr, max_hr,
    #            elevation_m, temperature_c, humidity_%, cadence, day_of_week]

    train_data = []
    for i in range(100):  # 100 training examples
        # Past 10 workouts
        history = np.random.randn(10, 10)  # Random data for demo
        # Target: [predicted_time, predicted_pace, confidence]
        target = np.array([25.0 + np.random.rand(), 5.0 + np.random.rand() * 0.5, 0.85])
        train_data.append((history, target))

    # Initialize and train
    predictor = PerformancePredictor()
    predictor.build_model(input_size=10, hidden_size=50)
    predictor.train(train_data, epochs=50)

    # Save model
    predictor.save_model()

    # Predict
    new_workout_history = np.random.randn(10, 10)
    prediction = predictor.predict(new_workout_history)
    print(f"\nPrediction for next 5km run:")
    print(f"  Time: {prediction['predicted_time']:.2f} minutes")
    print(f"  Pace: {prediction['predicted_pace']:.2f} min/km")
    print(f"  Confidence: {prediction['confidence']:.2%}")
