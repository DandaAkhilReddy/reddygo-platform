import { apiClient } from './api';
import type { PytorchPrediction, WorkoutHistory } from '../types/models';

export const pytorchApi = {
  // Predict next performance
  async predict(workouts: WorkoutHistory[]): Promise<PytorchPrediction> {
    // Convert workouts to 2D array format expected by PyTorch
    const workout_history = workouts.map(w => [
      w.distance_km,
      w.duration_min,
      w.pace_min_per_km,
      w.avg_hr || 0,
      w.max_hr || 0,
      w.elevation_m || 0,
      w.temperature_c || 20,
      w.humidity || 50,
      w.cadence || 180,
      w.day_of_week || 0,
    ]);

    const response = await apiClient.post<PytorchPrediction>(
      '/api/ml/pytorch/predict',
      { workout_history }
    );

    return response.data;
  },

  // Get model status
  async getModelStatus() {
    const response = await apiClient.get<{
      is_loaded: boolean;
      model_path: string;
      parameters: number;
    }>('/api/ml/pytorch/status');

    return response.data;
  },

  // Train model (for demo purposes)
  async trainModel(trainingData: { inputs: number[][][]; targets: number[][] }) {
    const response = await apiClient.post<{
      status: string;
      epochs: number;
      final_loss: number;
    }>('/api/ml/pytorch/train', trainingData);

    return response.data;
  },
};

export default pytorchApi;
