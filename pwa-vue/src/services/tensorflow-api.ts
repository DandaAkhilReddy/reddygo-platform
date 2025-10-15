import { apiClient } from './api';
import type { TensorflowRecommendRequest } from '../types/api';
import type { ChallengeRecommendation } from '../types/models';

export const tensorflowApi = {
  // Get challenge recommendations
  async recommend(
    userId: number,
    candidateChallenges: number[],
    topK: number = 10
  ): Promise<ChallengeRecommendation[]> {
    const response = await apiClient.post<ChallengeRecommendation[]>(
      '/api/ml/tensorflow/recommend',
      {
        user_id: userId,
        candidate_challenges: candidateChallenges,
        top_k: topK,
      } as TensorflowRecommendRequest
    );

    return response.data;
  },

  // Get similar challenges
  async getSimilarChallenges(challengeId: number, topK: number = 5) {
    const response = await apiClient.get<Array<{
      challenge_id: number;
      similarity: number;
    }>>(`/api/ml/tensorflow/similar/${challengeId}?top_k=${topK}`);

    return response.data;
  },

  // Get model metrics
  async getModelMetrics() {
    const response = await apiClient.get<{
      accuracy: number;
      auc: number;
      loss: number;
      num_users: number;
      num_challenges: number;
    }>('/api/ml/tensorflow/metrics');

    return response.data;
  },
};

export default tensorflowApi;
