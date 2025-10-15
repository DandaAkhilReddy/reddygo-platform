import { apiClient } from './api';
import type { QdrantSearchRequest } from '../types/api';
import type { VectorSearchResult } from '../types/models';

export const qdrantApi = {
  // Search workouts
  async searchWorkouts(
    query: string,
    userId?: string,
    limit: number = 5
  ): Promise<VectorSearchResult[]> {
    const response = await apiClient.post<VectorSearchResult[]>(
      '/api/ml/qdrant/search',
      {
        query,
        collection: 'workouts',
        user_id: userId,
        limit,
      } as QdrantSearchRequest
    );

    return response.data;
  },

  // Search memories
  async searchMemories(
    query: string,
    userId: string,
    limit: number = 5
  ): Promise<VectorSearchResult[]> {
    const response = await apiClient.post<VectorSearchResult[]>(
      '/api/ml/qdrant/search',
      {
        query,
        collection: 'memories',
        user_id: userId,
        limit,
      } as QdrantSearchRequest
    );

    return response.data;
  },

  // Search exercises
  async searchExercises(
    query: string,
    limit: number = 5
  ): Promise<VectorSearchResult[]> {
    const response = await apiClient.post<VectorSearchResult[]>(
      '/api/ml/qdrant/search',
      {
        query,
        collection: 'exercises',
        limit,
      } as QdrantSearchRequest
    );

    return response.data;
  },

  // Add workout
  async addWorkout(userId: string, workoutData: any) {
    const response = await apiClient.post<{ id: string }>(
      '/api/ml/qdrant/workouts',
      { user_id: userId, ...workoutData }
    );

    return response.data;
  },

  // Add memory
  async addMemory(userId: string, memoryText: string, memoryType: string = 'preference') {
    const response = await apiClient.post<{ id: string }>(
      '/api/ml/qdrant/memories',
      { user_id: userId, memory_text: memoryText, memory_type: memoryType }
    );

    return response.data;
  },

  // Get collection stats
  async getStats() {
    const response = await apiClient.get<{
      workouts_count: number;
      memories_count: number;
      exercises_count: number;
      embedding_dim: number;
    }>('/api/ml/qdrant/stats');

    return response.data;
  },
};

export default qdrantApi;
