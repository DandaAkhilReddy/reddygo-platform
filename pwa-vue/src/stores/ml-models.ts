import { defineStore } from 'pinia';
import { ref } from 'vue';
import pytorchApi from '../services/pytorch-api';
import tensorflowApi from '../services/tensorflow-api';
import qdrantApi from '../services/qdrant-api';
import type {
  PytorchPrediction,
  ChallengeRecommendation,
  VectorSearchResult,
  MLModelStatus
} from '../types/models';

export const useMLModelsStore = defineStore('mlModels', () => {
  // State
  const pytorchPrediction = ref<PytorchPrediction | null>(null);
  const tensorflowRecommendations = ref<ChallengeRecommendation[]>([]);
  const vectorSearchResults = ref<VectorSearchResult[]>([]);
  const modelStatus = ref<Record<string, MLModelStatus>>({});
  const loading = ref(false);
  const error = ref<string | null>(null);

  // PyTorch Actions
  async function predictPerformance(workouts: any[]) {
    loading.value = true;
    error.value = null;

    try {
      const prediction = await pytorchApi.predict(workouts);
      pytorchPrediction.value = prediction;
      return prediction;
    } catch (err: any) {
      error.value = err.message || 'Failed to predict performance';
      throw err;
    } finally {
      loading.value = false;
    }
  }

  // TensorFlow Actions
  async function getRecommendations(
    userId: number,
    candidateChallenges: number[],
    topK: number = 10
  ) {
    loading.value = true;
    error.value = null;

    try {
      const recommendations = await tensorflowApi.recommend(userId, candidateChallenges, topK);
      tensorflowRecommendations.value = recommendations;
      return recommendations;
    } catch (err: any) {
      error.value = err.message || 'Failed to get recommendations';
      throw err;
    } finally {
      loading.value = false;
    }
  }

  // Qdrant Actions
  async function searchWorkouts(query: string, userId?: string, limit: number = 5) {
    loading.value = true;
    error.value = null;

    try {
      const results = await qdrantApi.searchWorkouts(query, userId, limit);
      vectorSearchResults.value = results;
      return results;
    } catch (err: any) {
      error.value = err.message || 'Failed to search workouts';
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function searchMemories(query: string, userId: string, limit: number = 5) {
    loading.value = true;
    error.value = null;

    try {
      const results = await qdrantApi.searchMemories(query, userId, limit);
      vectorSearchResults.value = results;
      return results;
    } catch (err: any) {
      error.value = err.message || 'Failed to search memories';
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function searchExercises(query: string, limit: number = 5) {
    loading.value = true;
    error.value = null;

    try {
      const results = await qdrantApi.searchExercises(query, limit);
      vectorSearchResults.value = results;
      return results;
    } catch (err: any) {
      error.value = err.message || 'Failed to search exercises';
      throw err;
    } finally {
      loading.value = false;
    }
  }

  // Model Status
  async function loadModelStatus(modelName: string) {
    try {
      // Fetch model status from backend
      // This would be implemented based on your backend API
      console.log(`Loading status for ${modelName}`);
    } catch (err: any) {
      console.error(`Failed to load ${modelName} status:`, err);
    }
  }

  function clearError() {
    error.value = null;
  }

  return {
    // State
    pytorchPrediction,
    tensorflowRecommendations,
    vectorSearchResults,
    modelStatus,
    loading,
    error,
    // Actions
    predictPerformance,
    getRecommendations,
    searchWorkouts,
    searchMemories,
    searchExercises,
    loadModelStatus,
    clearError,
  };
});
