<template>
  <div class="min-h-screen">
    <!-- Header -->
    <header class="bg-white dark:bg-gray-800 shadow">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div class="flex items-center justify-between">
          <button
            @click="router.push('/')"
            class="flex items-center space-x-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
          >
            <span>←</span>
            <span>Back to Dashboard</span>
          </button>

          <div class="flex items-center space-x-3">
            <Badge variant="primary">PyTorch LSTM</Badge>
          </div>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div class="text-center mb-8">
        <div class="text-6xl mb-4">🔮</div>
        <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Performance Predictor
        </h1>
        <p class="text-gray-600 dark:text-gray-400">
          Use PyTorch LSTM to predict your next 5km race time
        </p>
      </div>

      <!-- Demo Content -->
      <div class="space-y-6">
        <Card>
          <template #header>
            <h2 class="text-xl font-bold text-gray-900 dark:text-white">
              📈 Your Past Runs
            </h2>
          </template>

          <div class="text-center py-8">
            <p class="text-gray-600 dark:text-gray-400 mb-4">
              This demo requires the backend API to be running
            </p>
            <Button @click="testPrediction">
              Test Prediction (Mock Data)
            </Button>
          </div>

          <div v-if="mlStore.pytorchPrediction" class="mt-6 p-4 bg-primary-50 dark:bg-primary-900/20 rounded-lg">
            <h3 class="font-bold text-gray-900 dark:text-white mb-3">🎯 AI Prediction:</h3>
            <div class="grid grid-cols-3 gap-4">
              <div>
                <p class="text-sm text-gray-600 dark:text-gray-400">Time</p>
                <p class="text-2xl font-bold text-gray-900 dark:text-white">
                  {{ mlStore.pytorchPrediction.predicted_time.toFixed(2) }} min
                </p>
              </div>
              <div>
                <p class="text-sm text-gray-600 dark:text-gray-400">Pace</p>
                <p class="text-2xl font-bold text-gray-900 dark:text-white">
                  {{ mlStore.pytorchPrediction.predicted_pace.toFixed(2) }} min/km
                </p>
              </div>
              <div>
                <p class="text-sm text-gray-600 dark:text-gray-400">Confidence</p>
                <p class="text-2xl font-bold text-gray-900 dark:text-white">
                  {{ (mlStore.pytorchPrediction.confidence * 100).toFixed(0) }}%
                </p>
              </div>
            </div>
          </div>

          <div v-if="mlStore.loading" class="mt-6">
            <LoadingSpinner centered text="Predicting performance..." />
          </div>

          <div v-if="mlStore.error" class="mt-6 p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
            <p class="text-red-600 dark:text-red-400">{{ mlStore.error }}</p>
          </div>
        </Card>

        <!-- Model Info -->
        <Card>
          <template #header>
            <h2 class="text-xl font-bold text-gray-900 dark:text-white">
              🧠 Model Architecture
            </h2>
          </template>

          <div class="space-y-3 text-sm">
            <div class="flex justify-between">
              <span class="text-gray-600 dark:text-gray-400">Type:</span>
              <span class="font-medium text-gray-900 dark:text-white">2-Layer LSTM</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-600 dark:text-gray-400">Hidden Units:</span>
              <span class="font-medium text-gray-900 dark:text-white">50</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-600 dark:text-gray-400">Input Features:</span>
              <span class="font-medium text-gray-900 dark:text-white">10 (distance, pace, HR, etc.)</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-600 dark:text-gray-400">Training Loss:</span>
              <span class="font-medium text-gray-900 dark:text-white">6.8</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-600 dark:text-gray-400">Parameters:</span>
              <span class="font-medium text-gray-900 dark:text-white">34,353</span>
            </div>
          </div>
        </Card>

        <!-- Learning Resources -->
        <Card>
          <template #header>
            <h2 class="text-xl font-bold text-gray-900 dark:text-white">
              📚 Learn More
            </h2>
          </template>

          <ul class="space-y-2 text-sm">
            <li class="flex items-center space-x-2">
              <span class="text-primary-500">→</span>
              <span class="text-gray-700 dark:text-gray-300">PyTorch LSTM documentation</span>
            </li>
            <li class="flex items-center space-x-2">
              <span class="text-primary-500">→</span>
              <span class="text-gray-700 dark:text-gray-300">Time series prediction tutorial</span>
            </li>
            <li class="flex items-center space-x-2">
              <span class="text-primary-500">→</span>
              <span class="text-gray-700 dark:text-gray-300">Model training best practices</span>
            </li>
          </ul>
        </Card>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router';
import { useMLModelsStore } from '../stores/ml-models';
import Card from '../components/ui/Card.vue';
import Button from '../components/ui/Button.vue';
import Badge from '../components/ui/Badge.vue';
import LoadingSpinner from '../components/ui/LoadingSpinner.vue';

const router = useRouter();
const mlStore = useMLModelsStore();

async function testPrediction() {
  // Mock workout data for testing
  const mockWorkouts = [
    { distance_km: 5, duration_min: 28.5, pace_min_per_km: 5.7 },
    { distance_km: 5, duration_min: 27.8, pace_min_per_km: 5.56 },
    { distance_km: 10, duration_min: 62.0, pace_min_per_km: 6.2 },
    { distance_km: 5, duration_min: 26.9, pace_min_per_km: 5.38 },
    { distance_km: 5, duration_min: 27.5, pace_min_per_km: 5.5 },
  ];

  try {
    await mlStore.predictPerformance(mockWorkouts);
  } catch (error) {
    console.error('Prediction error:', error);
  }
}
</script>
