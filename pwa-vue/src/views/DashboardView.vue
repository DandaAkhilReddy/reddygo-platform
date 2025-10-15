<template>
  <div class="min-h-screen">
    <!-- Header -->
    <header class="bg-white dark:bg-gray-800 shadow">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div class="flex items-center justify-between">
          <div class="flex items-center space-x-3">
            <span class="text-3xl">🏃</span>
            <div>
              <h1 class="text-2xl font-bold text-gray-900 dark:text-white">
                ReddyGo ML/AI Platform
              </h1>
              <p class="text-sm text-gray-600 dark:text-gray-400">
                Interactive Machine Learning Showcase
              </p>
            </div>
          </div>

          <Button variant="outline" @click="handleSignOut">
            Sign Out
          </Button>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Welcome Section -->
      <div class="mb-8">
        <h2 class="text-xl font-semibold text-gray-900 dark:text-white mb-2">
          Welcome, {{ authStore.user?.name || 'User' }}! 👋
        </h2>
        <p class="text-gray-600 dark:text-gray-400">
          Explore ML/AI capabilities with interactive demos
        </p>
      </div>

      <!-- Stats Grid -->
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <Card padding="md">
          <div class="flex items-center space-x-3">
            <div class="p-3 bg-primary-100 dark:bg-primary-900 rounded-lg">
              <span class="text-2xl">🧠</span>
            </div>
            <div>
              <p class="text-sm text-gray-600 dark:text-gray-400">Models</p>
              <p class="text-2xl font-bold text-gray-900 dark:text-white">5</p>
            </div>
          </div>
        </Card>

        <Card padding="md">
          <div class="flex items-center space-x-3">
            <div class="p-3 bg-green-100 dark:bg-green-900 rounded-lg">
              <span class="text-2xl">🎯</span>
            </div>
            <div>
              <p class="text-sm text-gray-600 dark:text-gray-400">Accuracy</p>
              <p class="text-2xl font-bold text-gray-900 dark:text-white">73%</p>
            </div>
          </div>
        </Card>

        <Card padding="md">
          <div class="flex items-center space-x-3">
            <div class="p-3 bg-yellow-100 dark:bg-yellow-900 rounded-lg">
              <span class="text-2xl">💰</span>
            </div>
            <div>
              <p class="text-sm text-gray-600 dark:text-gray-400">Cost Saved</p>
              <p class="text-2xl font-bold text-gray-900 dark:text-white">$127</p>
            </div>
          </div>
        </Card>

        <Card padding="md">
          <div class="flex items-center space-x-3">
            <div class="p-3 bg-purple-100 dark:bg-purple-900 rounded-lg">
              <span class="text-2xl">📚</span>
            </div>
            <div>
              <p class="text-sm text-gray-600 dark:text-gray-400">Value</p>
              <p class="text-2xl font-bold text-gray-900 dark:text-white">$8K+</p>
            </div>
          </div>
        </Card>
      </div>

      <!-- ML Demos Grid -->
      <div>
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          🔥 Interactive Demos
        </h3>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <!-- PyTorch Demo -->
          <Card padding="md" hover @click="router.push('/pytorch')">
            <div class="text-center">
              <div class="text-5xl mb-4">🔮</div>
              <h4 class="text-lg font-bold text-gray-900 dark:text-white mb-2">
                PyTorch LSTM
              </h4>
              <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Predict workout performance using deep learning
              </p>
              <Badge variant="primary">LSTM Neural Network</Badge>
            </div>
          </Card>

          <!-- TensorFlow Demo -->
          <Card padding="md" hover @click="router.push('/tensorflow')">
            <div class="text-center">
              <div class="text-5xl mb-4">🎯</div>
              <h4 class="text-lg font-bold text-gray-900 dark:text-white mb-2">
                TensorFlow Recommender
              </h4>
              <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Get personalized challenge recommendations
              </p>
              <Badge variant="success">Collaborative Filtering</Badge>
            </div>
          </Card>

          <!-- Qdrant Demo -->
          <Card padding="md" hover @click="router.push('/qdrant')">
            <div class="text-center">
              <div class="text-5xl mb-4">🔍</div>
              <h4 class="text-lg font-bold text-gray-900 dark:text-white mb-2">
                Qdrant Vector Search
              </h4>
              <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Semantic search across workouts and memories
              </p>
              <Badge variant="warning">Vector Database</Badge>
            </div>
          </Card>

          <!-- LangChain Demo -->
          <Card padding="md" hover @click="router.push('/langchain')">
            <div class="text-center">
              <div class="text-5xl mb-4">🤖</div>
              <h4 class="text-lg font-bold text-gray-900 dark:text-white mb-2">
                LangChain AI Coach
              </h4>
              <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Chat with multi-agent AI coach
              </p>
              <Badge variant="primary">Multi-Agent System</Badge>
            </div>
          </Card>

          <!-- Cost Analysis -->
          <Card padding="md" hover @click="router.push('/cost-analysis')">
            <div class="text-center">
              <div class="text-5xl mb-4">💰</div>
              <h4 class="text-lg font-bold text-gray-900 dark:text-white mb-2">
                Cost Analysis
              </h4>
              <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Compare local vs cloud LLM costs
              </p>
              <Badge variant="success">Free LLaMA</Badge>
            </div>
          </Card>

          <!-- Coming Soon -->
          <Card padding="md">
            <div class="text-center opacity-60">
              <div class="text-5xl mb-4">📈</div>
              <h4 class="text-lg font-bold text-gray-900 dark:text-white mb-2">
                More Features
              </h4>
              <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Additional ML models coming soon
              </p>
              <Badge>Coming Soon</Badge>
            </div>
          </Card>
        </div>
      </div>

      <!-- Learning Path Section -->
      <div class="mt-12">
        <Card>
          <template #header>
            <h3 class="text-lg font-bold text-gray-900 dark:text-white">
              📚 4-Week Learning Path
            </h3>
          </template>

          <div class="space-y-3">
            <div class="flex items-center space-x-3">
              <span class="text-green-500 text-xl">✓</span>
              <div>
                <p class="font-medium text-gray-900 dark:text-white">Week 1: PyTorch Basics</p>
                <p class="text-sm text-gray-600 dark:text-gray-400">LSTM, training loops, model persistence</p>
              </div>
            </div>

            <div class="flex items-center space-x-3">
              <span class="text-blue-500 text-xl">→</span>
              <div>
                <p class="font-medium text-gray-900 dark:text-white">Week 2: Vector Databases</p>
                <p class="text-sm text-gray-600 dark:text-gray-400">Embeddings, semantic search, RAG patterns</p>
              </div>
            </div>

            <div class="flex items-center space-x-3">
              <span class="text-gray-400 text-xl">☐</span>
              <div>
                <p class="font-medium text-gray-600 dark:text-gray-400">Week 3: LangChain Agents</p>
                <p class="text-sm text-gray-600 dark:text-gray-400">Multi-agent systems, tool calling, orchestration</p>
              </div>
            </div>

            <div class="flex items-center space-x-3">
              <span class="text-gray-400 text-xl">☐</span>
              <div>
                <p class="font-medium text-gray-600 dark:text-gray-400">Week 4: Production Deployment</p>
                <p class="text-sm text-gray-600 dark:text-gray-400">API endpoints, PWA, monitoring</p>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router';
import { useAuthStore } from '../stores/auth';
import Card from '../components/ui/Card.vue';
import Button from '../components/ui/Button.vue';
import Badge from '../components/ui/Badge.vue';

const router = useRouter();
const authStore = useAuthStore();

async function handleSignOut() {
  try {
    await authStore.signOut();
    router.push('/login');
  } catch (error) {
    console.error('Sign out error:', error);
  }
}
</script>
