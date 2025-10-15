<template>
  <div class="min-h-screen">
    <header class="bg-white dark:bg-gray-800 shadow">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <button
          @click="router.push('/')"
          class="flex items-center space-x-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
        >
          <span>←</span>
          <span>Back to Dashboard</span>
        </button>
      </div>
    </header>

    <main class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div class="text-center mb-8">
        <div class="text-6xl mb-4">🔍</div>
        <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Vector Search
        </h1>
        <p class="text-gray-600 dark:text-gray-400">
          Semantic search using Qdrant vector database
        </p>
      </div>

      <Card>
        <div class="text-center py-12">
          <p class="text-gray-600 dark:text-gray-400">
            Qdrant Vector Search Demo - Coming Soon
          </p>
        </div>
      </Card>
    </main>
  </div>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router';
import Card from '../components/ui/Card.vue';

const router = useRouter();
</script>
