<template>
  <div :class="containerClasses">
    <div :class="spinnerClasses">
      <div class="spinner-ring"></div>
    </div>
    <p v-if="text" :class="textClasses">{{ text }}</p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

interface Props {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  text?: string;
  centered?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  size: 'md',
  centered: false,
});

const containerClasses = computed(() => {
  return props.centered ? 'flex flex-col items-center justify-center' : '';
});

const spinnerClasses = computed(() => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
    xl: 'w-16 h-16',
  };

  return `spinner ${sizeClasses[props.size]}`;
});

const textClasses = computed(() => {
  const sizeClasses = {
    sm: 'text-sm mt-1',
    md: 'text-base mt-2',
    lg: 'text-lg mt-3',
    xl: 'text-xl mt-4',
  };

  return `text-gray-600 dark:text-gray-400 ${sizeClasses[props.size]}`;
});
</script>

<style scoped>
.spinner {
  position: relative;
}

.spinner-ring {
  width: 100%;
  height: 100%;
  border: 3px solid rgba(99, 102, 241, 0.1);
  border-top-color: rgb(99, 102, 241);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

@media (prefers-color-scheme: dark) {
  .spinner-ring {
    border-color: rgba(139, 92, 246, 0.1);
    border-top-color: rgb(139, 92, 246);
  }
}
</style>
