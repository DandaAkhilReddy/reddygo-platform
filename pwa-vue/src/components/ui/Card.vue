<template>
  <div :class="cardClasses">
    <div v-if="$slots.header || title" class="border-b border-gray-200 dark:border-gray-700 pb-4 mb-4">
      <slot name="header">
        <h3 class="text-lg font-bold text-gray-900 dark:text-white">{{ title }}</h3>
        <p v-if="subtitle" class="text-sm text-gray-600 dark:text-gray-400 mt-1">{{ subtitle }}</p>
      </slot>
    </div>

    <div class="card-content">
      <slot></slot>
    </div>

    <div v-if="$slots.footer" class="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
      <slot name="footer"></slot>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

interface Props {
  title?: string;
  subtitle?: string;
  padding?: 'none' | 'sm' | 'md' | 'lg';
  shadow?: boolean;
  hover?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  padding: 'md',
  shadow: true,
  hover: false,
});

const cardClasses = computed(() => {
  const baseClasses = 'bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700';

  const paddingClasses = {
    none: '',
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
  };

  const shadowClass = props.shadow ? 'shadow-lg' : '';
  const hoverClass = props.hover ? 'hover:shadow-xl hover:scale-[1.02] transition-all duration-200 cursor-pointer' : '';

  return `${baseClasses} ${paddingClasses[props.padding]} ${shadowClass} ${hoverClass}`;
});
</script>
