<template>
  <div class="min-h-screen flex items-center justify-center px-4 py-12">
    <Card class="w-full max-w-md" padding="lg">
      <template #header>
        <div class="text-center">
          <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            🏃 ReddyGo ML/AI
          </h1>
          <p class="text-gray-600 dark:text-gray-400">
            ML Learning Platform
          </p>
        </div>
      </template>

      <form @submit.prevent="handleSubmit" class="space-y-4">
        <Input
          v-model="email"
          type="email"
          label="Email"
          placeholder="your.email@example.com"
          required
          :error="emailError"
        />

        <Input
          v-model="password"
          type="password"
          label="Password"
          placeholder="••••••••"
          required
          :error="passwordError"
        />

        <div v-if="authStore.error" class="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <p class="text-sm text-red-600 dark:text-red-400">{{ authStore.error }}</p>
        </div>

        <Button
          type="submit"
          :loading="authStore.loading"
          fullWidth
          size="lg"
        >
          {{ isSignUp ? 'Sign Up' : 'Sign In' }}
        </Button>

        <div class="text-center">
          <button
            type="button"
            class="text-sm text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300"
            @click="toggleMode"
          >
            {{ isSignUp ? 'Already have an account? Sign in' : 'Need an account? Sign up' }}
          </button>
        </div>
      </form>

      <template #footer>
        <div class="text-center text-sm text-gray-500 dark:text-gray-400">
          <p>Demo Credentials:</p>
          <p class="font-mono text-xs mt-1">test@reddygo.com / password123</p>
        </div>
      </template>
    </Card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { useAuthStore } from '../stores/auth';
import Card from '../components/ui/Card.vue';
import Input from '../components/ui/Input.vue';
import Button from '../components/ui/Button.vue';

const router = useRouter();
const authStore = useAuthStore();

const email = ref('');
const password = ref('');
const isSignUp = ref(false);
const emailError = ref('');
const passwordError = ref('');

function toggleMode() {
  isSignUp.value = !isSignUp.value;
  authStore.clearError();
  emailError.value = '';
  passwordError.value = '';
}

async function handleSubmit() {
  // Reset errors
  emailError.value = '';
  passwordError.value = '';
  authStore.clearError();

  // Basic validation
  if (!email.value) {
    emailError.value = 'Email is required';
    return;
  }

  if (!password.value) {
    passwordError.value = 'Password is required';
    return;
  }

  if (password.value.length < 6) {
    passwordError.value = 'Password must be at least 6 characters';
    return;
  }

  try {
    if (isSignUp.value) {
      await authStore.signUp(email.value, password.value);
    } else {
      await authStore.signIn(email.value, password.value);
    }

    // Redirect to dashboard on success
    const redirect = router.currentRoute.value.query.redirect as string || '/';
    router.push(redirect);
  } catch (error) {
    console.error('Authentication error:', error);
  }
}
</script>
