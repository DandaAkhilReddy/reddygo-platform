import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import { firebaseAuth } from '../services/firebase';
import type { User } from '../types/user';

export const useAuthStore = defineStore('auth', () => {
  // State
  const user = ref<User | null>(null);
  const loading = ref(false);
  const error = ref<string | null>(null);

  // Getters
  const isAuthenticated = computed(() => user.value !== null);
  const userId = computed(() => user.value?.id || null);

  // Actions
  async function signIn(email: string, password: string) {
    loading.value = true;
    error.value = null;

    try {
      const firebaseUser = await firebaseAuth.signIn(email, password);
      user.value = {
        id: firebaseUser.uid,
        email: firebaseUser.email || '',
        name: firebaseUser.displayName || 'User',
        created_at: new Date(firebaseUser.metadata.creationTime || Date.now()),
      };
    } catch (err: any) {
      error.value = err.message || 'Failed to sign in';
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function signUp(email: string, password: string) {
    loading.value = true;
    error.value = null;

    try {
      const firebaseUser = await firebaseAuth.signUp(email, password);
      user.value = {
        id: firebaseUser.uid,
        email: firebaseUser.email || '',
        name: 'New User',
        created_at: new Date(),
      };
    } catch (err: any) {
      error.value = err.message || 'Failed to sign up';
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function signOut() {
    loading.value = true;
    error.value = null;

    try {
      await firebaseAuth.signOut();
      user.value = null;
    } catch (err: any) {
      error.value = err.message || 'Failed to sign out';
      throw err;
    } finally {
      loading.value = false;
    }
  }

  function setUser(firebaseUser: any) {
    if (firebaseUser) {
      user.value = {
        id: firebaseUser.uid,
        email: firebaseUser.email || '',
        name: firebaseUser.displayName || 'User',
        created_at: new Date(firebaseUser.metadata.creationTime || Date.now()),
      };
    } else {
      user.value = null;
    }
  }

  function clearError() {
    error.value = null;
  }

  return {
    // State
    user,
    loading,
    error,
    // Getters
    isAuthenticated,
    userId,
    // Actions
    signIn,
    signUp,
    signOut,
    setUser,
    clearError,
  };
});
