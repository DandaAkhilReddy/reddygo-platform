import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import langchainApi from '../services/langchain-api';
import type { ChatMessage } from '../types/models';

export const useChatStore = defineStore('chat', () => {
  // State
  const messages = ref<ChatMessage[]>([]);
  const loading = ref(false);
  const error = ref<string | null>(null);
  const useLocalLLM = ref(false);
  const totalCost = ref(0);
  const toolsUsedCount = ref<Record<string, number>>({});

  // Getters
  const messageCount = computed(() => messages.value.length);
  const lastMessage = computed(() => messages.value[messages.value.length - 1] || null);

  // Actions
  async function sendMessage(userId: string, content: string, context?: string) {
    loading.value = true;
    error.value = null;

    // Add user message
    const userMessage: ChatMessage = {
      role: 'user',
      content,
      timestamp: new Date(),
    };
    messages.value.push(userMessage);

    try {
      const response = await langchainApi.chat(userId, content, context, useLocalLLM.value);

      // Add assistant message
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
        tools_used: response.tools_used,
        cost_estimate: response.cost_estimate,
      };
      messages.value.push(assistantMessage);

      // Update cost tracking
      totalCost.value += response.cost_estimate;

      // Update tools used count
      response.tools_used.forEach(tool => {
        toolsUsedCount.value[tool] = (toolsUsedCount.value[tool] || 0) + 1;
      });

      return assistantMessage;
    } catch (err: any) {
      error.value = err.message || 'Failed to send message';
      // Remove the user message if sending failed
      messages.value.pop();
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function loadHistory(userId: string) {
    loading.value = true;
    error.value = null;

    try {
      const history = await langchainApi.getChatHistory(userId);
      messages.value = history;
    } catch (err: any) {
      error.value = err.message || 'Failed to load chat history';
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function clearHistory(userId: string) {
    loading.value = true;
    error.value = null;

    try {
      await langchainApi.clearHistory(userId);
      messages.value = [];
      totalCost.value = 0;
      toolsUsedCount.value = {};
    } catch (err: any) {
      error.value = err.message || 'Failed to clear history';
      throw err;
    } finally {
      loading.value = false;
    }
  }

  function toggleLLMMode() {
    useLocalLLM.value = !useLocalLLM.value;
  }

  function clearError() {
    error.value = null;
  }

  return {
    // State
    messages,
    loading,
    error,
    useLocalLLM,
    totalCost,
    toolsUsedCount,
    // Getters
    messageCount,
    lastMessage,
    // Actions
    sendMessage,
    loadHistory,
    clearHistory,
    toggleLLMMode,
    clearError,
  };
});
