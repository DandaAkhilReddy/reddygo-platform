import { apiClient } from './api';
import type { LangchainChatRequest, LangchainChatResponse } from '../types/api';
import type { ChatMessage } from '../types/models';

export const langchainApi = {
  // Chat with AI agent
  async chat(
    userId: string,
    query: string,
    context?: string,
    useLocalLlm: boolean = false
  ): Promise<LangchainChatResponse> {
    const response = await apiClient.post<LangchainChatResponse>(
      '/api/ml/langchain/chat',
      {
        user_id: userId,
        query,
        context,
        use_local_llm: useLocalLlm,
      } as LangchainChatRequest
    );

    return response.data;
  },

  // Get chat history
  async getChatHistory(userId: string): Promise<ChatMessage[]> {
    const response = await apiClient.get<ChatMessage[]>(
      `/api/ml/langchain/history/${userId}`
    );

    return response.data;
  },

  // Clear chat history
  async clearHistory(userId: string) {
    const response = await apiClient.delete<{ status: string }>(
      `/api/ml/langchain/history/${userId}`
    );

    return response.data;
  },

  // Get available tools
  async getTools() {
    const response = await apiClient.get<Array<{
      name: string;
      description: string;
      category: string;
    }>>('/api/ml/langchain/tools');

    return response.data;
  },

  // Get cost analysis
  async getCostAnalysis(userId: string) {
    const response = await apiClient.get<{
      local_queries: number;
      cloud_queries: number;
      total_tokens: number;
      estimated_cost: number;
      savings: number;
    }>(`/api/ml/langchain/cost-analysis/${userId}`);

    return response.data;
  },
};

export default langchainApi;
