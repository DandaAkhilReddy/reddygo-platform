// API Request/Response Types

export interface ApiResponse<T = any> {
  data: T;
  status: number;
  message?: string;
}

export interface ApiError {
  message: string;
  status: number;
  details?: any;
}

export interface PytorchPredictRequest {
  workout_history: number[][];  // 2D array: [seq_len, features]
  user_id: string;
}

export interface TensorflowRecommendRequest {
  user_id: number;
  candidate_challenges: number[];
  top_k?: number;
}

export interface QdrantSearchRequest {
  query: string;
  collection: 'workouts' | 'memories' | 'exercises';
  user_id?: string;
  limit?: number;
}

export interface LangchainChatRequest {
  user_id: string;
  query: string;
  context?: string;
  use_local_llm?: boolean;
}

export interface LangchainChatResponse {
  response: string;
  tools_used: string[];
  cost_estimate: number;
  model_used: string;
}
