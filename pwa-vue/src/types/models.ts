// ML Model Types

export interface PytorchPrediction {
  predicted_time: number;
  predicted_pace: number;
  confidence: number;
}

export interface WorkoutHistory {
  distance_km: number;
  duration_min: number;
  pace_min_per_km: number;
  avg_hr?: number;
  max_hr?: number;
  elevation_m?: number;
  temperature_c?: number;
  humidity?: number;
  cadence?: number;
  day_of_week?: number;
}

export interface ChallengeRecommendation {
  challenge_id: number;
  score: number;
  rank: number;
  title?: string;
  distance?: number;
  location?: string;
  participants?: number;
}

export interface VectorSearchResult {
  text: string;
  similarity: number;
  metadata: Record<string, any>;
  type?: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  tools_used?: string[];
  cost_estimate?: number;
}

export interface ModelMetrics {
  accuracy?: number;
  loss?: number;
  auc?: number;
  tokens_generated?: number;
  duration_ms?: number;
  cost?: number;
}

export interface MLModelStatus {
  model_name: string;
  is_loaded: boolean;
  is_training: boolean;
  last_updated: Date;
  metrics: ModelMetrics;
}
