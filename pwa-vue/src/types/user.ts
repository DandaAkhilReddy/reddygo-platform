// User and Authentication Types

export interface User {
  id: string;
  email: string;
  name: string;
  created_at: Date;
  stats?: UserStats;
  preferences?: UserPreferences;
}

export interface UserStats {
  challenges_completed: number;
  total_distance_km: number;
  total_workouts: number;
  points_balance: number;
  current_streak: number;
}

export interface UserPreferences {
  preferred_distance?: string;
  preferred_time?: string;
  fitness_goals?: string[];
  injuries?: string[];
  training_zones?: {
    zone1: number[];
    zone2: number[];
    zone3: number[];
    zone4: number[];
    zone5: number[];
  };
}

export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  token: string | null;
}
