import axios, { type AxiosInstance, type AxiosError } from 'axios';
import type { ApiResponse, ApiError } from '../types/api';

// Create Axios instance with default config
const api: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8080',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor - Add auth token to requests
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('firebase_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor - Handle errors globally
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error: AxiosError) => {
    const apiError: ApiError = {
      message: error.message || 'An error occurred',
      status: error.response?.status || 500,
      details: error.response?.data,
    };

    // Handle specific error codes
    if (error.response?.status === 401) {
      // Unauthorized - clear token and redirect to login
      localStorage.removeItem('firebase_token');
      window.location.href = '/login';
    }

    return Promise.reject(apiError);
  }
);

// Generic API methods
export const apiClient = {
  get: <T>(url: string, config = {}): Promise<ApiResponse<T>> =>
    api.get(url, config).then(res => res.data),

  post: <T>(url: string, data?: any, config = {}): Promise<ApiResponse<T>> =>
    api.post(url, data, config).then(res => res.data),

  put: <T>(url: string, data?: any, config = {}): Promise<ApiResponse<T>> =>
    api.put(url, data, config).then(res => res.data),

  delete: <T>(url: string, config = {}): Promise<ApiResponse<T>> =>
    api.delete(url, config).then(res => res.data),
};

export default api;
