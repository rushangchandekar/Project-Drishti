import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API functions
export const apiService = {
  // Get current status
  getStatus: async () => {
    const response = await api.get('/status');
    return response.data;
  },

  // Get detailed state
  getDetailedState: async () => {
    const response = await api.get('/detailed-state');
    return response.data;
  },

  // Get n8n status
  getN8nStatus: async () => {
    const response = await api.get('/n8n-status');
    return response.data;
  },

  // Get intelligence stats
  getIntelligenceStats: async () => {
    const response = await api.get('/intelligence-stats');
    return response.data;
  },

  // Get performance metrics
  getPerformance: async () => {
    const response = await api.get('/performance');
    return response.data;
  },

  // Ask chatbot question
  askQuestion: async (question) => {
    const response = await api.post('/query', { question });
    return response.data;
  },

  // Get AI summary
  getSummary: async () => {
    const response = await api.get('/summary');
    return response.data;
  },

  // Get video feed URL
  getVideoFeedUrl: () => {
    return `${API_BASE_URL}/video-feed`;
  },

  // List videos
  listVideos: async () => {
    const response = await api.get('/list-videos');
    return response.data;
  },

  // Switch video source
  switchSource: async (type, path = null) => {
    const response = await api.post('/switch-source', { type, path });
    return response.data;
  },
};

export default apiService;