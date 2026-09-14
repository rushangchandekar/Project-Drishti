// Central API base configuration
export const API_BASE = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000').replace(/\/+$/, '');

// Dynamic WebSocket URL deriving ws:// or wss:// from API_BASE
export const getWsUrl = (path = '/ws/telemetry') => {
  const normalizedPath = path.startsWith('/') ? path : '/' + path;
  if (API_BASE.startsWith('https://')) {
    return API_BASE.replace(/^https:\/\//, 'wss://') + normalizedPath;
  }
  if (API_BASE.startsWith('http://')) {
    return API_BASE.replace(/^http:\/\//, 'ws://') + normalizedPath;
  }
  const protocol = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = typeof window !== 'undefined' ? window.location.host : 'localhost:8000';
  return protocol + '//' + host + normalizedPath;
};
