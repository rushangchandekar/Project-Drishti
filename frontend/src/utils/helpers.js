// Get color based on density level
export const getDensityColor = (level) => {
  const colors = {
    EMPTY: '#6b7280',
    VERY_LOW: '#10b981',
    LOW: '#10b981',
    MODERATE: '#fbbf24',
    HIGH: '#f59e0b',
    VERY_HIGH: '#ef4444',
    CRITICAL: '#dc2626',
  };
  return colors[level] || '#6b7280';
};

// Get color based on risk score
export const getRiskColor = (score) => {
  if (score >= 80) return '#dc2626';
  if (score >= 60) return '#ef4444';
  if (score >= 40) return '#f59e0b';
  if (score >= 20) return '#fbbf24';
  return '#10b981';
};

// Format timestamp
export const formatTimestamp = (timestamp) => {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString();
};

// Format duration
export const formatDuration = (ms) => {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
};

// Get status badge color
export const getStatusBadgeColor = (status) => {
  const colors = {
    EXECUTED: 'bg-green-500',
    ACTIVE: 'bg-green-500',
    IN_PROGRESS: 'bg-blue-500',
    STANDBY: 'bg-yellow-500',
    FAILED: 'bg-red-500',
    IDLE: 'bg-gray-500',
  };
  return colors[status] || 'bg-gray-500';
};

// Calculate percentage
export const calculatePercentage = (value, max) => {
  return Math.min(100, Math.max(0, (value / max) * 100));
};

// Truncate text
export const truncate = (text, maxLength) => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
};