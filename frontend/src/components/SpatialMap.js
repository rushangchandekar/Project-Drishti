import React from 'react';
import { motion } from 'framer-motion';
import {
    Flame, Users, Activity, Zap, Eye, AlertTriangle, Camera, RefreshCw
} from 'lucide-react';

function SpatialMap({ status, getDensityColor, isHeatmapMain, videoKey, videoError, setVideoError, onRefresh }) {
    return (
        <div className="spatial-map">
            {/* Video Feed */}
            <div className="video-background">
                {!videoError ? (
                    <img
                        key={videoKey}
                        src={`http://localhost:8000/video-feed?t=${videoKey}`}
                        alt="Live Feed"
                        className="video-feed"
                        onError={() => setVideoError(true)}
                        onLoad={() => setVideoError(false)}
                    />
                ) : (
                    <div className="video-error">
                        <Camera size={48} />
                        <p>Video feed unavailable</p>
                        <button onClick={onRefresh} className="btn-primary">
                            <RefreshCw size={16} />
                            Retry
                        </button>
                    </div>
                )}

                <div className="video-overlay-gradient" />
            </div>

            {/* Incident Markers */}
            {status.fire_detected && (
                <motion.div
                    className="incident-marker fire"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    style={{ left: '70%', top: '30%' }}
                >
                    <Flame size={24} />
                    <div className="pulse-ring" />
                </motion.div>
            )}

            {status.anomaly_detected && (
                <motion.div
                    className="incident-marker anomaly"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    style={{ left: '35%', top: '45%' }}
                >
                    <AlertTriangle size={20} />
                    <div className="pulse-ring" />
                </motion.div>
            )}

            {/* Stats Overlay */}
            <div className="map-stats">
                <div className="stat-item">
                    <Users size={18} />
                    <div className="stat-content">
                        <span className="stat-value">{status.person_count}</span>
                        <span className="stat-label">People</span>
                    </div>
                </div>
                <div className="stat-item" style={{ borderColor: getDensityColor(status.density_level) }}>
                    <Activity size={18} style={{ color: getDensityColor(status.density_level) }} />
                    <div className="stat-content">
                        <span className="stat-value" style={{ color: getDensityColor(status.density_level) }}>
                            {status.density_level}
                        </span>
                        <span className="stat-label">Density</span>
                    </div>
                </div>
                <div className="stat-item">
                    <Zap size={18} />
                    <div className="stat-content">
                        <span className="stat-value">{status.risk_score}</span>
                        <span className="stat-label">Risk</span>
                    </div>
                </div>
            </div>

            {/* View Mode Badge */}
            <div className="view-mode-badge">
                <Eye size={14} />
                <span>{isHeatmapMain ? 'Heatmap View' : 'Standard View'}</span>
            </div>

            <style>{`
        .spatial-map {
          position: relative;
          width: 100%;
          height: 100%;
          overflow: hidden;
          background: #0a0a0a;
        }

        .video-background {
          position: absolute;
          inset: 0;
        }

        .video-feed {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }

        .video-error {
          width: 100%;
          height: 100%;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 16px;
          color: var(--text-muted);
        }

        .video-error button {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .heatmap-overlay {
          position: absolute;
          inset: 0;
          pointer-events: none;
          mix-blend-mode: screen;
          opacity: 0.7;
        }

        .heatmap-svg {
          width: 100%;
          height: 100%;
        }

        .video-overlay-gradient {
          position: absolute;
          inset: 0;
          background: linear-gradient(
            180deg,
            rgba(11, 14, 20, 0.3) 0%,
            rgba(11, 14, 20, 0.05) 50%,
            rgba(11, 14, 20, 0.4) 100%
          );
          pointer-events: none;
        }

        .zone-label {
          position: absolute;
          transform: translate(-50%, -50%);
          background: rgba(11, 14, 20, 0.85);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          padding: 12px 18px;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 4px;
          min-width: 80px;
          pointer-events: none;
        }

        .zone-name {
          font-size: 11px;
          color: var(--text-muted);
          font-weight: 500;
        }

        .zone-count {
          font-size: 20px;
          font-weight: 700;
        }

        .incident-marker {
          position: absolute;
          transform: translate(-50%, -50%);
          z-index: 20;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .incident-marker.fire {
          color: var(--status-critical);
          filter: drop-shadow(0 0 20px rgba(248, 81, 73, 0.6));
        }

        .incident-marker.anomaly {
          color: var(--status-warning);
          filter: drop-shadow(0 0 20px rgba(210, 153, 34, 0.6));
        }

        .pulse-ring {
          position: absolute;
          width: 60px;
          height: 60px;
          border-radius: 50%;
          border: 2px solid currentColor;
          animation: pulse-ring 2s ease-out infinite;
        }

        @keyframes pulse-ring {
          0% { transform: scale(0.5); opacity: 1; }
          100% { transform: scale(1.5); opacity: 0; }
        }

        .map-stats {
          position: absolute;
          top: 16px;
          left: 16px;
          display: flex;
          gap: 12px;
          z-index: 10;
        }

        .stat-item {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 10px 16px;
          background: rgba(11, 14, 20, 0.85);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 10px;
        }

        .stat-content {
          display: flex;
          flex-direction: column;
        }

        .stat-value {
          font-size: 16px;
          font-weight: 700;
        }

        .stat-label {
          font-size: 10px;
          color: var(--text-muted);
        }

        .view-mode-badge {
          position: absolute;
          bottom: 16px;
          right: 16px;
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 8px 14px;
          background: rgba(11, 14, 20, 0.85);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 8px;
          font-size: 11px;
          color: var(--text-secondary);
          z-index: 10;
        }
      `}</style>
        </div>
    );
}

export default SpatialMap;
