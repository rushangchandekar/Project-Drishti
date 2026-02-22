import React from 'react';
import { motion } from 'framer-motion';
import { Camera, RotateCcw } from 'lucide-react';

function VideoPreview({ isHeatmapMain, onSwap, videoKey }) {
    return (
        <motion.div
            className="video-preview"
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            onClick={onSwap}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.98 }}
        >
            <div className="preview-header">
                <Camera size={12} />
                <span>{isHeatmapMain ? 'Raw Feed' : 'Heatmap'}</span>
                <div className="swap-hint">
                    <RotateCcw size={10} />
                    Swap
                </div>
            </div>
            <div className="preview-video">
                <img
                    key={videoKey}
                    src={`http://localhost:8000/video-feed?t=${videoKey}`}
                    alt="Preview"
                    onError={(e) => {
                        e.target.style.display = 'none';
                        e.target.nextSibling.style.display = 'flex';
                    }}
                />
                <div className="preview-placeholder" style={{ display: 'none' }}>
                    <Camera size={20} />
                </div>
                <div className="preview-overlay">
                    <RotateCcw size={24} />
                </div>
            </div>

            <style>{`
        .video-preview {
          position: fixed;
          bottom: 120px;
          left: 90px;
          width: 220px;
          background: var(--bg-secondary);
          border: 1px solid var(--border-primary);
          border-radius: 12px;
          overflow: hidden;
          cursor: pointer;
          z-index: 100;
          transition: all 0.2s ease;
        }

        .video-preview:hover {
          border-color: var(--accent-blue);
          box-shadow: 0 0 20px rgba(88, 166, 255, 0.2);
        }

        .preview-header {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 8px 12px;
          font-size: 11px;
          font-weight: 600;
          border-bottom: 1px solid var(--border-primary);
          color: var(--text-secondary);
        }

        .swap-hint {
          margin-left: auto;
          display: flex;
          align-items: center;
          gap: 4px;
          font-size: 9px;
          color: var(--accent-blue);
        }

        .preview-video {
          position: relative;
          height: 120px;
          background: #000;
        }

        .preview-video img {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }

        .preview-placeholder {
          width: 100%;
          height: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--text-muted);
        }

        .preview-overlay {
          position: absolute;
          inset: 0;
          background: rgba(0, 0, 0, 0.6);
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          opacity: 0;
          transition: opacity 0.2s ease;
        }

        .video-preview:hover .preview-overlay {
          opacity: 1;
        }
      `}</style>
        </motion.div>
    );
}

export default VideoPreview;
