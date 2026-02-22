import React from 'react';
import { motion } from 'framer-motion';
import { Camera, Webcam, Monitor } from 'lucide-react';

function CameraSwitcher({ selectedCamera, onCameraSelect }) {
    const cameras = [
        { id: 'webcam', name: 'Main Camera', icon: Webcam, status: 'active' },
        { id: 'cam2', name: 'Entrance Gate', icon: Monitor, status: 'inactive' },
        { id: 'cam3', name: 'Exit Zone A', icon: Monitor, status: 'inactive' },
        { id: 'cam4', name: 'VIP Section', icon: Monitor, status: 'inactive' },
    ];

    return (
        <div className="camera-switcher">
            <div className="switcher-header">
                <Camera size={16} />
                <span>Camera Sources</span>
                <span className="camera-count">1 Active</span>
            </div>

            <div className="camera-list">
                {cameras.map(camera => {
                    const Icon = camera.icon;
                    return (
                        <motion.button
                            key={camera.id}
                            className={`camera-card ${selectedCamera === camera.id ? 'active' : ''} ${camera.status}`}
                            onClick={() => onCameraSelect(camera.id)}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                        >
                            <div className="camera-icon">
                                <Icon size={20} />
                            </div>
                            <div className="camera-info">
                                <span className="camera-name">{camera.name}</span>
                                <span className="camera-type">{camera.id === 'webcam' ? 'USB Camera' : 'IP Stream'}</span>
                            </div>
                            <div className="camera-status">
                                <div className={`status-indicator ${camera.status}`} />
                                <span>{camera.status === 'active' ? 'Live' : 'Offline'}</span>
                            </div>
                            {selectedCamera === camera.id && (
                                <motion.div className="selected-indicator" layoutId="selectedCamera" />
                            )}
                        </motion.button>
                    );
                })}
            </div>

            <style>{`
        .camera-switcher {
          height: 100px;
          background: var(--bg-secondary);
          border-top: 1px solid var(--border-primary);
          padding: 12px 24px;
        }

        .switcher-header {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 12px;
          font-size: 13px;
          font-weight: 600;
          color: var(--text-secondary);
        }

        .camera-count {
          margin-left: auto;
          font-size: 11px;
          color: var(--status-safe);
          background: rgba(63, 185, 80, 0.15);
          padding: 4px 10px;
          border-radius: 12px;
        }

        .camera-list {
          display: flex;
          gap: 12px;
          overflow-x: auto;
        }

        .camera-card {
          position: relative;
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 12px 16px;
          background: var(--bg-tertiary);
          border: 1px solid var(--border-primary);
          border-radius: 10px;
          cursor: pointer;
          transition: all 0.2s ease;
          min-width: 200px;
        }

        .camera-card:hover {
          background: var(--bg-hover);
          border-color: var(--accent-blue);
        }

        .camera-card.active {
          border-color: var(--accent-blue);
          background: rgba(88, 166, 255, 0.1);
        }

        .camera-card.inactive {
          opacity: 0.5;
        }

        .camera-icon {
          width: 40px;
          height: 40px;
          border-radius: 10px;
          background: var(--bg-secondary);
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--text-secondary);
        }

        .camera-card.active .camera-icon {
          background: var(--accent-blue);
          color: white;
        }

        .camera-info {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .camera-name {
          font-size: 13px;
          font-weight: 600;
          color: var(--text-primary);
        }

        .camera-type {
          font-size: 10px;
          color: var(--text-muted);
        }

        .camera-status {
          margin-left: auto;
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 10px;
          color: var(--text-muted);
        }

        .status-indicator {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }

        .status-indicator.active {
          background: var(--status-safe);
          box-shadow: 0 0 8px rgba(63, 185, 80, 0.6);
        }

        .status-indicator.inactive {
          background: var(--text-muted);
        }

        .selected-indicator {
          position: absolute;
          bottom: -1px;
          left: 20%;
          right: 20%;
          height: 3px;
          background: var(--accent-blue);
          border-radius: 3px 3px 0 0;
        }
      `}</style>
        </div>
    );
}

export default CameraSwitcher;
