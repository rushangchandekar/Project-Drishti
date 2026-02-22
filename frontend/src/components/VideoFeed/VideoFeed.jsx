import React, { useState, useEffect } from 'react';
import { Camera, AlertCircle, Play, Pause } from 'lucide-react';
import apiService from '../../services/api';

const VideoFeed = () => {
  const [isPlaying, setIsPlaying] = useState(true);
  const [videoError, setVideoError] = useState(false);
  const videoUrl = apiService.getVideoFeedUrl();

  const handleError = () => {
    setVideoError(true);
  };

  const handleLoad = () => {
    setVideoError(false);
  };

  return (
    <div className="bg-slate-800 rounded-lg shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-slate-900 px-4 py-3 flex items-center justify-between border-b border-slate-700">
        <div className="flex items-center space-x-2">
          <Camera className="w-5 h-5 text-blue-400" />
          <h2 className="text-lg font-semibold text-white">Live Video Feed</h2>
        </div>
        <div className="flex items-center space-x-2">
          {isPlaying ? (
            <div className="flex items-center space-x-2 text-green-400">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm">LIVE</span>
            </div>
          ) : (
            <div className="flex items-center space-x-2 text-gray-400">
              <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
              <span className="text-sm">PAUSED</span>
            </div>
          )}
        </div>
      </div>

      {/* Video Container */}
      <div className="relative bg-black aspect-video">
        {videoError ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <AlertCircle className="w-16 h-16 text-red-400 mb-4" />
            <p className="text-white text-lg">Video feed unavailable</p>
            <p className="text-slate-400 text-sm mt-2">
              Make sure the backend is running
            </p>
          </div>
        ) : (
          <img
            src={videoUrl}
            alt="Live Feed"
            className="w-full h-full object-contain"
            onError={handleError}
            onLoad={handleLoad}
          />
        )}
      </div>

      {/* Controls */}
      <div className="bg-slate-900 px-4 py-2 flex items-center justify-between border-t border-slate-700">
        <div className="flex items-center space-x-2 text-slate-400 text-sm">
          <Camera className="w-4 h-4" />
          <span>Camera 1 - Main Entrance</span>
        </div>
      </div>
    </div>
  );
};

export default VideoFeed;