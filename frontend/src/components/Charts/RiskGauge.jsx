import React from 'react';
import { AlertTriangle } from 'lucide-react';
import { getRiskColor } from '../../utils/helpers';

const RiskGauge = ({ riskScore }) => {
  const score = riskScore || 0;
  const percentage = Math.min(100, Math.max(0, score));
  const color = getRiskColor(score);

  const getRiskLevel = () => {
    if (score >= 80) return 'CRITICAL';
    if (score >= 60) return 'HIGH';
    if (score >= 40) return 'MODERATE';
    if (score >= 20) return 'LOW';
    return 'MINIMAL';
  };

  return (
    <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <AlertTriangle className="w-5 h-5 text-yellow-400" />
          <h3 className="text-lg font-semibold text-white">Risk Assessment</h3>
        </div>
      </div>

      {/* Circular Gauge */}
      <div className="flex flex-col items-center">
        <div className="relative w-48 h-48">
          {/* Background Circle */}
          <svg className="w-full h-full transform -rotate-90">
            <circle
              cx="96"
              cy="96"
              r="80"
              stroke="#334155"
              strokeWidth="16"
              fill="none"
            />
            {/* Progress Circle */}
            <circle
              cx="96"
              cy="96"
              r="80"
              stroke={color}
              strokeWidth="16"
              fill="none"
              strokeDasharray={`${percentage * 5.03} 503`}
              strokeLinecap="round"
              className="transition-all duration-500"
            />
          </svg>
          {/* Center Text */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <p className="text-5xl font-bold text-white">{score}</p>
            <p className="text-sm text-slate-400">/ 100</p>
          </div>
        </div>

        {/* Risk Level */}
        <div className="mt-6 text-center">
          <div
            className="inline-block px-6 py-2 rounded-full text-white font-bold text-lg"
            style={{ backgroundColor: color }}
          >
            {getRiskLevel()}
          </div>
          <p className="text-sm text-slate-400 mt-2">Risk Level</p>
        </div>
      </div>
    </div>
  );
};

export default RiskGauge;