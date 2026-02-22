import React from 'react';
import { Users, Activity, AlertTriangle, TrendingUp, Eye, Brain } from 'lucide-react';
import { getDensityColor, getRiskColor } from '../../utils/helpers';

const MetricCard = ({ icon: Icon, label, value, subValue, color, trend }) => (
  <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 hover:border-slate-600 transition-colors">
    <div className="flex items-center justify-between mb-2">
      <div className="flex items-center space-x-2">
        <div className={`p-2 rounded-lg`} style={{ backgroundColor: color + '20' }}>
          <Icon className="w-5 h-5" style={{ color }} />
        </div>
        <span className="text-sm text-slate-400">{label}</span>
      </div>
      {trend && (
        <div className={`flex items-center space-x-1 text-xs ${
          trend === 'INCREASING' ? 'text-red-400' : 
          trend === 'DECREASING' ? 'text-green-400' : 
          'text-slate-400'
        }`}>
          <TrendingUp className={`w-4 h-4 ${trend === 'DECREASING' ? 'rotate-180' : ''}`} />
          <span>{trend}</span>
        </div>
      )}
    </div>
    <div className="mt-2">
      <p className="text-3xl font-bold text-white">{value}</p>
      {subValue && <p className="text-sm text-slate-400 mt-1">{subValue}</p>}
    </div>
  </div>
);

const MetricsPanel = ({ status }) => {
  const {
    person_count = 0,
    density_level = 'UNKNOWN',
    density_value = 0,
    risk_score = 0,
    trend = 'STABLE',
    anomaly_detected = false,
    anomaly_type = null,
    agents_activated = 0,
    actions_executed = 0,
  } = status;

  const densityColor = getDensityColor(density_level);
  const riskColor = getRiskColor(risk_score);

  return (
    <div className="space-y-4">
      {/* Main Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <MetricCard
          icon={Users}
          label="Crowd Count"
          value={person_count}
          subValue="People detected"
          color="#3b82f6"
          trend={trend}
        />
        
        <MetricCard
          icon={Activity}
          label="Density Level"
          value={density_level}
          subValue={`${density_value.toFixed(2)} people/m²`}
          color={densityColor}
        />
        
        <MetricCard
          icon={AlertTriangle}
          label="Risk Score"
          value={`${risk_score}/100`}
          subValue={risk_score > 70 ? 'High Risk' : risk_score > 40 ? 'Moderate Risk' : 'Low Risk'}
          color={riskColor}
        />
      </div>

      {/* Secondary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          icon={Eye}
          label="Anomaly Status"
          value={anomaly_detected ? 'DETECTED' : 'CLEAR'}
          subValue={anomaly_type || 'No anomalies'}
          color={anomaly_detected ? '#ef4444' : '#10b981'}
        />
        
        <MetricCard
          icon={Brain}
          label="Agents Active"
          value={agents_activated}
          subValue="AI agents running"
          color="#8b5cf6"
        />
        
        <MetricCard
          icon={AlertTriangle}
          label="Actions Executed"
          value={actions_executed}
          subValue="Automated responses"
          color="#f59e0b"
        />
      </div>
    </div>
  );
};

export default MetricsPanel;