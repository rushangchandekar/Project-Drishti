import React from 'react';
import { Activity, AlertCircle, Shield, Users } from 'lucide-react';

const Header = ({ systemStatus }) => {
  const { person_count, density_level, risk_score, fire_detected } = systemStatus;

  return (
    <header className="bg-slate-900 border-b border-slate-700 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Logo and Title */}
        <div className="flex items-center space-x-4">
          <div className="bg-blue-600 p-2 rounded-lg">
            <Shield className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">Project Drishti</h1>
            <p className="text-sm text-slate-400">AI-Powered Crowd Safety System</p>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="flex items-center space-x-6">
          {/* People Count */}
          <div className="flex items-center space-x-2">
            <Users className="w-5 h-5 text-blue-400" />
            <div>
              <p className="text-xs text-slate-400">People</p>
              <p className="text-lg font-bold text-white">{person_count || 0}</p>
            </div>
          </div>

          {/* Density */}
          <div className="flex items-center space-x-2">
            <Activity className="w-5 h-5 text-yellow-400" />
            <div>
              <p className="text-xs text-slate-400">Density</p>
              <p className="text-lg font-bold text-white">{density_level || 'N/A'}</p>
            </div>
          </div>

          {/* Risk Score */}
          <div className="flex items-center space-x-2">
            <AlertCircle className={`w-5 h-5 ${risk_score > 70 ? 'text-red-400' : 'text-green-400'}`} />
            <div>
              <p className="text-xs text-slate-400">Risk</p>
              <p className="text-lg font-bold text-white">{risk_score || 0}/100</p>
            </div>
          </div>

          {/* Fire Status */}
          {fire_detected && (
            <div className="flex items-center space-x-2 bg-red-500 px-4 py-2 rounded-lg animate-pulse">
              <AlertCircle className="w-5 h-5 text-white" />
              <p className="text-sm font-bold text-white">FIRE DETECTED</p>
            </div>
          )}

          {/* System Status */}
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <p className="text-sm text-slate-300">System Online</p>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;