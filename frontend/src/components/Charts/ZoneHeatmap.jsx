import React from 'react';
import { Grid } from 'lucide-react';

const ZoneHeatmap = ({ zones }) => {
  if (!zones) {
    zones = {
      top_left: 0, top_center: 0, top_right: 0,
      mid_left: 0, mid_center: 0, mid_right: 0,
      bot_left: 0, bot_center: 0, bot_right: 0,
    };
  }

  const total = Object.values(zones).reduce((sum, val) => sum + val, 0);

  const getZoneColor = (count) => {
    if (total === 0) return '#1e293b';
    const percentage = (count / total) * 100;
    if (percentage > 40) return '#dc2626';
    if (percentage > 25) return '#f59e0b';
    if (percentage > 15) return '#fbbf24';
    if (percentage > 5) return '#3b82f6';
    return '#1e293b';
  };

  const zoneLayout = [
    ['top_left', 'top_center', 'top_right'],
    ['mid_left', 'mid_center', 'mid_right'],
    ['bot_left', 'bot_center', 'bot_right'],
  ];

  return (
    <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Grid className="w-5 h-5 text-blue-400" />
          <h3 className="text-lg font-semibold text-white">Zone Distribution</h3>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2">
        {zoneLayout.map((row, rowIndex) =>
          row.map((zoneName, colIndex) => {
            const count = zones[zoneName] || 0;
            const percentage = total > 0 ? ((count / total) * 100).toFixed(0) : 0;
            const color = getZoneColor(count);

            return (
              <div
                key={zoneName}
                className="aspect-square rounded-lg p-3 flex flex-col items-center justify-center border border-slate-600 transition-all hover:scale-105"
                style={{ backgroundColor: color }}
              >
                <p className="text-3xl font-bold text-white">{count}</p>
                <p className="text-xs text-white/70 mt-1">{percentage}%</p>
                <p className="text-xs text-white/50 mt-1">
                  {zoneName.replace('_', ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                </p>
              </div>
            );
          })
        )}
      </div>

      {/* Legend */}
      <div className="mt-4 flex items-center justify-center space-x-4 text-xs">
        <div className="flex items-center space-x-1">
          <div className="w-3 h-3 rounded bg-slate-800"></div>
          <span className="text-slate-400">0-5%</span>
        </div>
        <div className="flex items-center space-x-1">
          <div className="w-3 h-3 rounded bg-blue-600"></div>
          <span className="text-slate-400">5-15%</span>
        </div>
        <div className="flex items-center space-x-1">
          <div className="w-3 h-3 rounded bg-yellow-500"></div>
          <span className="text-slate-400">15-25%</span>
        </div>
        <div className="flex items-center space-x-1">
          <div className="w-3 h-3 rounded bg-orange-500"></div>
          <span className="text-slate-400">25-40%</span>
        </div>
        <div className="flex items-center space-x-1">
          <div className="w-3 h-3 rounded bg-red-600"></div>
          <span className="text-slate-400">&gt;40%</span>
        </div>
      </div>
    </div>
  );
};

export default ZoneHeatmap;