import React, { useState, useEffect } from 'react';
import { Shield, ChevronDown, ChevronUp, Activity, CheckCircle, XCircle } from 'lucide-react';
import { getStatusBadgeColor } from '../../utils/helpers';

const AgentCard = ({ agent, isExpanded, onToggle }) => {
  const { agent_name, status, enabled, total_decisions, total_actions, success_rate } = agent;

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 overflow-hidden">
      {/* Header */}
      <div
        className="px-4 py-3 flex items-center justify-between cursor-pointer hover:bg-slate-750 transition-colors"
        onClick={onToggle}
      >
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg ${enabled ? 'bg-green-500/20' : 'bg-gray-500/20'}`}>
            <Shield className={`w-5 h-5 ${enabled ? 'text-green-400' : 'text-gray-400'}`} />
          </div>
          <div>
            <h3 className="text-white font-semibold">{agent_name}</h3>
            <div className="flex items-center space-x-2 mt-1">
              <span className={`px-2 py-0.5 text-xs rounded ${getStatusBadgeColor(status)} text-white`}>
                {status}
              </span>
              {enabled ? (
                <span className="text-xs text-green-400 flex items-center space-x-1">
                  <CheckCircle className="w-3 h-3" />
                  <span>Enabled</span>
                </span>
              ) : (
                <span className="text-xs text-gray-400 flex items-center space-x-1">
                  <XCircle className="w-3 h-3" />
                  <span>Disabled</span>
                </span>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-right">
            <p className="text-sm text-slate-400">Success Rate</p>
            <p className="text-lg font-bold text-white">{(success_rate * 100).toFixed(0)}%</p>
          </div>
          {isExpanded ? (
            <ChevronUp className="w-5 h-5 text-slate-400" />
          ) : (
            <ChevronDown className="w-5 h-5 text-slate-400" />
          )}
        </div>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="px-4 py-3 border-t border-slate-700 bg-slate-900/50">
          <div className="grid grid-cols-3 gap-4">
            <div>
              <p className="text-xs text-slate-400">Total Decisions</p>
              <p className="text-2xl font-bold text-white">{total_decisions}</p>
            </div>
            <div>
              <p className="text-xs text-slate-400">Total Actions</p>
              <p className="text-2xl font-bold text-white">{total_actions}</p>
            </div>
            <div>
              <p className="text-xs text-slate-400">Success Count</p>
              <p className="text-2xl font-bold text-white">{Math.floor(total_decisions * success_rate)}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const AgentStatus = ({ agentStats }) => {
  const [expandedAgent, setExpandedAgent] = useState(null);

  if (!agentStats || !agentStats.agent_stats) {
    return (
      <div className="bg-slate-800 rounded-lg p-8 text-center">
        <Activity className="w-12 h-12 text-slate-600 mx-auto mb-4" />
        <p className="text-slate-400">Loading agent statistics...</p>
      </div>
    );
  }

  const agents = Object.values(agentStats.agent_stats);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Shield className="w-6 h-6 text-blue-400" />
          <h2 className="text-xl font-bold text-white">Agent System Status</h2>
        </div>
        <div className="text-sm text-slate-400">
          {agentStats.enabled_agents}/{agentStats.total_agents} Active
        </div>
      </div>

      {/* Agent Cards */}
      <div className="space-y-3">
        {agents.map((agent, index) => (
          <AgentCard
            key={index}
            agent={agent}
            isExpanded={expandedAgent === index}
            onToggle={() => setExpandedAgent(expandedAgent === index ? null : index)}
          />
        ))}
      </div>
    </div>
  );
};

export default AgentStatus;