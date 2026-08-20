'use client';
import { motion } from 'framer-motion';

const STATS = [
  { value: '9', label: 'AI Agents' },
  { value: '<100ms', label: 'Detection Latency' },
  { value: '24/7', label: 'Autonomous Monitoring' },
];

export default function StatsSection() {
  return (
    <section className="lp-stats">
      <div className="lp-container">
        <div className="lp-stats-grid">
          {STATS.map((stat, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: i * 0.1 }}
              viewport={{ once: true }}
              style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}
            >
              <span className="lp-stat-value">{stat.value}</span>
              <span className="lp-stat-label">{stat.label}</span>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
