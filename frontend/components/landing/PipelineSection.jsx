'use client';
import { motion } from 'framer-motion';

const STEPS = [
  {
    id: '1',
    title: 'Video Stream Ingestion & Multi-Camera Support',
    image: 'https://images.unsplash.com/photo-1592478411213-6153e4ebc07d?q=80&w=300&auto=format&fit=crop',
    badge: 'Step 1',
    tag: 'Input Layer',
  },
  {
    id: '2',
    title: 'YOLOv11 Object Detection & Crowd Analysis',
    image: 'https://images.unsplash.com/photo-1507238691740-187a5b1d37b8?q=80&w=300&auto=format&fit=crop',
    badge: 'Step 2',
    tag: 'Detection',
  },
  {
    id: '3',
    title: 'Intelligent Decision Engine & Risk Scoring',
    image: 'https://images.unsplash.com/photo-1555066931-4365d14bab8c?q=80&w=300&auto=format&fit=crop',
    badge: 'Step 3',
    tag: 'Intelligence',
  },
  {
    id: '4',
    title: 'Autonomous Response, Alerts & Agent Dispatch',
    image: 'https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=300&auto=format&fit=crop',
    badge: 'Step 4',
    tag: 'Response',
  },
];

export default function PipelineSection() {
  return (
    <section className="lp-pipeline">
      <div className="lp-container">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: [0.25, 0.1, 0.25, 1] }}
          viewport={{ once: true, margin: '-100px' }}
          className="lp-capabilities-header"
        >
          <div>
            <div className="lp-section-tag">
              <div className="lp-section-tag-line" />
              <span className="lp-section-tag-text">How It Works</span>
            </div>
            <h2 className="lp-section-title">
              System <span className="lp-section-title-italic">pipeline</span>
            </h2>
            <p className="lp-section-subtitle">
              From raw video streams to autonomous safety responses in milliseconds.
            </p>
          </div>

          <a href="/dashboard" className="lp-view-all-btn">
            <span className="accent-gradient-border" style={{ position: 'absolute', inset: 0, borderRadius: '9999px', opacity: 0 }} />
            <span>See it live</span>
            <span className="arrow">→</span>
          </a>
        </motion.div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {STEPS.map((step) => (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="lp-pipeline-entry"
            >
              <div className="lp-pipeline-entry-left">
                <div className="lp-pipeline-entry-img">
                  <img src={step.image} alt={step.title} />
                </div>
                <h3 className="lp-pipeline-entry-title">{step.title}</h3>
              </div>

              <div className="lp-pipeline-entry-meta">
                <span className="lp-pipeline-entry-badge">{step.badge}</span>
                <span className="lp-pipeline-entry-tag">{step.tag}</span>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
