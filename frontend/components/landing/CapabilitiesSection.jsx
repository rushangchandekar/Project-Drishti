'use client';
import { motion } from 'framer-motion';

const CAPABILITIES = [
  {
    id: '1',
    title: 'Real-Time Crowd Detection',
    category: 'Computer Vision',
    image: 'https://images.unsplash.com/photo-1531058020387-3be344556be6?q=80&w=1200&auto=format&fit=crop',
    spanClass: 'lp-cap-span-7',
  },
  {
    id: '2',
    title: 'Anomaly Detection',
    category: 'AI Intelligence',
    image: 'https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=1200&auto=format&fit=crop',
    spanClass: 'lp-cap-span-5',
  },
  {
    id: '3',
    title: 'Fire & Hazard Detection',
    category: 'Safety Systems',
    image: 'https://images.unsplash.com/photo-1486551937199-baf066858de7?q=80&w=1200&auto=format&fit=crop',
    spanClass: 'lp-cap-span-5',
  },
  {
    id: '4',
    title: 'Multi-Agent Orchestration',
    category: '9 AI Agents',
    image: 'https://images.unsplash.com/photo-1558494949-ef010cbdcc31?q=80&w=1200&auto=format&fit=crop',
    spanClass: 'lp-cap-span-7',
  },
];

export default function CapabilitiesSection() {
  return (
    <section id="capabilities" className="lp-capabilities">
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
              <span className="lp-section-tag-text">Capabilities</span>
            </div>
            <h2 className="lp-section-title">
              Key <span className="lp-section-title-italic">capabilities</span>
            </h2>
            <p className="lp-section-subtitle">
              A comprehensive AI surveillance system with 9 specialized agents working in concert.
            </p>
          </div>

          <a href="/dashboard" className="lp-view-all-btn">
            <span className="accent-gradient-border" style={{ position: 'absolute', inset: 0, borderRadius: '9999px', opacity: 0 }} />
            <span>Open Dashboard</span>
            <span className="arrow">→</span>
          </a>
        </motion.div>

        <div className="lp-capabilities-grid">
          {CAPABILITIES.map((cap) => (
            <motion.div
              key={cap.id}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className={`lp-cap-card ${cap.spanClass}`}
            >
              <img src={cap.image} alt={cap.title} />
              <div className="lp-cap-card-overlay-dots" />
              <div className="lp-cap-card-hover">
                <div className="lp-cap-card-hover-pill">
                  <span style={{ fontWeight: 500 }}>
                    {cap.category} — <span className="lp-font-display">{cap.title}</span>
                  </span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
