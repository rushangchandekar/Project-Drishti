'use client';
import { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import ScrollTrigger from 'gsap/ScrollTrigger';

if (typeof window !== 'undefined') {
  gsap.registerPlugin(ScrollTrigger);
}

const ITEMS = [
  {
    id: '1',
    title: 'YOLOv11 Detection',
    image: 'https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=600&auto=format&fit=crop',
    rotationClass: 'lp-rotate-n6',
  },
  {
    id: '2',
    title: 'Gemini Vision AI',
    image: 'https://images.unsplash.com/photo-1634017839464-5c339ebe3cb4?q=80&w=600&auto=format&fit=crop',
    rotationClass: 'lp-rotate-3',
  },
  {
    id: '3',
    title: 'OpenCV Pipeline',
    image: 'https://images.unsplash.com/photo-1633167606207-d840b5070fc2?q=80&w=600&auto=format&fit=crop',
    rotationClass: 'lp-rotate-n3',
  },
  {
    id: '4',
    title: 'Agent Orchestrator',
    image: 'https://images.unsplash.com/photo-1558494949-ef010cbdcc31?q=80&w=600&auto=format&fit=crop',
    rotationClass: 'lp-rotate-6',
  },
  {
    id: '5',
    title: 'Decision Engine',
    image: 'https://images.unsplash.com/photo-1550684848-fac1c5b4e853?q=80&w=600&auto=format&fit=crop',
    rotationClass: 'lp-rotate-n12',
  },
  {
    id: '6',
    title: 'Real-time Dashboard',
    image: 'https://images.unsplash.com/photo-1541701494587-cb58502866ab?q=80&w=600&auto=format&fit=crop',
    rotationClass: 'lp-rotate-12',
  },
];

export default function TechnologySection() {
  const containerRef = useRef(null);
  const contentRef = useRef(null);
  const colLeftRef = useRef(null);
  const colRightRef = useRef(null);
  const [activeLightbox, setActiveLightbox] = useState(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      ScrollTrigger.create({
        trigger: containerRef.current,
        start: 'top top',
        end: 'bottom bottom',
        pin: contentRef.current,
        pinSpacing: false,
      });

      gsap.fromTo(
        colLeftRef.current,
        { y: '20%' },
        {
          y: '-30%',
          ease: 'none',
          scrollTrigger: {
            trigger: containerRef.current,
            start: 'top bottom',
            end: 'bottom top',
            scrub: true,
          },
        }
      );

      gsap.fromTo(
        colRightRef.current,
        { y: '50%' },
        {
          y: '-50%',
          ease: 'none',
          scrollTrigger: {
            trigger: containerRef.current,
            start: 'top bottom',
            end: 'bottom top',
            scrub: true,
          },
        }
      );
    }, containerRef);

    return () => ctx.revert();
  }, []);

  const colLeft = ITEMS.slice(0, 3);
  const colRight = ITEMS.slice(3, 6);

  return (
    <section ref={containerRef} className="lp-explorations">
      <div ref={contentRef} className="lp-explorations-content">
        <div className="lp-explorations-center">
          <span className="lp-section-tag-text" style={{ marginBottom: '12px', display: 'block' }}>
            Technology Stack
          </span>
          <h2>
            Under the <span className="lp-section-title-italic">hood</span>
          </h2>
          <p>
            Powered by YOLOv11, Google Gemini, OpenCV, and a custom multi-agent orchestration engine.
          </p>
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="lp-explorations-link"
          >
            View on GitHub ↗
          </a>
        </div>
      </div>

      <div className="lp-explorations-grid-wrap">
        <div className="lp-explorations-grid">
          <div ref={colLeftRef} className="lp-explorations-col-left">
            {colLeft.map((item) => (
              <div
                key={item.id}
                onClick={() => setActiveLightbox(item.image)}
                className={`lp-explorations-card ${item.rotationClass}`}
              >
                <img src={item.image} alt={item.title} />
              </div>
            ))}
          </div>

          <div ref={colRightRef} className="lp-explorations-col-right">
            {colRight.map((item) => (
              <div
                key={item.id}
                onClick={() => setActiveLightbox(item.image)}
                className={`lp-explorations-card ${item.rotationClass}`}
              >
                <img src={item.image} alt={item.title} />
              </div>
            ))}
          </div>
        </div>
      </div>

      {activeLightbox && (
        <div className="lp-lightbox" onClick={() => setActiveLightbox(null)}>
          <img src={activeLightbox} alt="Enlarged view" />
        </div>
      )}
    </section>
  );
}
