'use client';
import { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';
import gsap from 'gsap';

const ROLES = ['Intelligent', 'Autonomous', 'Real-time', 'Adaptive'];
const HLS_URL = 'https://stream.mux.com/Aa02T7oM1wH5Mk5EEVDYhbZ1ChcdhRsS2m1NYyx4Ua1g.m3u8';

export default function HeroSection() {
  const videoRef = useRef(null);
  const heroRef = useRef(null);
  const [roleIndex, setRoleIndex] = useState(0);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (Hls.isSupported()) {
      const hls = new Hls({ enableWorker: false });
      hls.loadSource(HLS_URL);
      hls.attachMedia(video);
      return () => hls.destroy();
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      video.src = HLS_URL;
    }
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setRoleIndex((prev) => (prev + 1) % ROLES.length);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });

      tl.to('.lp-hero-title', {
        opacity: 1,
        y: 0,
        duration: 1.2,
        delay: 0.1,
      }).to(
        '.blur-in',
        {
          opacity: 1,
          filter: 'blur(0px)',
          y: 0,
          duration: 1,
          stagger: 0.1,
        },
        '-=0.8'
      );
    }, heroRef);

    return () => ctx.revert();
  }, []);

  return (
    <section id="hero" ref={heroRef} className="lp-hero">
      <div className="lp-hero-video-wrap">
        <video
          ref={videoRef}
          autoPlay
          muted
          loop
          playsInline
          className="lp-hero-video"
        />
        <div className="lp-hero-overlay" />
        <div className="lp-hero-gradient-bottom" />
      </div>

      <div className="lp-hero-content">
        <span className="lp-hero-tag blur-in">
          AI SURVEILLANCE PLATFORM
        </span>

        <h1 className="lp-hero-title">
          Project Drishti
        </h1>

        <p className="lp-hero-subtitle blur-in">
          An{' '}
          <span key={roleIndex} className="lp-hero-role lp-animate-role-fade-in">
            {ROLES[roleIndex]}
          </span>{' '}
          crowd safety system.
        </p>

        <p className="lp-hero-desc blur-in">
          AI-powered multi-agent surveillance with real-time crowd analytics,
          anomaly detection, and autonomous emergency response.
        </p>

        <div className="lp-hero-buttons blur-in">
          <a href="#capabilities" className="lp-hero-btn-primary">
            <span className="accent-gradient-border" style={{ position: 'absolute', inset: 0, borderRadius: '9999px', opacity: 0 }} />
            Explore Features
          </a>

          <a href="/dashboard" className="lp-hero-btn-secondary">
            <span className="accent-gradient-border" style={{ position: 'absolute', inset: 0, borderRadius: '9999px', opacity: 0 }} />
            Go to Dashboard
          </a>
        </div>
      </div>

      <div className="lp-hero-scroll-indicator">
        <span className="lp-hero-scroll-label">SCROLL</span>
        <div className="lp-hero-scroll-track">
          <div className="lp-hero-scroll-thumb accent-gradient lp-animate-scroll-down" />
        </div>
      </div>
    </section>
  );
}
