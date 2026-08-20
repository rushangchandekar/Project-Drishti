'use client';
import { useEffect, useRef } from 'react';
import Hls from 'hls.js';
import gsap from 'gsap';

const HLS_URL = 'https://stream.mux.com/Aa02T7oM1wH5Mk5EEVDYhbZ1ChcdhRsS2m1NYyx4Ua1g.m3u8';

export default function ContactFooterSection() {
  const videoRef = useRef(null);
  const marqueeRef = useRef(null);

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
    const ctx = gsap.context(() => {
      gsap.to(marqueeRef.current, {
        xPercent: -50,
        duration: 40,
        ease: 'none',
        repeat: -1,
      });
    });
    return () => ctx.revert();
  }, []);

  return (
    <footer id="contact" className="lp-footer">
      <div className="lp-footer-video-wrap">
        <video
          ref={videoRef}
          autoPlay
          muted
          loop
          playsInline
          className="lp-footer-video"
        />
        <div className="lp-footer-video-overlay" />
      </div>

      <div className="lp-footer-content lp-container">
        <div className="lp-marquee-wrap">
          <div ref={marqueeRef} className="lp-marquee-text">
            {Array(10).fill('INTELLIGENT SURVEILLANCE • ').join('')}
          </div>
        </div>

        <span className="lp-footer-tag">Ready to deploy?</span>

        <h2 className="lp-footer-title">
          Secure your venue.
        </h2>

        <a href="/dashboard" className="lp-footer-cta">
          <span className="accent-gradient-border" style={{ position: 'absolute', inset: 0, borderRadius: '9999px', opacity: 0 }} />
          Launch Dashboard →
        </a>

        <div className="lp-footer-bottom">
          <div className="lp-footer-socials">
            {['GitHub', 'LinkedIn', 'Documentation'].map((link) => (
              <a key={link} href="#">
                {link}
              </a>
            ))}
          </div>

          <div className="lp-footer-status">
            <span className="lp-footer-status-dot lp-animate-pulse" />
            <span>System Online</span>
          </div>

          <div>© {new Date().getFullYear()} Project Drishti. All rights reserved.</div>
        </div>
      </div>
    </footer>
  );
}
