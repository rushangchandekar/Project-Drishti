'use client';
import { useState, useEffect } from 'react';

export default function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 100);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (id) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <nav className="lp-navbar">
      <div className={`lp-navbar-inner${isScrolled ? ' scrolled' : ''}`}>
        <div className="lp-nav-logo accent-gradient" onClick={() => scrollToSection('hero')}>
          <div className="lp-nav-logo-inner">
            <span>D</span>
          </div>
        </div>

        <div className="lp-nav-divider lp-nav-divider-hidden" />

        <div className="lp-nav-links">
          <button
            onClick={() => scrollToSection('hero')}
            className="lp-nav-link active"
          >
            Home
          </button>
          <button
            onClick={() => scrollToSection('capabilities')}
            className="lp-nav-link"
          >
            Features
          </button>
          <a
            href="/dashboard"
            className="lp-nav-link"
          >
            Dashboard
          </a>
        </div>

        <div className="lp-nav-divider" />

        <button
          onClick={() => scrollToSection('contact')}
          className="lp-nav-cta"
        >
          <span className="lp-nav-cta-glow accent-gradient" />
          <span className="lp-nav-cta-inner">
            Contact <span style={{ fontSize: '12px' }}>↗</span>
          </span>
        </button>
      </div>
    </nav>
  );
}
