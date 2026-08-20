'use client';

import { useState } from 'react';
import { AnimatePresence } from 'framer-motion';
import LoadingScreen from '../landing/LoadingScreen';
import Navbar from '../landing/Navbar';
import HeroSection from '../landing/HeroSection';
import CapabilitiesSection from '../landing/CapabilitiesSection';
import PipelineSection from '../landing/PipelineSection';
import TechnologySection from '../landing/TechnologySection';
import StatsSection from '../landing/StatsSection';
import ContactFooterSection from '../landing/ContactFooterSection';

export default function LandingPage() {
  const [isLoading, setIsLoading] = useState(true);

  return (
    <div className="landing-page">
      <AnimatePresence mode="wait">
        {isLoading && <LoadingScreen onComplete={() => setIsLoading(false)} />}
      </AnimatePresence>

      {!isLoading && (
        <>
          <Navbar />
          <main>
            <HeroSection />
            <CapabilitiesSection />
            <PipelineSection />
            <TechnologySection />
            <StatsSection />
            <ContactFooterSection />
          </main>
        </>
      )}
    </div>
  );
}
