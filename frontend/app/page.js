'use client';

import { useState } from 'react';
import SetupWizard from '@/components/setup/SetupWizard';
import Dashboard from '@/components/Dashboard';

export default function HomePage() {
  // Bypassed setup wizard temporarily by setting initial state to true
  const [isSetupComplete, setIsSetupComplete] = useState(true);

  if (!isSetupComplete) {
    return <SetupWizard onComplete={() => setIsSetupComplete(true)} />;
  }

  return <Dashboard />;
}
