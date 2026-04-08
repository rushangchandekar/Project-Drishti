'use client';
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, Camera, MapPin, Maximize, Play, Loader2, Video } from 'lucide-react';

function SetupWizard({ onComplete }) {
    const [step, setStep] = useState('loading');
    const [videos, setVideos] = useState([]);
    const [formData, setFormData] = useState({
        sourceType: 'webcam',
        videoPath: '',
        venueName: 'Downtown Stadium, NYC',
        squareFeet: 1000
    });

    useEffect(() => {
        const initSystem = async () => {
             // Mock loading transition for nice UI
             setTimeout(() => setStep('config'), 1500);
        };
        initSystem();
    }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setStep('finalizing');
        setTimeout(() => {
            onComplete();
        }, 1500);
    };

    return (
        <div style={{ position: 'fixed', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-primary)', zIndex: 9999 }}>
            <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(circle at 50% 50%, rgba(102, 126, 234, 0.1) 0%, rgba(13, 17, 23, 1) 100%)' }} />

            <AnimatePresence mode="wait">
                {step === 'loading' && (
                    <motion.div
                        key="loading"
                        className="glass"
                        style={{ padding: '40px', borderRadius: '20px', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', maxWidth: '400px', zIndex: 10 }}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 1.1 }}
                    >
                        <Shield size={64} style={{ color: 'var(--accent-indigo)', marginBottom: '24px' }} className="animate-pulse" />
                        <h1 className="gradient-text" style={{ fontSize: '28px', fontWeight: 800, letterSpacing: '4px', marginBottom: '16px' }}>DRISHTI</h1>
                        <p style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--text-secondary)', fontSize: '14px' }}>
                            <Loader2 className="animate-spin" size={16} />
                            Initializing Core Intelligence...
                        </p>
                    </motion.div>
                )}

                {step === 'config' && (
                    <motion.div
                        key="config"
                        className="glass"
                        style={{ padding: '40px', borderRadius: '20px', width: '100%', maxWidth: '500px', zIndex: 10 }}
                        initial={{ opacity: 0, y: 40 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                    >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '32px' }}>
                            <Shield size={32} style={{ color: 'var(--accent-indigo)' }} />
                            <div>
                                <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '4px' }}>System Configuration</h2>
                                <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Calibrate Drishti AI for your venue environment</p>
                            </div>
                        </div>

                        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}><MapPin size={16} /> Venue Name</label>
                                <input type="text" required value={formData.venueName} onChange={(e) => setFormData({ ...formData, venueName: e.target.value })} style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)', color: 'white', padding: '12px 16px', borderRadius: '12px', fontSize: '14px', outline: 'none' }} />
                            </div>
                            <button type="submit" className="btn-primary" style={{ padding: '16px', borderRadius: '12px', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px', marginTop: '16px', fontSize: '15px' }}>
                                <Play size={20} fill="currentColor" />
                                Initiate Surveillance
                            </button>
                        </form>
                    </motion.div>
                )}

                {step === 'finalizing' && (
                    <motion.div
                        key="finalizing"
                         className="glass"
                        style={{ padding: '40px', borderRadius: '20px', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', maxWidth: '400px', zIndex: 10 }}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        <Shield size={64} style={{ color: 'var(--status-safe)' }} />
                        <h2 style={{ marginTop: '24px', marginBottom: '16px' }}>System Calibrated</h2>
                        <p style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--text-secondary)' }}>
                            Starting intelligent detection engine...
                        </p>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

export default SetupWizard;
