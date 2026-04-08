import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, Camera, MapPin, Maximize, Play, Loader2, Video } from 'lucide-react';

function SetupWizard({ onComplete }) {
    const [step, setStep] = useState('loading'); // 'loading', 'config', 'finalizing'
    const [videos, setVideos] = useState([]);
    const [formData, setFormData] = useState({
        sourceType: 'webcam',
        videoPath: '',
        venueName: '',
        squareFeet: 1000
    });

    useEffect(() => {
        const initSystem = async () => {
            try {
                // Check backend health
                const healthRes = await fetch('http://localhost:8000/');
                if (!healthRes.ok) throw new Error('Backend offline');

                // Fetch available video files
                const vidRes = await fetch('http://localhost:8000/list-videos');
                const vidData = await vidRes.json();

                setVideos(vidData.videos || []);
                if (vidData.videos?.length > 0) {
                    setFormData(prev => ({ ...prev, videoPath: vidData.videos[0].path }));
                }

                setTimeout(() => setStep('config'), 1500);
            } catch (err) {
                console.log("Waiting for backend...");
                setTimeout(initSystem, 2000);
            }
        };
        initSystem();
    }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setStep('finalizing');

        try {
            const res = await fetch('http://localhost:8000/configure', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video_source_type: formData.sourceType,
                    video_path: formData.sourceType === 'file' ? formData.videoPath : null,
                    venue_name: formData.venueName || "Unnamed Venue",
                    square_feet: Number(formData.squareFeet)
                })
            });

            if (!res.ok) throw new Error("Configuration failed");

            setTimeout(() => {
                onComplete();
            }, 1500);
        } catch (err) {
            console.error(err);
            alert("Failed to initialize system. Make sure backend is running.");
            setStep('config');
        }
    };

    return (
        <div className="setup-wizard">
            <div className="wizard-background" />

            <AnimatePresence mode="wait">
                {step === 'loading' && (
                    <motion.div
                        key="loading"
                        className="wizard-step centered"
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 1.1 }}
                    >
                        <Shield size={64} className="wizard-logo pulse-glow" />
                        <h1 className="gradient-text">PROJECT DRISHTI</h1>
                        <p className="loading-text">
                            <Loader2 className="spinner" size={16} />
                            Initializing Core Intelligence...
                        </p>
                    </motion.div>
                )}

                {step === 'config' && (
                    <motion.div
                        key="config"
                        className="wizard-step"
                        initial={{ opacity: 0, y: 40 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                    >
                        <div className="wizard-header">
                            <Shield size={32} className="wizard-logo" />
                            <div>
                                <h2>System Configuration</h2>
                                <p>Calibrate Drishti for your venue environment</p>
                            </div>
                        </div>

                        <form onSubmit={handleSubmit} className="setup-form">
                            <div className="form-group">
                                <label><MapPin size={16} /> Venue Name</label>
                                <input
                                    type="text"
                                    required
                                    placeholder="e.g. Main Concert Hall"
                                    value={formData.venueName}
                                    onChange={(e) => setFormData({ ...formData, venueName: e.target.value })}
                                />
                            </div>

                            <div className="form-group">
                                <label><Maximize size={16} /> Monitored Area (Sq. Ft.)</label>
                                <input
                                    type="number"
                                    required
                                    min="100"
                                    step="50"
                                    placeholder="1000"
                                    value={formData.squareFeet}
                                    onChange={(e) => setFormData({ ...formData, squareFeet: e.target.value })}
                                />
                                <span className="helper-text">Used to dynamically calibrate density alerts (approx {Math.floor(formData.squareFeet / 15)} safe capacity).</span>
                            </div>

                            <div className="form-group">
                                <label><Camera size={16} /> Video Source</label>
                                <div className="source-toggles">
                                    <button
                                        type="button"
                                        className={formData.sourceType === 'webcam' ? 'active' : ''}
                                        onClick={() => setFormData({ ...formData, sourceType: 'webcam' })}
                                    >
                                        <Camera size={18} /> Live Webcam
                                    </button>
                                    <button
                                        type="button"
                                        className={formData.sourceType === 'file' ? 'active' : ''}
                                        onClick={() => setFormData({ ...formData, sourceType: 'file' })}
                                    >
                                        <Video size={18} /> Video File
                                    </button>
                                </div>
                            </div>

                            {formData.sourceType === 'file' && (
                                <div className="form-group slide-down">
                                    <label>Select Video File</label>
                                    <select
                                        value={formData.videoPath}
                                        onChange={(e) => setFormData({ ...formData, videoPath: e.target.value })}
                                        required
                                    >
                                        {videos.length === 0 ? (
                                            <option value="">No videos found in data/ folder</option>
                                        ) : (
                                            videos.map(v => (
                                                <option key={v.path} value={v.path}>{v.name}</option>
                                            ))
                                        )}
                                    </select>
                                </div>
                            )}

                            <button type="submit" className="start-btn">
                                <Play size={20} fill="currentColor" />
                                Initiate Surveillance
                            </button>
                        </form>
                    </motion.div>
                )}

                {step === 'finalizing' && (
                    <motion.div
                        key="finalizing"
                        className="wizard-step centered"
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        <Shield size={64} className="wizard-logo" style={{ color: 'var(--status-safe)' }} />
                        <h2 style={{ marginTop: 24 }}>System Calibrated</h2>
                        <p className="loading-text">
                            Starting intelligent detection engine...
                        </p>
                    </motion.div>
                )}
            </AnimatePresence>

            <style>{`
                .setup-wizard {
                    position: fixed;
                    inset: 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: var(--bg-primary);
                    z-index: 9999;
                    font-family: inherit;
                }

                .wizard-background {
                    position: absolute;
                    inset: 0;
                    background: radial-gradient(circle at 50% 50%, rgba(102, 126, 234, 0.1) 0%, rgba(13, 17, 23, 1) 100%);
                }

                .wizard-step {
                    position: relative;
                    z-index: 10;
                    width: 100%;
                    max-width: 500px;
                    background: rgba(13, 17, 23, 0.85);
                    backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 20px;
                    padding: 40px;
                    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                }

                .wizard-step.centered {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    text-align: center;
                    max-width: 400px;
                }

                .wizard-logo {
                    color: #667eea;
                    margin-bottom: 24px;
                }

                .pulse-glow {
                    animation: pulse-glow 2s infinite ease-in-out;
                }

                @keyframes pulse-glow {
                    0%, 100% { filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.4)); opacity: 1; }
                    50% { filter: drop-shadow(0 0 40px rgba(102, 126, 234, 0.8)); opacity: 0.7; }
                }

                .gradient-text {
                    font-size: 28px;
                    font-weight: 800;
                    letter-spacing: 4px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin-bottom: 16px;
                }

                .loading-text {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    color: var(--text-secondary);
                    font-size: 14px;
                }

                .spinner {
                    animation: spin 1s linear infinite;
                }

                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }

                .wizard-header {
                    display: flex;
                    align-items: center;
                    gap: 16px;
                    margin-bottom: 32px;
                }

                .wizard-header h2 {
                    font-size: 20px;
                    font-weight: 600;
                    margin-bottom: 4px;
                }

                .wizard-header p {
                    font-size: 12px;
                    color: var(--text-muted);
                }

                .setup-form {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }

                .form-group {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }

                .form-group label {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 12px;
                    font-weight: 600;
                    color: var(--text-secondary);
                }

                .form-group input, .form-group select {
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    color: var(--text-primary);
                    padding: 12px 16px;
                    border-radius: 12px;
                    font-size: 14px;
                    outline: none;
                    transition: border-color 0.2s;
                }

                .form-group input:focus, .form-group select:focus {
                    border-color: #667eea;
                }
                
                .form-group option {
                    background: var(--bg-primary);
                    color: var(--text-primary);
                }

                .helper-text {
                    font-size: 11px;
                    color: var(--text-muted);
                    margin-left: 4px;
                }

                .source-toggles {
                    display: flex;
                    gap: 12px;
                }

                .source-toggles button {
                    flex: 1;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    padding: 12px;
                    border-radius: 12px;
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    color: var(--text-secondary);
                    cursor: pointer;
                    font-weight: 500;
                    transition: all 0.2s;
                }

                .source-toggles button:hover {
                    background: rgba(255, 255, 255, 0.08);
                }

                .source-toggles button.active {
                    background: rgba(102, 126, 234, 0.15);
                    border-color: #667eea;
                    color: #667eea;
                }

                .start-btn {
                    margin-top: 16px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    padding: 16px;
                    border-radius: 12px;
                    font-size: 15px;
                    font-weight: 600;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 10px;
                    cursor: pointer;
                    transition: transform 0.2s, box-shadow 0.2s;
                    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
                }

                .start-btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 12px 32px rgba(102, 126, 234, 0.5);
                }
            `}</style>
        </div>
    );
}

export default SetupWizard;
