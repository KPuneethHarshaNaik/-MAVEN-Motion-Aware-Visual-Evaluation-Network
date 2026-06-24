import { useState, useRef, useEffect, useCallback } from 'react';
import GLSLHills from './components/ui/glsl-hills';
import { Button } from './components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Badge } from './components/ui/badge';
import { Progress } from './components/ui/progress';
import {
  ChevronDown, Upload, FileVideo, Zap, Brain, Eye, BarChart3, Shield,
  CheckCircle2, AlertCircle, Clock, Cpu, Layers, Sparkles, ArrowRight, X
} from 'lucide-react';
import { runPrediction, fetchModelInfo } from './lib/api';
import type { ModelInfo, PredictionResult } from './lib/api';
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  PointElement, LineElement, BarElement,
  Title, Tooltip, Legend, Filler,
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler);

/* ─── Pipeline stages ───────────────────────────────────────────── */
const STAGES = [
  { icon: FileVideo, label: 'Video Read', desc: 'Reading metadata & codec' },
  { icon: Layers, label: 'Frame Extract', desc: 'Sampling 16 key frames' },
  { icon: Cpu, label: 'CNN Backbone', desc: 'EfficientNetV2-S encoding' },
  { icon: Brain, label: 'Transformer', desc: 'Temporal attention analysis' },
  { icon: Shield, label: 'Classification', desc: 'Final ASD/TD prediction' },
];

/* ─── Feature cards for homepage ───────────────────────────────── */
const FEATURES = [
  { icon: Eye, title: 'Motion Analysis', desc: 'Tracks spatial movement patterns across video frames to identify behavioral markers.' },
  { icon: Brain, title: 'Temporal Attention', desc: 'Transformer-based architecture learns which moments in the video matter most.' },
  { icon: BarChart3, title: 'Explainable AI', desc: 'See exactly which frames and features drove the prediction, with full transparency.' },
  { icon: Zap, title: 'Real-time Inference', desc: 'GPU-accelerated pipeline delivers results in seconds, not minutes.' },
];

function App() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStage, setCurrentStage] = useState(-1);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const screeningRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchModelInfo().then(setModelInfo).catch(() => {});
  }, []);

  const scrollToScreening = () => screeningRef.current?.scrollIntoView({ behavior: 'smooth' });

  const handleFile = useCallback((f: File) => {
    if (f.size > 200 * 1024 * 1024) { setError('File too large (max 200 MB).'); return; }
    const ext = f.name.split('.').pop()?.toLowerCase();
    if (!['mp4', 'avi', 'mov', 'mkv', 'webm'].includes(ext || '')) { setError('Unsupported format. Use MP4, AVI, MOV, MKV, or WebM.'); return; }
    setFile(f); setError(null); setResult(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setIsDragging(false);
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const runAnalysis = async () => {
    if (!file) return;
    setIsProcessing(true); setError(null); setResult(null); setCurrentStage(0);

    const stageTimers = [500, 1200, 2500, 3500];
    stageTimers.forEach((ms, i) => setTimeout(() => setCurrentStage(i + 1), ms));

    try {
      const data = await runPrediction(file);
      setCurrentStage(4);
      setTimeout(() => { setResult(data); setIsProcessing(false); }, 600);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
      setIsProcessing(false); setCurrentStage(-1);
    }
  };

  const clearAll = () => { setFile(null); setResult(null); setError(null); setCurrentStage(-1); };

  /* ─── Charts ──────────────────────────────────────────────────── */
  const chartOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b', font: { size: 10 } }, border: { display: false } },
      y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b', font: { size: 10 } }, border: { display: false } },
    },
  };

  const energyChart = result ? {
    labels: result.frame_energies.map((_: number, i: number) => `${i + 1}`),
    datasets: [{
      label: 'Motion Energy',
      data: result.frame_energies,
      borderColor: '#06b6d4',
      backgroundColor: 'rgba(6, 182, 212, 0.08)',
      fill: true, tension: 0.4, pointRadius: 3, pointBackgroundColor: '#06b6d4',
    }],
  } : null;

  const attnChart = result ? {
    labels: result.frame_weights.map((_: number, i: number) => `${i + 1}`),
    datasets: [{
      label: 'Attention %',
      data: result.frame_weights,
      backgroundColor: result.frame_weights.map((_: number, i: number) =>
        result.top_frames.includes(i) ? '#8b5cf6' : 'rgba(139, 92, 246, 0.2)'
      ),
      borderRadius: 4,
    }],
  } : null;

  return (
    <div className="min-h-screen bg-background text-foreground">

      {/* ══════ HEADER ══════ */}
      <header className="fixed top-0 left-0 right-0 z-50 glass-strong">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center font-black text-sm text-white shadow-lg shadow-primary/20">
              M
            </div>
            <span className="font-bold text-lg tracking-tight">MAVEN</span>
            <Badge variant="outline" className="text-[10px] font-medium border-primary/30 text-primary ml-1 hidden sm:inline-flex">v2.0</Badge>
          </div>
          <div className="flex items-center gap-3">
            {modelInfo ? (
              <>
                <div className="hidden md:flex items-center gap-2 text-xs text-muted-foreground">
                  <span>Epoch {modelInfo.epoch}</span>
                  <span className="w-1 h-1 rounded-full bg-muted-foreground/50" />
                  <span>AUC {modelInfo.auc}</span>
                </div>
                <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 text-xs">
                  <CheckCircle2 className="w-3 h-3 mr-1" /> Ready
                </Badge>
                <Badge variant="secondary" className="text-xs hidden sm:inline-flex">
                  <Cpu className="w-3 h-3 mr-1" />{modelInfo.device}
                </Badge>
              </>
            ) : (
              <Badge variant="secondary" className="text-xs animate-pulse">Connecting...</Badge>
            )}
          </div>
        </div>
      </header>

      {/* ══════ HERO ══════ */}
      <section className="relative h-screen flex items-center justify-center overflow-hidden">
        {/* Animated 3D Background */}
        <GLSLHills />
        
        {/* Radial gradients for depth */}
        <div className="absolute inset-0 z-[2]">
          <div className="absolute top-1/4 left-1/4 w-[600px] h-[600px] radial-purple opacity-50" />
          <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] radial-cyan opacity-30" />
        </div>

        {/* Grid pattern */}
        <div className="absolute inset-0 z-[3] grid-pattern opacity-40" />

        {/* Hero content */}
        <div className="relative z-10 text-center max-w-5xl px-6">
          <div className="animate-fade-up">
            <Badge className="mb-8 px-4 py-1.5 text-sm bg-primary/15 text-primary border-primary/25 backdrop-blur-md">
              <Sparkles className="w-3.5 h-3.5 mr-1.5" />
              AI-Powered ASD Screening
            </Badge>
          </div>

          <h1 className="text-gradient-subtle text-6xl md:text-8xl font-black tracking-tight mb-8 leading-[0.95] animate-fade-up" style={{ animationDelay: '0.15s' }}>
            Motion-Aware
            <br />
            <span className="text-gradient">Visual Evaluation</span>
          </h1>

          <p className="text-lg md:text-xl text-muted-foreground mb-12 max-w-2xl mx-auto leading-relaxed animate-fade-up" style={{ animationDelay: '0.3s' }}>
            Advanced deep learning for early Autism Spectrum Disorder detection
            through spatial-temporal video analysis.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center animate-fade-up" style={{ animationDelay: '0.45s' }}>
            <Button
              size="lg"
              className="rounded-full px-10 h-14 text-base font-semibold bg-primary hover:bg-primary/90 text-white animate-pulse-glow transition-all"
              onClick={scrollToScreening}
            >
              Start Analysis
              <ArrowRight className="ml-2 w-4 h-4" />
            </Button>
            <Button
              variant="outline"
              size="lg"
              className="rounded-full px-8 h-14 text-base border-border/50 hover:bg-white/5"
              onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
            >
              Learn More
            </Button>
          </div>
        </div>

        {/* Scroll hint */}
        <div className="absolute bottom-10 left-1/2 -translate-x-1/2 z-10 flex flex-col items-center gap-2 text-muted-foreground/50 animate-float">
          <span className="text-xs tracking-widest uppercase">Scroll</span>
          <ChevronDown className="w-5 h-5" />
        </div>
      </section>

      {/* ══════ FEATURES ══════ */}
      <section id="features" className="relative py-32 overflow-hidden">
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/20 to-transparent" />
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-20">
            <Badge variant="secondary" className="mb-4 text-xs">Architecture</Badge>
            <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-6 text-gradient-subtle">How MAVEN Works</h2>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
              An EfficientNetV2 backbone extracts spatial features while a Temporal Transformer learns motion dynamics.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {FEATURES.map((f, i) => (
              <div
                key={i}
                className="group glass rounded-2xl p-8 hover:bg-white/[0.03] transition-all duration-500 hover:-translate-y-1"
              >
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary/20 to-accent/10 flex items-center justify-center mb-6 group-hover:shadow-lg group-hover:shadow-primary/10 transition-shadow">
                  <f.icon className="w-6 h-6 text-primary" />
                </div>
                <h3 className="font-semibold text-lg mb-3">{f.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ══════ SCREENING ══════ */}
      <section ref={screeningRef} className="relative py-32 min-h-screen">
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-accent/20 to-transparent" />
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <Badge variant="secondary" className="mb-4 text-xs">
              <Zap className="w-3 h-3 mr-1" /> Live Analysis
            </Badge>
            <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-4 text-gradient-subtle">Run Screening</h2>
            <p className="text-muted-foreground text-lg">Upload a video and get an AI-powered assessment in seconds.</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* ── Left: Upload + Pipeline ── */}
            <div className="lg:col-span-4 space-y-6">

              {/* Upload */}
              <div className="glass rounded-2xl p-6">
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <Upload className="w-4 h-4 text-primary" /> Video Upload
                </h3>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                />
                <div
                  className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-300
                    ${isDragging ? 'border-primary bg-primary/5 scale-[1.02]' : ''}
                    ${file ? 'border-primary/40 bg-primary/5' : 'border-border/50 hover:border-primary/30 hover:bg-white/[0.02]'}
                    ${isProcessing ? 'pointer-events-none opacity-50' : ''}
                  `}
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                >
                  {file ? (
                    <div className="space-y-3">
                      <FileVideo className="w-10 h-10 mx-auto text-primary" />
                      <p className="font-medium text-sm truncate max-w-[200px] mx-auto">{file.name}</p>
                      <p className="text-xs text-muted-foreground">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
                      {!isProcessing && (
                        <Button size="sm" className="mt-2 rounded-full" onClick={(e) => { e.stopPropagation(); runAnalysis(); }}>
                          <Zap className="w-3 h-3 mr-1" /> Run Analysis
                        </Button>
                      )}
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="w-16 h-16 mx-auto rounded-2xl bg-secondary/50 flex items-center justify-center">
                        <Upload className="w-7 h-7 text-muted-foreground" />
                      </div>
                      <div>
                        <p className="font-medium text-sm">Drop video here or click to browse</p>
                        <p className="text-xs text-muted-foreground mt-1">MP4, AVI, MOV, MKV, WebM • Max 200MB</p>
                      </div>
                    </div>
                  )}
                </div>

                {error && (
                  <div className="mt-4 p-3 rounded-lg bg-destructive/10 border border-destructive/20 flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
                    <p className="text-xs text-destructive">{error}</p>
                    <button onClick={() => setError(null)} className="ml-auto text-destructive/60 hover:text-destructive">
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                )}
              </div>

              {/* Pipeline */}
              {(isProcessing || result) && (
                <div className="glass rounded-2xl p-6 animate-scale-in">
                  <h3 className="font-semibold mb-5 text-sm">Pipeline Progress</h3>
                  <div className="space-y-3">
                    {STAGES.map((s, i) => {
                      const done = currentStage > i || !!result;
                      const active = currentStage === i && !result;
                      return (
                        <div key={i} className={`flex items-center gap-3 p-2.5 rounded-lg transition-all duration-300
                          ${done ? 'bg-primary/5' : active ? 'bg-accent/5' : 'opacity-40'}
                        `}>
                          <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 transition-all
                            ${done ? 'bg-primary/20 text-primary' : active ? 'bg-accent/20 text-accent animate-pulse' : 'bg-secondary text-muted-foreground'}
                          `}>
                            {done ? <CheckCircle2 className="w-4 h-4" /> : <s.icon className="w-4 h-4" />}
                          </div>
                          <div>
                            <p className="text-xs font-medium">{s.label}</p>
                            <p className="text-[10px] text-muted-foreground">{s.desc}</p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Video Meta */}
              {result && (
                <div className="glass rounded-2xl p-6 animate-scale-in">
                  <h3 className="font-semibold mb-4 text-sm text-muted-foreground">Video Metadata</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    {[
                      ['Resolution', `${result.video_meta.width}×${result.video_meta.height}`],
                      ['Duration', `${result.video_meta.duration}s`],
                      ['FPS', `${result.video_meta.fps}`],
                      ['Frames', `${result.video_meta.frames}`],
                    ].map(([label, val]) => (
                      <div key={label}>
                        <p className="text-xs text-muted-foreground">{label}</p>
                        <p className="font-medium">{val}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* ── Right: Results ── */}
            <div className="lg:col-span-8">

              {/* Empty state */}
              {!result && !isProcessing && (
                <div className="h-full min-h-[500px] glass rounded-2xl flex flex-col items-center justify-center">
                  <div className="text-center space-y-4">
                    <div className="w-20 h-20 mx-auto rounded-2xl bg-secondary/30 flex items-center justify-center">
                      <Brain className="w-10 h-10 text-muted-foreground/30" />
                    </div>
                    <p className="text-muted-foreground/50 text-sm">Upload a video to begin AI-powered screening</p>
                  </div>
                </div>
              )}

              {/* Processing state */}
              {isProcessing && !result && (
                <div className="h-full min-h-[500px] glass rounded-2xl flex flex-col items-center justify-center relative overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-primary/5 to-transparent animate-shimmer" />
                  <div className="relative z-10 text-center space-y-6">
                    <div className="w-20 h-20 mx-auto rounded-2xl bg-primary/10 flex items-center justify-center animate-pulse">
                      <Brain className="w-10 h-10 text-primary" />
                    </div>
                    <div>
                      <p className="text-lg font-semibold">Analyzing Video</p>
                      <p className="text-sm text-muted-foreground mt-1">
                        {STAGES[Math.min(currentStage, 4)]?.desc || 'Processing...'}
                      </p>
                    </div>
                    <Progress value={(currentStage + 1) * 20} className="w-48 mx-auto h-1.5" />
                  </div>
                </div>
              )}

              {/* Results */}
              {result && (
                <div className="space-y-6">

                  {/* Prediction Card */}
                  <div className={`glass rounded-2xl overflow-hidden animate-scale-in border ${
                    result.label === 'ASD' ? 'border-primary/30' : 'border-emerald-500/30'
                  }`}>
                    <div className={`h-1 ${result.label === 'ASD' ? 'bg-gradient-to-r from-primary to-accent' : 'bg-gradient-to-r from-emerald-500 to-cyan-500'}`} />
                    <div className="p-8">
                      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
                        <div>
                          <div className="flex items-center gap-3 mb-2">
                            <h3 className="text-3xl font-bold">{result.label} Detected</h3>
                            <Badge className={result.label === 'ASD' ? 'bg-primary/15 text-primary border-primary/30' : 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'}>
                              {result.confidence.toFixed(1)}% confidence
                            </Badge>
                          </div>
                          <p className="text-muted-foreground">Based on motion energy and temporal attention analysis</p>
                        </div>

                        {/* Confidence Ring */}
                        <div className="relative w-28 h-28 shrink-0">
                          <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                            <circle cx="50" cy="50" r="42" fill="none" stroke="currentColor" strokeWidth="6" className="text-secondary" />
                            <circle cx="50" cy="50" r="42" fill="none" strokeWidth="6"
                              strokeDasharray={`${result.confidence * 2.64} 264`}
                              strokeLinecap="round"
                              className={result.label === 'ASD' ? 'text-primary' : 'text-emerald-400'}
                              style={{ filter: `drop-shadow(0 0 6px ${result.label === 'ASD' ? 'rgba(139,92,246,0.5)' : 'rgba(16,185,129,0.5)'})` }}
                            />
                          </svg>
                          <div className="absolute inset-0 flex flex-col items-center justify-center">
                            <span className="text-2xl font-bold">{result.confidence.toFixed(0)}%</span>
                            <span className="text-[10px] text-muted-foreground">confidence</span>
                          </div>
                        </div>
                      </div>

                      {/* Probability Bars */}
                      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span className={result.label === 'ASD' ? 'text-primary font-medium' : 'text-muted-foreground'}>ASD Probability</span>
                            <span className="font-mono">{result.asd_prob.toFixed(1)}%</span>
                          </div>
                          <div className="h-2 bg-secondary rounded-full overflow-hidden">
                            <div className="h-full bg-gradient-to-r from-primary to-primary/70 rounded-full transition-all duration-1000" style={{ width: `${result.asd_prob}%` }} />
                          </div>
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span className={result.label === 'TD' ? 'text-emerald-400 font-medium' : 'text-muted-foreground'}>TD Probability</span>
                            <span className="font-mono">{result.td_prob.toFixed(1)}%</span>
                          </div>
                          <div className="h-2 bg-secondary rounded-full overflow-hidden">
                            <div className="h-full bg-gradient-to-r from-emerald-500 to-emerald-500/70 rounded-full transition-all duration-1000" style={{ width: `${result.td_prob}%` }} />
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Charts */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card className="glass rounded-2xl border-0 animate-fade-up" style={{ animationDelay: '0.1s' }}>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium flex items-center gap-2">
                          <Zap className="w-4 h-4 text-accent" /> Motion Energy
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="h-[200px]">
                          {energyChart && <Line data={energyChart} options={chartOpts} />}
                        </div>
                      </CardContent>
                    </Card>
                    <Card className="glass rounded-2xl border-0 animate-fade-up" style={{ animationDelay: '0.2s' }}>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium flex items-center gap-2">
                          <Brain className="w-4 h-4 text-primary" /> Temporal Attention
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="h-[200px]">
                          {attnChart && <Bar data={attnChart} options={chartOpts} />}
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Frame Grid */}
                  <div className="glass rounded-2xl p-6 animate-fade-up" style={{ animationDelay: '0.3s' }}>
                    <h3 className="font-semibold text-sm mb-1">Analyzed Frames</h3>
                    <p className="text-xs text-muted-foreground mb-5">Highlighted frames had the highest impact on the decision</p>
                    <div className="grid grid-cols-4 md:grid-cols-8 gap-2">
                      {result.thumbs.map((thumb: string, i: number) => {
                        const isTop = result.top_frames.includes(i);
                        return (
                          <div key={i} className={`relative rounded-lg overflow-hidden aspect-square transition-all duration-300
                            ${isTop ? 'ring-2 ring-primary shadow-lg shadow-primary/20 scale-105 z-10' : 'opacity-60 hover:opacity-100'}
                          `}>
                            <img src={`data:image/jpeg;base64,${thumb}`} alt={`Frame ${i + 1}`} className="w-full h-full object-cover" />
                            {isTop && (
                              <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-primary/80 to-transparent p-1">
                                <span className="text-[9px] font-bold text-white">{result.frame_weights[i].toFixed(1)}%</span>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Timing + Actions */}
                  <div className="flex flex-wrap items-center gap-3 animate-fade-up" style={{ animationDelay: '0.4s' }}>
                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                      <Clock className="w-3 h-3" />
                      <span>Total: {result.timing.total_ms}ms</span>
                    </div>
                    {[
                      ['Read', result.timing.video_read_ms],
                      ['Extract', result.timing.frame_extract_ms],
                      ['CNN', result.timing.cnn_encode_ms],
                    ].map(([l, v]) => (
                      <Badge key={l as string} variant="outline" className="text-[10px] font-mono">{l}: {v}ms</Badge>
                    ))}
                    <div className="flex-1" />
                    <Button variant="outline" size="sm" className="rounded-full text-xs" onClick={clearAll}>
                      New Analysis
                    </Button>
                  </div>

                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ══════ FOOTER ══════ */}
      <footer className="border-t border-border/50 py-12">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center font-bold text-[10px] text-white">M</div>
            <span className="text-sm font-medium text-muted-foreground">MAVEN — Motion-Aware Visual Evaluation Network</span>
          </div>
          <p className="text-xs text-muted-foreground/50">Built with EfficientNetV2-S + Temporal Transformer • For research use only</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
