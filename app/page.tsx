'use client'

import { useEffect, useRef, useState } from 'react'
import {
  Activity,
  ArrowDown,
  ArrowRight,
  ArrowUpRight,
  BadgeCheck,
  BookOpen,
  BrainCircuit,
  Check,
  ChevronDown,
  CloudUpload,
  Download,
  HeartPulse,
  Languages,
  MonitorPlay,
  Pause,
  Play,
  Radio,
  ScanFace,
  ShieldCheck,
  Sparkles,
  Square,
  Upload,
  Users,
  Volume2,
} from 'lucide-react'

const features = [
  ['Live Translation', Radio, 'See signed conversations become clear, useful text in real time.'],
  ['Video Upload', CloudUpload, 'Drop a practice clip in the demo and follow each processing step.'],
  ['English Output', Languages, 'A natural English transcript designed for everyday conversation.'],
  ['Hindi Output', BookOpen, 'Switch to Hindi when your audience feels most at home there.'],
  ['Conversation History', Activity, 'Keep a simple record of moments worth returning to.'],
  ['Accessibility', Users, 'Built around dignity, clarity, and more equal access to communication.'],
  ['Offline Ready', ShieldCheck, 'A future-ready direction for dependable, private interpretation.'],
  ['AI Powered', BrainCircuit, 'Research-led intelligence designed to understand context, not just gestures.'],
] as const

const datasets = [
  ['INCLUDE', 'Isolated Indian Sign Language words', '4,287 videos', 'Gesture recognition baseline'],
  ['ISLTranslate', 'Continuous ISL sentence translation', '31,000+ samples', 'Sequence and language research'],
  ['ISL-CSLTR', 'Continuous sign language recognition', '5,000 sentences', 'Temporal modeling'],
  ['CISLR', 'Large-vocabulary continuous ISL', '700+ glosses', 'Real-world vocabulary'],
  ['ISL Fingerspelling', 'Alphabet and number gestures', '36 classes', 'Spelling uncommon words'],
]

const roadmap = [
  ['01', 'Continuous ISL', 'Move from individual gestures to expressive, flowing sentences.'],
  ['02', '3D Avatar', 'Make translated speech visible, expressive, and two-way.'],
  ['03', 'Mobile App', 'Put a helpful interpreter in every pocket.'],
  ['04', 'Offline AI', 'Interpret with confidence, even when connectivity is limited.'],
  ['05', 'Emergency Mode', 'Prioritize critical words when every second matters.'],
  ['06', 'Multilingual Support', 'Build bridges across more Indian languages.'],
]

function Mark({ small = false }: { small?: boolean }) {
  return (
    <span className={`brand-mark ${small ? 'brand-mark-small' : ''}`} aria-hidden="true">
      <span />
      <span />
      <span />
    </span>
  )
}

function DemoPill({ children = 'Demo Simulation' }: { children?: string }) {
  return <span className="demo-pill"><span className="demo-dot" />{children}</span>
}

export default function Page() {
  const [previewStep, setPreviewStep] = useState(0)
  const [isTranslating, setIsTranslating] = useState(false)
  const [language, setLanguage] = useState<'english' | 'hindi'>('english')
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [uploadState, setUploadState] = useState<'idle' | 'processing' | 'ready'>('idle')
  const [fileName, setFileName] = useState('')
  const uploadRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const timer = window.setInterval(() => setPreviewStep((step) => (step + 1) % 5), 1800)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    if (!isTranslating) return
    const timer = window.setInterval(() => setPreviewStep((step) => (step + 1) % 5), 1300)
    return () => window.clearInterval(timer)
  }, [isTranslating])

  const transcript = language === 'english' ? 'Hello, how are you today?' : 'नमस्ते, आज आप कैसे हैं?'
  const previewLabels = ['Camera', 'Gesture Detection', 'English', 'Hindi', 'Speech']
  const previewIcons = [MonitorPlay, ScanFace, Languages, BookOpen, Volume2]

  function handleFile(file?: File) {
    if (!file) return
    setFileName(file.name)
    setUploadState('processing')
    window.setTimeout(() => setUploadState('ready'), 1400)
  }

  return (
    <main className="site-shell">
      <nav className="navbar" aria-label="Main navigation">
        <a href="#top" className="brand" aria-label="Silent Interpreter home"><Mark /><span>Silent Interpreter</span></a>
        <div className="nav-links">
          <a href="#features">Features</a>
          <a href="#workflow">Workflow</a>
          <a href="#demo">Demo</a>
          <a href="#datasets">Datasets</a>
          <a href="#roadmap">Roadmap</a>
        </div>
        <a className="nav-cta" href="#demo">Sign up <ArrowRight size={15} /></a>
      </nav>

      <section className="hero section-pad" id="top">
        <div className="ambient ambient-one" /><div className="ambient ambient-two" />
        <div className="hero-copy reveal">
          <div className="eyebrow"><span className="eyebrow-line" /> ISL TO SPEECH / TEXT</div>
          <h1>Breaking the barrier between <em>silence</em> and conversation.</h1>
          <p className="hero-lede">Silent Interpreter gives Indian Sign Language a voice in the moments that matter — thoughtfully, instantly, and for everyone.</p>
          <div className="button-row">
            <a className="button button-primary" href="#demo">Start live translation <ArrowRight size={17} /></a>
            <a className="button button-ghost" href="#demo"><Play size={16} fill="currentColor" /> Watch demo</a>
          </div>
          <div className="hero-note"><BadgeCheck size={17} /> Designed with accessibility at the center</div>
        </div>

        <div className="hero-visual reveal reveal-delay" aria-label="Simulated translation pipeline">
          <div className="visual-orbit orbit-one" /><div className="visual-orbit orbit-two" />
          <div className="preview-card">
            <div className="preview-top"><DemoPill /><span className="preview-live"><span /> Live preview</span></div>
            <div className="camera-frame">
              <div className="frame-corner corner-tl" /><div className="frame-corner corner-tr" /><div className="frame-corner corner-bl" /><div className="frame-corner corner-br" />
              <div className="hand-illustration" aria-hidden="true"><span className="finger finger-one" /><span className="finger finger-two" /><span className="finger finger-three" /><span className="finger finger-four" /><span className="palm" /></div>
              <span className="scan-line" />
              <div className="camera-caption"><ScanFace size={14} /> Gesture input detected</div>
            </div>
            <div className="pipeline">
              {previewLabels.map((label, index) => { const Icon = previewIcons[index]; return <div key={label} className={`pipeline-step ${previewStep === index ? 'active' : ''}`}><div className="pipeline-icon"><Icon size={17} /></div><span>{label}</span>{index < previewLabels.length - 1 && <span className="pipeline-arrow"><ArrowDown size={13} /></span>}</div> })}
            </div>
            <div className="preview-output"><div><span className="micro-label">CURRENT OUTPUT</span><strong>{previewStep > 3 ? 'Speaking clearly' : previewStep > 2 ? 'नमस्ते' : 'Hello'}</strong></div><div className="audio-bars" aria-hidden="true">{[1, 2, 3, 4, 5, 6, 7].map((bar) => <i key={bar} style={{ animationDelay: `${bar * 0.08}s` }} />)}</div></div>
          </div>
          <div className="float-tag float-tag-one"><Sparkles size={15} /> Context-aware</div>
          <div className="float-tag float-tag-two"><span className="green-check"><Check size={12} /></span> 94% confidence</div>
        </div>
      </section>

      <section className="trust-strip"><span>MADE FOR MORE HUMAN MOMENTS</span><div><span>Healthcare</span><span>Education</span><span>Community</span><span>Everyday life</span></div></section>

      <section className="section-pad impact-section" aria-labelledby="impact-title">
        <div className="section-intro"><div className="eyebrow">WHY THIS MATTERS</div><h2 id="impact-title">Communication should never be a privilege.</h2><p>Millions of people sign every day. Silent Interpreter is a small step toward a world where everyone gets to be understood on their own terms.</p></div>
        <div className="impact-grid"><div className="impact-card impact-large"><div className="impact-number">63<span>m+</span></div><p>people in India live with hearing loss. Access to communication is access to opportunity.</p><span className="impact-icon"><Users size={21} /></span></div><div className="impact-card"><HeartPulse size={21} /><strong>Healthcare</strong><span>Make appointments and urgent care more accessible.</span></div><div className="impact-card"><BookOpen size={21} /><strong>Education</strong><span>Help classrooms become more inclusive places to learn.</span></div><div className="impact-card"><Languages size={21} /><strong>Daily conversations</strong><span>Turn small exchanges into moments of belonging.</span></div></div>
      </section>

      <section className="section-pad workflow-section" id="workflow" aria-labelledby="workflow-title"><div className="section-kicker"><span>HOW IT WORKS</span><span className="kicker-rule" /></div><div className="workflow-heading"><h2 id="workflow-title">From gesture to <em>meaning.</em></h2><p>A simple, transparent flow designed to keep the person — not the technology — at the center.</p></div><div className="workflow-line">{[['01', 'Camera', MonitorPlay], ['02', 'MediaPipe', ScanFace], ['03', 'Recognition', BrainCircuit], ['04', 'Translation', Languages], ['05', 'Speech output', Volume2]].map(([number, label, Icon], index) => { const WorkflowIcon = Icon as typeof MonitorPlay; return <div className="workflow-step" key={label as string}><div className="workflow-node"><span>{number as string}</span><WorkflowIcon size={22} /></div><strong>{label as string}</strong>{index < 4 && <span className="workflow-connector" />}</div> })}</div></section>

      <section className="section-pad features-section" id="features" aria-labelledby="features-title"><div className="section-intro centered"><div className="eyebrow">BUILT FOR THE REAL WORLD</div><h2 id="features-title">Quietly powerful. <em>Deeply human.</em></h2><p>Everything you need to make communication feel more natural — and nothing you don’t.</p></div><div className="feature-grid">{features.map(([title, Icon, copy], index) => { const FeatureIcon = Icon; return <article className={`feature-card ${index === 0 ? 'feature-highlight' : ''}`} key={title}><div className="feature-icon"><FeatureIcon size={20} /></div><h3>{title}</h3><p>{copy}</p><ArrowRight className="feature-arrow" size={17} /></article> })}</div></section>

      <section className="section-pad demo-section" id="demo" aria-labelledby="demo-title"><div className="demo-heading"><div><div className="eyebrow">TRY THE EXPERIENCE</div><h2 id="demo-title">A glimpse of what’s <em>possible.</em></h2><p>This is a front-end simulation of the experience. No camera, model, or backend is connected yet.</p></div><DemoPill /></div><div className="demo-panel"><div className="demo-camera"><div className="demo-camera-top"><span><span className={`status-light ${isTranslating ? 'status-active' : ''}`} /> {isTranslating ? 'Translation active' : 'Ready to translate'}</span><span className="camera-time">00:0{isTranslating ? '8' : '0'}</span></div><div className="demo-camera-view"><div className="view-grid" /><div className="demo-hand"><span className="finger finger-one" /><span className="finger finger-two" /><span className="finger finger-three" /><span className="finger finger-four" /><span className="palm" /></div>{isTranslating && <div className="recognition-box"><ScanFace size={13} /> ISL gesture</div>}<div className="demo-camera-label">Camera preview is simulated</div></div><div className="demo-controls"><button className={`button ${isTranslating ? 'button-stop' : 'button-primary'}`} onClick={() => setIsTranslating(!isTranslating)}>{isTranslating ? <><Square size={15} fill="currentColor" /> Stop</> : <><Radio size={16} /> Start translation</>}</button><div className="language-toggle" role="group" aria-label="Translation language"><button className={language === 'english' ? 'selected' : ''} onClick={() => setLanguage('english')}>EN</button><button className={language === 'hindi' ? 'selected' : ''} onClick={() => setLanguage('hindi')}>HI</button></div></div></div><div className="demo-result"><div className="result-top"><span className="micro-label">LIVE TRANSCRIPT</span><span className="confidence"><span>Confidence</span><strong>{isTranslating ? '94%' : '—'}</strong><span className="confidence-track"><i style={{ width: isTranslating ? '94%' : '8%' }} /></span></span></div><div className={`transcript ${isTranslating ? 'transcript-active' : ''}`}><span className="transcript-mark">“</span><strong>{isTranslating ? transcript : 'Start the simulation to see a translation.'}</strong><span className="transcript-mark transcript-end">”</span></div><div className="result-bottom"><div className="speech-state"><button className="play-button" aria-label={isSpeaking ? 'Pause speech' : 'Play speech'} onClick={() => setIsSpeaking(!isSpeaking)}>{isSpeaking ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" />}</button><span>{isSpeaking ? 'Playing speech...' : 'Play speech preview'}</span></div><div className={`waveform ${isSpeaking ? 'waveform-playing' : ''}`} aria-label="Speech waveform">{[2, 5, 8, 12, 16, 11, 7, 4, 9, 14, 6, 3, 10, 16, 8, 4, 12, 6].map((height, index) => <i key={index} style={{ height: `${height + 5}px`, animationDelay: `${index * 0.05}s` }} />)}</div></div></div></div></section>

      <section className="section-pad upload-section" aria-labelledby="upload-title"><div className="upload-copy"><div className="eyebrow">VIDEO TO TEXT</div><h2 id="upload-title">Bring a conversation <em>with you.</em></h2><p>Upload a short clip to explore the future of accessible video interpretation.</p><div className="upload-list"><span><Check size={14} /> Local-only demo flow</span><span><Check size={14} /> No file leaves your device</span><span><Check size={14} /> No model connected</span></div></div><div className="upload-card" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); handleFile(event.dataTransfer.files[0]) }}><input ref={uploadRef} type="file" accept="video/*" hidden onChange={(event) => handleFile(event.target.files?.[0])} />{uploadState === 'idle' && <button className="drop-zone" onClick={() => uploadRef.current?.click()}><span className="upload-icon"><Upload size={22} /></span><strong>Drop a video here</strong><span>or click to browse · MP4, MOV up to 50MB</span><small><DemoPill /></small></button>}{uploadState === 'processing' && <div className="upload-status"><div className="processing-ring"><Sparkles size={22} /></div><strong>Processing simulation...</strong><span>{fileName || 'your video'} is being analyzed locally</span><div className="progress-track"><i /></div></div>}{uploadState === 'ready' && <div className="upload-status ready"><div className="processing-ring"><Check size={22} /></div><strong>Demo processed successfully</strong><span>Recognized text: “Hello, welcome.”</span><div className="upload-actions"><button className="button button-primary" disabled><Download size={15} /> Download transcript</button><button className="reset-link" onClick={() => { setUploadState('idle'); setFileName('') }}>Try another</button></div></div>}</div></section>

      <section className="section-pad datasets-section" id="datasets" aria-labelledby="datasets-title"><div className="section-heading-row"><div><div className="eyebrow">RESEARCH ROOTS</div><h2 id="datasets-title">Built on open <em>knowledge.</em></h2></div><p>Exploring the datasets helping researchers teach machines the richness of Indian Sign Language.</p></div><div className="dataset-grid">{datasets.map(([name, purpose, size, usage]) => <article className="dataset-card" key={name}><div className="dataset-top"><span className="dataset-mark">{name.slice(0, 2)}</span><span className="dataset-size">{size}</span></div><h3>{name}</h3><p>{purpose}</p><div className="dataset-foot"><span>{usage}</span><ArrowUpRight size={15} /></div></article>)}</div></section>

      <section className="section-pad roadmap-section" id="roadmap" aria-labelledby="roadmap-title"><div className="roadmap-intro"><div className="eyebrow">THE ROAD AHEAD</div><h2 id="roadmap-title">The future is <em>fluent.</em></h2><p>We’re starting with a single bridge. The destination is a world where language never limits connection.</p><a className="text-link" href="#demo">Join the journey <ArrowRight size={16} /></a></div><div className="roadmap-list">{roadmap.map(([number, title, copy], index) => <div className={`roadmap-item ${index === 0 ? 'roadmap-current' : ''}`} key={number}><span className="roadmap-number">{number}</span><div><h3>{title}{index === 0 && <span className="current-pill">Now</span>}</h3><p>{copy}</p></div><ChevronDown className="roadmap-chevron" size={17} /></div>)}</div></section>

      <footer className="footer"><div className="footer-main"><a href="#top" className="brand"><Mark small /><span>Silent Interpreter</span></a><p>A more thoughtful way to be understood.</p><div className="footer-links"><a href="#features">Features</a><a href="#datasets">Research</a><a href="#demo">Contact</a><a href="#top">Accessibility</a><a href="https://github.com" target="_blank" rel="noreferrer"><Activity size={15} /> GitHub</a></div></div><div className="footer-bottom"><span>Built for hackathons, with care.</span><span>© 2026 Silent Interpreter</span></div></footer>
    </main>
  )
}
