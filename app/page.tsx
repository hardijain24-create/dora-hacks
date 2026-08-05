'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  ArrowDown,
  ArrowRight,
  ArrowUpRight,
  BriefcaseBusiness,
  Building2,
  Check,
  ChevronDown,
  CircleDot,
  CloudUpload,
  Coffee,
  GraduationCap,
  HeartPulse,
  Landmark,
  MapPin,
  Menu,
  MonitorPlay,
  Pause,
  Play,
  Radio,
  ScanFace,
  School,
  Send,
  ShoppingBag,
  Sparkles,
  Square,
  TrainFront,
  Upload,
  Volume2,
  Waves,
  X,
  Plane,
} from 'lucide-react'

const contexts = [
  { id: 'hospital', label: 'Hospital', icon: HeartPulse, gesture: 'Need an appointment', english: 'I need help finding my appointment.', hindi: 'मुझे अपनी अपॉइंटमेंट ढूंढने में मदद चाहिए।', confidence: 96 },
  { id: 'restaurant', label: 'Restaurant', icon: Coffee, gesture: 'Order request', english: 'Could I have the menu, please?', hindi: 'क्या मुझे मेनू मिल सकता है?', confidence: 94 },
  { id: 'airport', label: 'Airport', icon: Plane, gesture: 'Travel question', english: 'Which gate is my flight leaving from?', hindi: 'मेरी उड़ान किस गेट से जा रही है?', confidence: 92 },
  { id: 'metro', label: 'Metro', icon: TrainFront, gesture: 'Direction request', english: 'Which platform goes to the city centre?', hindi: 'शहर के केंद्र के लिए कौन सा प्लेटफॉर्म है?', confidence: 91 },
  { id: 'classroom', label: 'Classroom', icon: GraduationCap, gesture: 'Learning question', english: 'Could you explain this lesson again?', hindi: 'क्या आप यह पाठ फिर से समझा सकते हैं?', confidence: 95 },
  { id: 'office', label: 'Office', icon: BriefcaseBusiness, gesture: 'Meeting thought', english: 'I have an idea for the next meeting.', hindi: 'अगली बैठक के लिए मेरे पास एक विचार है।', confidence: 93 },
  { id: 'shopping', label: 'Shopping', icon: ShoppingBag, gesture: 'Product question', english: 'Is this available in another size?', hindi: 'क्या यह दूसरे आकार में उपलब्ध है?', confidence: 90 },
  { id: 'bank', label: 'Bank', icon: Landmark, gesture: 'Account question', english: 'I would like to update my details.', hindi: 'मैं अपनी जानकारी अपडेट करना चाहता हूँ।', confidence: 89 },
] as const

const datasets = [
  { name: 'INCLUDE', meta: '4,287 videos', purpose: 'Isolated ISL words', role: 'A clear baseline for recognizing individual gestures.', mark: 'IN' },
  { name: 'ISLTranslate', meta: '31,000+ samples', purpose: 'Continuous sentence translation', role: 'Teaches models how meaning flows across sequences.', mark: 'IS' },
  { name: 'ISL-CSLTR', meta: '5,000 sentences', purpose: 'Continuous sign recognition', role: 'Supports temporal modeling and natural signing.', mark: 'CS' },
  { name: 'CISLR', meta: '700+ glosses', purpose: 'Large-vocabulary ISL', role: 'Adds real-world breadth to the research desk.', mark: 'CI' },
  { name: 'ISL-Fingerspelling', meta: '36 classes', purpose: 'Alphabet and numbers', role: 'Helps spell the words that gesture vocabularies miss.', mark: 'FS' },
]

const pipeline = ['Camera', 'MediaPipe', 'Landmarks', 'Sliding window', 'LSTM', 'English', 'IndicTrans2', 'Hindi', 'Speech', 'Replay']
const roadmap = ['Hackathon MVP', 'Continuous ISL', 'Transformer Models', 'Offline Edge AI', '3D Avatar', 'Smart Glasses', 'Universal Communication']
const featureNotes = [
  ['Continuous signing', 'Understand the rhythm between gestures, not only isolated signs.'],
  ['English + Hindi', 'Let each conversation choose the language that feels most natural.'],
  ['Offline first', 'A future direction for private, dependable interpretation at the edge.'],
  ['Adaptive vocabulary', 'Make space for the words, people, and places in your world.'],
  ['Conversation replay', 'Return to a simulated exchange with its original context intact.'],
  ['Sentence understanding', 'Move from literal gestures toward useful, human meaning.'],
  ['MediaPipe optimized', 'A research path grounded in accessible computer vision tools.'],
  ['AI4Bharat compatible', 'Designed to leave room for India-first language technology.'],
]

type Context = (typeof contexts)[number]
type Replay = { time: string; gesture: string; english: string; hindi: string; confidence: number }

function Mark() {
  return <span className="mark" aria-hidden="true"><span /><span /><span /></span>
}

function DemoLabel({ children = 'Demo Simulation' }: { children?: string }) {
  return <span className="demo-label"><CircleDot size={11} /> {children}</span>
}

function SectionLabel({ children }: { children: string }) {
  return <p className="section-label"><span />{children}</p>
}

function TabletScene({ context, language, isRunning, confidence, onSpeak, speaking }: { context: Context; language: 'english' | 'hindi'; isRunning: boolean; confidence: number; onSpeak: () => void; speaking: boolean }) {
  const Icon = context.icon
  return (
    <div className="tablet-scene" aria-label="Simulated Indian Sign Language conversation preview">
      <div className="tablet-shadow" />
      <div className="tablet">
        <div className="tablet-camera"><span /><span /><span /></div>
        <div className="tablet-screen">
          <div className="screen-top"><span><span className={`signal ${isRunning ? 'signal-on' : ''}`} /> {isRunning ? 'tracking gesture' : 'ready to explore'}</span><DemoLabel /></div>
          <div className="screen-video">
            <div className="video-grid" /><div className="signer" aria-hidden="true"><i className="head" /><i className="body" /><i className="arm arm-left" /><i className="arm arm-right" /><i className="hand" /></div>
            {isRunning && <span className="landmark landmark-a" />}{isRunning && <span className="landmark landmark-b" />}{isRunning && <span className="landmark landmark-c" />}
            <div className="tracking-box"><ScanFace size={13} /> {context.gesture}</div>
          </div>
          <div className="screen-result"><div className="result-kicker"><Icon size={13} /> {context.label} context</div><p>{language === 'english' ? context.english : context.hindi}</p><div className="result-bottom"><span className="confidence-mini">{confidence}% confidence</span><button onClick={onSpeak} aria-label={speaking ? 'Stop simulated speech' : 'Play simulated speech'} className={`speak-button ${speaking ? 'speaking' : ''}`}>{speaking ? <Pause size={14} /> : <Volume2 size={14} />} {speaking ? 'playing' : 'speak'}</button></div></div>
          <div className="screen-wave" aria-hidden="true">{Array.from({ length: 26 }).map((_, i) => <i key={i} style={{ height: `${10 + ((i * 17) % 28)}px`, animationDelay: `${i * 0.035}s` }} />)}</div>
        </div>
      </div>
      <span className="depth-note depth-note-one"><Sparkles size={13} /> context-aware</span><span className="depth-note depth-note-two"><MapPin size={13} /> {context.label}</span>
    </div>
  )
}

export default function Page() {
  const [selectedId, setSelectedId] = useState<Context['id']>('hospital')
  const [language, setLanguage] = useState<'english' | 'hindi'>('english')
  const [isRunning, setIsRunning] = useState(false)
  const [speaking, setSpeaking] = useState(false)
  const [activeDataset, setActiveDataset] = useState<string | null>(null)
  const [uploadState, setUploadState] = useState<'idle' | 'processing' | 'ready'>('idle')
  const [fileName, setFileName] = useState('')
  const [replays, setReplays] = useState<Replay[]>([])
  const [replayPlaying, setReplayPlaying] = useState<number | null>(null)
  const [mobileOpen, setMobileOpen] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const activeContext = useMemo(() => contexts.find((item) => item.id === selectedId) ?? contexts[0], [selectedId])
  const confidence = isRunning ? activeContext.confidence : 0

  function selectContext(id: Context['id']) {
    setSelectedId(id)
    if (isRunning) addReplay(contexts.find((item) => item.id === id) ?? activeContext)
  }

  function addReplay(context: Context) {
    const date = new Date()
    setReplays((items) => [{ time: date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), gesture: context.gesture, english: context.english, hindi: context.hindi, confidence: context.confidence }, ...items].slice(0, 5))
  }

  function toggleRunning() {
    const next = !isRunning
    setIsRunning(next)
    if (next) addReplay(activeContext)
  }

  function handleFile(file?: File) {
    if (!file) return
    setFileName(file.name)
    setUploadState('processing')
    window.setTimeout(() => setUploadState('ready'), 1500)
  }

  return (
    <main className="keynote-shell">
      <nav className="keynote-nav" aria-label="Main navigation"><a href="#top" className="brand"><Mark /><span>Silent Interpreter</span></a><button className="mobile-menu" onClick={() => setMobileOpen(!mobileOpen)} aria-label={mobileOpen ? 'Close menu' : 'Open menu'}>{mobileOpen ? <X size={18} /> : <Menu size={18} />}</button><div className={`keynote-links ${mobileOpen ? 'open' : ''}`}><a href="#story" onClick={() => setMobileOpen(false)}>Story</a><a href="#thinking" onClick={() => setMobileOpen(false)}>How it thinks</a><a href="#research" onClick={() => setMobileOpen(false)}>Research</a><a href="#replay" onClick={() => setMobileOpen(false)}>Replay</a><a href="#roadmap" onClick={() => setMobileOpen(false)}>Roadmap</a></div><a href="#live" className="nav-button">Explore demo <ArrowUpRight size={14} /></a></nav>

      <section className="keynote-hero section-wrap" id="top"><div className="spotlight spotlight-left" /><div className="spotlight spotlight-right" /><div className="hero-intro reveal"><SectionLabel>AN OPENING NOTE ON UNDERSTANDING</SectionLabel><h1>Every conversation begins with <em>understanding.</em></h1><p>Silent Interpreter is a quiet bridge between Indian Sign Language and the people, places, and moments that matter.</p><div className="hero-actions"><a href="#live" className="solid-button">Explore the simulation <ArrowRight size={16} /></a><a href="#story" className="soft-link"><Play size={15} fill="currentColor" /> See the story</a></div><DemoLabel>Demo Experience — no AI backend connected yet</DemoLabel></div><TabletScene context={activeContext} language={language} isRunning={isRunning} confidence={confidence} onSpeak={() => setSpeaking(!speaking)} speaking={speaking} /></section>

      <section className="context-section section-wrap" id="story" aria-labelledby="context-title"><div className="context-heading"><div><SectionLabel>THE WORLD AROUND A CONVERSATION</SectionLabel><h2 id="context-title">Meaning changes with <em>place.</em></h2></div><p>Select a setting to move the simulated conversation through an ordinary day. The tablet, transcript, and city below follow along.</p></div><div className="context-layout"><div className="context-choices" role="list" aria-label="Conversation contexts">{contexts.map((context) => { const Icon = context.icon; return <button key={context.id} className={`context-choice ${selectedId === context.id ? 'selected' : ''}`} onClick={() => selectContext(context.id)} aria-pressed={selectedId === context.id}><Icon size={16} /><span>{context.label}</span><ArrowUpRight size={13} /></button> })}</div><div className="context-card"><div className="context-card-top"><DemoLabel /><span>{activeContext.confidence}% simulated confidence</span></div><div className="context-conversation"><span className="conversation-tag">ISL gesture</span><strong>{activeContext.gesture}</strong><ArrowDown size={17} /><span className="conversation-tag">{language === 'english' ? 'English' : 'Hindi'} meaning</span><p>{language === 'english' ? activeContext.english : activeContext.hindi}</p></div><div className="context-card-footer"><button className="small-button" onClick={() => setLanguage(language === 'english' ? 'hindi' : 'english')}><span className="lang-dot" /> Switch to {language === 'english' ? 'Hindi' : 'English'}</button><span>Local state only</span></div></div></div></section>

      <section className="thinking-section" id="thinking" aria-labelledby="thinking-title"><div className="section-wrap"><div className="thinking-head"><div><SectionLabel>THE QUIET MACHINERY BEHIND THE MOMENT</SectionLabel><h2 id="thinking-title">How it <em>thinks.</em></h2></div><p>The technology is a supporting character. This progressive view is a future architecture story, not a connected backend.</p></div><div className="neural-stage"><div className="neural-glow" /><svg className="neural-lines" viewBox="0 0 1100 290" aria-hidden="true"><path d="M60 145 C170 30 210 260 330 145 S490 30 560 145 S720 260 800 145 S970 30 1040 145" /><path d="M60 145 C180 230 225 50 330 145 S480 240 560 145 S710 50 800 145 S950 240 1040 145" /></svg><div className="neural-nodes">{pipeline.map((item, index) => <div className="neural-node" key={item} style={{ animationDelay: `${index * 0.1}s` }}><span className="node-core" /><strong>{item}</strong><small>{index < 5 ? 'vision layer' : index < 8 ? 'language layer' : 'human layer'}</small></div>)}</div></div><div className="pipeline-caption"><DemoLabel /><span>Future inference pipeline — simulated visualization only</span></div></div></section>

      <section className="city-section section-wrap" aria-labelledby="city-title"><div className="city-copy"><SectionLabel>A SMALL CITY OF POSSIBILITIES</SectionLabel><h2 id="city-title">Communication lives <em>everywhere.</em></h2><p>Explore a few of the places where a shared language can turn a task into an exchange, and an exchange into belonging.</p><DemoLabel /></div><div className="city-scene" role="img" aria-label="Illustrated city with interactive simulated communication hotspots"><div className="city-sky" /><div className="sun" /><div className="city-buildings"><i /><i /><i /><i /><i /><i /><i /></div><div className="city-road" /><div className="city-hotspots">{['hospital', 'airport', 'restaurant', 'metro', 'bank', 'office', 'classroom'].map((id, index) => { const item = contexts.find((context) => context.id === id) ?? contexts[0]; const Icon = item.icon; return <button key={id} className={`city-hotspot hotspot-${index + 1} ${selectedId === id ? 'selected' : ''}`} onClick={() => selectContext(item.id)} aria-label={`Select ${item.label} city context`}><Icon size={14} /><span>{item.label}</span></button> })}</div><div className="city-caption"><MapPin size={14} /> {activeContext.label}: {activeContext.gesture}</div></div></section>

      <section className="live-section section-wrap" id="live" aria-labelledby="live-title"><div className="live-heading"><div><SectionLabel>A MOMENT YOU CAN TRY</SectionLabel><h2 id="live-title">The live mode <em>within reach.</em></h2><p>Start a local simulation to see gesture recognition, language output, confidence, and speech move together.</p></div><DemoLabel>Demo Simulation</DemoLabel></div><div className="live-panel"><div className="camera-panel"><div className="camera-panel-top"><span><span className={`signal ${isRunning ? 'signal-on' : ''}`} /> {isRunning ? 'gesture recognition active' : 'camera simulation ready'}</span><span>00:0{isRunning ? '8' : '0'}</span></div><div className="camera-stage"><div className="stage-grid" /><div className="signer large"><i className="head" /><i className="body" /><i className="arm arm-left" /><i className="arm arm-right" /><i className="hand" /></div>{isRunning && <div className="recognition-chip"><ScanFace size={13} /> {activeContext.gesture}</div>}<span className="camera-footnote">No webcam connected · visual only</span></div><div className="camera-controls"><button className={`solid-button ${isRunning ? 'stop-button' : ''}`} onClick={toggleRunning}>{isRunning ? <><Square size={15} fill="currentColor" /> Stop</> : <><Radio size={15} /> Start translation</>}</button><div className="language-toggle" role="group" aria-label="Output language"><button className={language === 'english' ? 'active' : ''} onClick={() => setLanguage('english')}>EN</button><button className={language === 'hindi' ? 'active' : ''} onClick={() => setLanguage('hindi')}>HI</button></div></div></div><div className="live-output"><div className="output-top"><span className="micro-label">SIMULATED TRANSCRIPT</span><span className="confidence-value">{isRunning ? `${confidence}% confidence` : 'awaiting gesture'}</span></div><div className="output-quote"><span className="quote-mark">“</span><p>{isRunning ? (language === 'english' ? activeContext.english : activeContext.hindi) : 'Start the simulation to see a conversation appear.'}</p></div><div className="meter"><span style={{ width: `${confidence}%` }} /></div><div className="waveform">{Array.from({ length: 34 }).map((_, i) => <i key={i} style={{ height: `${8 + ((i * 13) % 22)}px`, animationDelay: `${i * 0.04}s` }} />)}</div><div className="output-bottom"><button className={`small-button ${speaking ? 'small-button-active' : ''}`} onClick={() => setSpeaking(!speaking)} disabled={!isRunning}>{speaking ? <Pause size={14} /> : <Volume2 size={14} />} {speaking ? 'Stop speech' : 'Play speech'}</button><span>{speaking ? 'Playing simulated speech...' : 'Web Speech API · future connection'}</span></div></div></div></section>

      <section className="city-story section-wrap" aria-labelledby="story-architecture-title"><div className="architecture-card"><div className="architecture-copy"><SectionLabel>A FUTURE CONNECTION</SectionLabel><h2 id="story-architecture-title">The path from a hand to a <em>voice.</em></h2><p>When the real system arrives, this is the story it will tell: visual landmarks become a sentence, then a language, then a moment someone can hear.</p><DemoLabel>Architecture Story — not connected</DemoLabel></div><div className="architecture-timeline">{pipeline.map((item, index) => <div className="architecture-step" key={item}><span className="architecture-dot">{String(index + 1).padStart(2, '0')}</span><strong>{item}</strong>{index < pipeline.length - 1 && <span className="architecture-connector" />}</div>)}</div></div></section>

      <section className="research-section section-wrap" id="research" aria-labelledby="research-title"><div className="research-heading"><div><SectionLabel>RESEARCH DESK</SectionLabel><h2 id="research-title">Open knowledge, <em>carefully held.</em></h2></div><p>These folders represent the research roots that can help teach a future interpreter the depth of Indian Sign Language.</p></div><div className="research-desk">{datasets.map((dataset) => { const open = activeDataset === dataset.name; return <article className={`research-folder ${open ? 'open' : ''}`} key={dataset.name}><button className="folder-trigger" onClick={() => setActiveDataset(open ? null : dataset.name)} aria-expanded={open}><span className="folder-tab">{dataset.mark}</span><span><strong>{dataset.name}</strong><small>{dataset.meta}</small></span><ChevronDown size={17} /></button>{open && <div className="folder-content"><p>{dataset.purpose}</p><span>{dataset.role}</span><div className="folder-bars"><i /><i /><i /><i /><i /></div></div>}</article> })}</div></section>

      <section className="upload-section section-wrap" aria-labelledby="upload-title"><div className="upload-copy"><SectionLabel>DROP A MEMORY</SectionLabel><h2 id="upload-title">Let a video become a <em>timeline.</em></h2><p>Choose a clip to explore the shape of a future upload flow. Nothing leaves this page, and every result is simulated.</p><ul><li><Check size={14} /> local-only processing</li><li><Check size={14} /> confidence and waveform preview</li><li><Check size={14} /> transcript download preview</li></ul></div><div className="upload-experience" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); handleFile(event.dataTransfer.files[0]) }}><input ref={fileRef} type="file" accept="video/*" hidden onChange={(event) => handleFile(event.target.files?.[0])} />{uploadState === 'idle' && <button className="drop-memory" onClick={() => fileRef.current?.click()}><CloudUpload size={28} /><strong>Drop a video here</strong><span>or browse for a local clip</span><DemoLabel /></button>}{uploadState === 'processing' && <div className="upload-progress"><Sparkles size={24} /><strong>Building a simulated timeline...</strong><span>{fileName}</span><div className="progress-line"><i /></div><div className="processing-steps"><span className="done"><Check size={12} /> gesture chips</span><span>transcript</span><span>speech preview</span></div></div>}{uploadState === 'ready' && <div className="upload-ready"><div className="upload-ready-top"><span className="ready-icon"><Check size={17} /></span><div><strong>Timeline ready</strong><span>Demo Simulation · local result</span></div><button onClick={() => { setUploadState('idle'); setFileName('') }} aria-label="Reset upload simulation"><X size={16} /></button></div><div className="upload-timeline"><span /><span /><span /><span /><span /></div><div className="upload-transcript"><small>RECOGNIZED TEXT</small><p>{activeContext.english}</p><div className="upload-wave">{Array.from({ length: 25 }).map((_, i) => <i key={i} style={{ height: `${8 + ((i * 11) % 22)}px` }} />)}</div></div><button className="download-button" disabled><Upload size={14} /> Download transcript preview</button></div>}</div></section>

      <section className="replay-section section-wrap" id="replay" aria-labelledby="replay-title"><div className="replay-heading"><div><SectionLabel>CONVERSATION REPLAY</SectionLabel><h2 id="replay-title">Nothing meaningful has to <em>disappear.</em></h2></div><p>Every context you explore can stay in this session as a small, local memory. It is a replay of the demo—not a stored conversation.</p></div><div className="replay-list">{replays.length === 0 && <div className="empty-replay"><Waves size={22} /><p>Start a context simulation to create your first replay.</p><DemoLabel /></div>}{replays.map((replay, index) => <article className="replay-item" key={`${replay.time}-${index}`}><span className="replay-time">{replay.time}</span><div className="replay-line"><span className="replay-node" /><span /></div><div className="replay-content"><div className="replay-meta"><strong>{replay.gesture}</strong><span>{replay.confidence}% confidence</span></div><p>{language === 'english' ? replay.english : replay.hindi}</p><small>{replay.hindi}</small></div><button className="replay-button" onClick={() => { setReplayPlaying(index); window.setTimeout(() => setReplayPlaying(null), 1000) }} aria-label={`Replay ${replay.gesture}`}>{replayPlaying === index ? <Pause size={14} /> : <Play size={14} fill="currentColor" />}</button></article>)}</div><div className="replay-note"><DemoLabel /> <span>Replay controls change local UI state only.</span></div></section>

      <section className="features-story section-wrap" aria-labelledby="features-title"><div className="features-story-heading"><SectionLabel>WHY IT FEELS DIFFERENT</SectionLabel><h2 id="features-title">Technology with a softer <em>edge.</em></h2></div><div className="feature-notes">{featureNotes.map(([title, copy], index) => <article key={title} className={index === 0 ? 'feature-note lead' : 'feature-note'}><span>0{index + 1}</span><h3>{title}</h3><p>{copy}</p><ArrowUpRight size={16} /></article>)}</div></section>

      <section className="roadmap-section section-wrap" id="roadmap" aria-labelledby="roadmap-title"><div className="roadmap-copy"><SectionLabel>THE ROAD AHEAD</SectionLabel><h2 id="roadmap-title">From a first step to <em>universal communication.</em></h2><p>A staircase, not a promise: each level is a direction for future research and care.</p><DemoLabel>Roadmap Simulation</DemoLabel></div><div className="staircase" aria-label="Silent Interpreter roadmap">{roadmap.map((item, index) => <div className={`stair ${index === 0 ? 'current' : ''}`} key={item} style={{ '--step': index } as React.CSSProperties}><span>{String(index + 1).padStart(2, '0')}</span><strong>{item}</strong>{index === 0 && <small>now</small>}</div>)}</div></section>

      <footer className="keynote-footer"><div className="section-wrap footer-inner"><a href="#top" className="brand"><Mark /><span>Silent Interpreter</span></a><p>Built for the moments when being understood matters.</p><div className="footer-links"><a href="https://github.com" target="_blank" rel="noreferrer">GitHub <ArrowUpRight size={13} /></a><a href="#live">Contact</a><a href="#top">Accessibility</a></div><span className="footer-end">Built for hackathons · 2026</span></div></footer>
    </main>
  )
}
