'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowDown, ArrowRight, ArrowUpRight, Check, ChevronDown, CircleDot,
  CloudUpload, Coffee, GraduationCap, HeartPulse, Menu, Mic, Pause, Plane,
  Play, Radio, ScanFace, Sparkles, Square, TrainFront, Upload, Video,
  Volume2, Waves, X,
} from 'lucide-react'

const contexts = [
  { id: 'hospital' as const, label: 'Hospital', icon: HeartPulse, emoji: '🏥', gesture: 'Open Palm', english: 'I need help finding my appointment.', hindi: 'मुझे अपनी अपॉइंटमेंट में मदद चाहिए।', confidence: 96 },
  { id: 'restaurant' as const, label: 'Restaurant', icon: Coffee, emoji: '☕', gesture: 'Point & Cup', english: 'One coffee, please.', hindi: 'एक कॉफ़ी दे दीजिए।', confidence: 94 },
  { id: 'airport' as const, label: 'Airport', icon: Plane, emoji: '✈️', gesture: 'Directional Sign', english: 'Where is Gate 12?', hindi: 'गेट 12 कहाँ है?', confidence: 92 },
  { id: 'metro' as const, label: 'Metro', icon: TrainFront, emoji: '🚇', gesture: 'Two Fingers', english: 'Which platform goes to the city centre?', hindi: 'शहर के केंद्र के लिए कौन सा प्लेटफॉर्म है?', confidence: 91 },
  { id: 'classroom' as const, label: 'Classroom', icon: GraduationCap, emoji: '📚', gesture: 'Raised Hand', english: 'Could you explain this lesson again?', hindi: 'क्या आप यह पाठ फिर से समझा सकते हैं?', confidence: 95 },
]

const UNIVERSE_LAYOUT = [
  { idx: 2, cx: 50, cy: 13 }, { idx: 1, cx: 84, cy: 38 }, { idx: 4, cx: 71, cy: 78 },
  { idx: 3, cx: 29, cy: 78 }, { idx: 0, cx: 16, cy: 38 },
]

const PIPELINE_SIMPLE = [
  { step: '01', label: 'Gesture', sub: 'Your hand speaks first.', highlight: false },
  { step: '02', label: 'Understanding', sub: 'MediaPipe reads each landmark.', highlight: false },
  { step: '03', label: 'Translation', sub: 'LSTM maps the sequence to meaning.', highlight: true },
  { step: '04', label: 'Voice', sub: 'IndicTrans2 speaks it in your language.', highlight: false },
]

const PIPELINE_FULL = [
  { label: 'Camera', sub: 'input' }, { label: 'MediaPipe', sub: 'vision' },
  { label: 'Landmarks', sub: 'vision' }, { label: 'Sliding Window', sub: 'model' },
  { label: 'LSTM', sub: 'model' }, { label: 'English', sub: 'language' },
  { label: 'IndicTrans2', sub: 'language' }, { label: 'Hindi', sub: 'language' },
  { label: 'Speech API', sub: 'output' }, { label: 'Replay', sub: 'output' },
]

const DATASETS = [
  { name: 'INCLUDE', meta: '4,287 videos', purpose: 'Isolated ISL words', role: 'A clear baseline for recognizing individual gestures.', mark: 'IN' },
  { name: 'ISLTranslate', meta: '31,000+ samples', purpose: 'Continuous sentence translation', role: 'Teaches models how meaning flows across sequences.', mark: 'IS' },
  { name: 'ISL-CSLTR', meta: '5,000 sentences', purpose: 'Continuous sign recognition', role: 'Supports temporal modeling and natural signing.', mark: 'CS' },
  { name: 'CISLR', meta: '700+ glosses', purpose: 'Large-vocabulary ISL', role: 'Adds real-world breadth to the research desk.', mark: 'CI' },
  { name: 'ISL-FS', meta: '36 classes', purpose: 'Alphabet and numbers', role: 'Helps spell the words that gesture vocabularies miss.', mark: 'FS' },
]

const ROADMAP = ['Hackathon MVP', 'Continuous ISL', 'Transformer Models', 'Offline Edge AI', '3D Avatar', 'Smart Glasses', 'Universal Communication']

const LM_PTS: [number, number][] = [
  [50,78],[48,65],[45,55],[44,47],[42,41],[52,44],[53,32],[54,26],[55,21],
  [57,45],[59,31],[60,24],[61,19],[62,47],[64,35],[65,29],[66,25],
  [66,50],[67,41],[67,36],[68,32],
]
const LM_CONN: [number,number][] = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],
]

const FLOAT_WORDS = [
  { word: 'Hello',      x: 82, y: 14, delay: 1.3 },
  { word: 'Understood', x: 88, y: 65, delay: 0.6 },
  { word: 'Language',   x: 74, y: 88, delay: 0.3 },
]

type CtxId = (typeof contexts)[number]['id']
type Ctx = (typeof contexts)[number]
type Replay = { time: string; gesture: string; english: string; hindi: string; confidence: number }

function Mark() {
  return (
    <span className="mark" aria-hidden="true">
      <span /><span /><span />
    </span>
  )
}

function DemoBadge({ children = 'Demo Simulation' }: { children?: string }) {
  return <span className="demo-badge"><CircleDot size={9} /> {children}</span>
}

function SectionLabel({ children }: { children: string }) {
  return <p className="section-label">{children}</p>
}

function DeviceScreen({ ctx, lang, onLang, isRunning, cameraGranted, videoRef, mini = false }: {
  ctx: Ctx; lang: 'english'|'hindi'; onLang: (l:'english'|'hindi')=>void
  isRunning: boolean; cameraGranted: boolean; videoRef: React.RefObject<HTMLVideoElement|null>; mini?: boolean
}) {
  const [flowStep, setFlowStep] = useState(0)
  const timerRef = useRef<ReturnType<typeof setTimeout>[]>([])

  useEffect(() => {
    timerRef.current.forEach(clearTimeout); timerRef.current = []
    if (!isRunning) { setFlowStep(0); return }
    setFlowStep(0)
    ;[1,2,3,4].forEach((step, i) => {
      const t = setTimeout(() => setFlowStep(step), i * 520 + 200)
      timerRef.current.push(t)
    })
    return () => timerRef.current.forEach(clearTimeout)
  }, [ctx.id, isRunning])

  const wBars = useMemo(() => Array.from({ length: 12 }, (_, i) => 18 + ((i*17+5)%56)), [])
  const circumference = 2 * Math.PI * 9

  return (
    <div className="device-frame">
      <div className="device-notch" aria-hidden="true">
        <span className="notch-dot"/><span className="notch-cam"/><span className="notch-dot"/>
      </div>
      <div className="device-screen-bg">
        <div className="device-status">
          <span className="status-left">
            <span className={`sig-dot ${isRunning?'on':''}`}/>
            {isRunning?'tracking gesture':'ready to explore'}
          </span>
          <DemoBadge/>
        </div>
        <div className="device-cam">
          {cameraGranted && <video ref={videoRef} autoPlay muted playsInline aria-hidden="true"/>}
          {!cameraGranted && (
            <div className="signer-wrap" aria-hidden="true">
              <div className="signer"><i className="sh"/><i className="sb"/><i className="sa sa-l"/><i className="sa sa-r"/><i className="shand"/></div>
            </div>
          )}
          <div className="device-cam-grid" aria-hidden="true"/>
          <svg className="device-lm-svg" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            {isRunning && (<>
              <circle className="conf-ring-bg" cx="88" cy="12" r="9"/>
              <circle className="conf-ring" cx="88" cy="12" r="9"
                strokeDasharray={circumference}
                strokeDashoffset={circumference*(1-ctx.confidence/100)}
                style={{transform:'rotate(-90deg)',transformOrigin:'88px 12px'}}
              />
            </>)}
            {LM_CONN.map(([a,b],i) => (
              <line key={i} className={`lm-line ${isRunning&&flowStep>=1?'on':''}`}
                x1={LM_PTS[a][0]} y1={LM_PTS[a][1]} x2={LM_PTS[b][0]} y2={LM_PTS[b][1]}
                style={{transitionDelay:`${i*12}ms`}}
              />
            ))}
            {LM_PTS.map(([x,y],i) => (
              <circle key={i} className={`lm-dot ${isRunning&&flowStep>=1?'on':''}`}
                cx={x} cy={y} r="1.3" style={{transitionDelay:`${i*20}ms`}}
              />
            ))}
          </svg>
          {isRunning && <div className="gesture-chip"><ScanFace size={9}/> {ctx.gesture}</div>}
        </div>
        {!mini && (
          <div className="device-track">
            <span className="track-chip">
              <span className={`sig-dot ${isRunning?'on':''}`} style={{width:5,height:5}}/>
              {isRunning?ctx.gesture:'awaiting gesture'}
            </span>
            <span style={{fontFamily:'var(--font-mono)',fontSize:9}}>{isRunning?'28 fps':'--'}</span>
            <div className="lang-pair" role="group" aria-label="Output language">
              <button className={lang==='english'?'on':''} onClick={()=>onLang('english')}>EN</button>
              <button className={lang==='hindi'?'on':''} onClick={()=>onLang('hindi')}>HI</button>
            </div>
          </div>
        )}
        <div className="device-flow" aria-live="polite">
          {!isRunning && <div className="flow-idle">Start simulation to see translation</div>}
          {isRunning && (<>
            <div className={`flow-row ${flowStep>=1?'show':''}`}>
              <span className="flow-icon">👋</span>
              <div><p className="flow-meta">gesture detected</p><p className="flow-text">{ctx.gesture}</p></div>
            </div>
            {flowStep>=1 && <div className={`flow-arrow ${flowStep>=2?'show':''}`}><ArrowDown size={10}/></div>}
            <div className={`flow-row ${flowStep>=2?'show':''}`} style={{transitionDelay:'.1s'}}>
              <span className="flow-icon">🌐</span>
              <div><p className="flow-meta">english</p><p className="flow-text">{ctx.english.split(' ').slice(0,3).join(' ')}…</p></div>
            </div>
            {flowStep>=2 && <div className={`flow-arrow ${flowStep>=3?'show':''}`}><ArrowDown size={10}/></div>}
            <div className={`flow-row ${flowStep>=3?'show':''}`} style={{transitionDelay:'.2s'}}>
              <span className="flow-icon">🇮🇳</span>
              <div><p className="flow-meta">hindi</p><p className="flow-text">{ctx.hindi.substring(0,14)}…</p></div>
            </div>
            {flowStep>=3 && <div className={`flow-arrow ${flowStep>=4?'show':''}`}><ArrowDown size={10}/></div>}
            <div className={`flow-row ${flowStep>=4?'show':''}`} style={{transitionDelay:'.3s'}}>
              <span className="flow-icon">🔊</span>
              <div>
                <p className="flow-meta">speaking</p>
                <div className="mini-wave">{wBars.map((h,i) => <i key={i} style={{height:h*.28,animationDelay:`${i*.06}s`}}/>)}</div>
              </div>
            </div>
          </>)}
        </div>
        <div className="device-conf">
          <span>Confidence</span>
          <div className="conf-track"><div className="conf-fill" style={{width:isRunning?`${ctx.confidence}%`:'0%'}}/></div>
          <span className="conf-pct">{isRunning?`${ctx.confidence}%`:'--'}</span>
        </div>
      </div>
    </div>
  )
}

function CommunicationUniverse({ selectedId, onSelect, lang }: {
  selectedId: CtxId; onSelect: (id:CtxId)=>void; lang:'english'|'hindi'
}) {
  const [hovered, setHovered] = useState<number|null>(null)
  const [changing, setChanging] = useState(false)
  const ctx = useMemo(() => contexts.find(c=>c.id===selectedId)??contexts[0],[selectedId])
  const wBars = useMemo(() => Array.from({length:8},(_,i)=>10+((i*11+3)%20)),[])

  const handleSelect = useCallback((idx:number) => {
    const newCtx = contexts[UNIVERSE_LAYOUT.find(n=>n.idx===idx)!?.idx??idx]
    if (!newCtx||newCtx.id===selectedId) return
    setChanging(true)
    setTimeout(()=>{onSelect(newCtx.id);setChanging(false)},280)
  },[selectedId,onSelect])

  return (
    <div className="universe-layout">
      <div className="universe-map" role="region" aria-label="Communication universe map">
        <div className="universe-bg-grain" aria-hidden="true"/>
        <svg className="universe-svg" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          {UNIVERSE_LAYOUT.map((node,i) => (
            <g key={i}>
              <line className="u-line-bg" x1="50" y1="50" x2={node.cx} y2={node.cy}/>
              <line className={`u-line ${(hovered===i||contexts[node.idx].id===selectedId)?'lit':''}`} x1="50" y1="50" x2={node.cx} y2={node.cy}/>
            </g>
          ))}
        </svg>
        <div className="universe-center" aria-hidden="true">
          <div className="center-orb">
            <span style={{display:'inline-flex',alignItems:'flex-end',gap:2,width:22,height:22,padding:'3.5px 4.5px'}}>
              <span style={{display:'block',width:3,height:6,borderRadius:3,background:'#fff'}}/>
              <span style={{display:'block',width:3,height:10,borderRadius:3,background:'#fff'}}/>
              <span style={{display:'block',width:3,height:14,borderRadius:3,background:'#fff'}}/>
            </span>
          </div>
          <p className="center-label">SILENT<br/>INTERPRETER</p>
        </div>
        {UNIVERSE_LAYOUT.map((node,i) => {
          const c = contexts[node.idx]; const active = c.id===selectedId
          return (
            <button key={c.id} className={`u-node ${active?'active':''}`}
              style={{left:`${node.cx}%`,top:`${node.cy}%`}}
              onMouseEnter={()=>setHovered(i)} onMouseLeave={()=>setHovered(null)}
              onClick={()=>handleSelect(node.idx)}
              aria-label={`Select ${c.label} context`} aria-pressed={active}
            >
              <span className="u-node-orb">{c.emoji}</span>
              <span className="u-node-label">{c.label}</span>
            </button>
          )
        })}
      </div>
      <div className={`universe-context-card ${changing?'uc-changing':''}`} aria-live="polite">
        <div className="uc-top"><DemoBadge/><span>{ctx.confidence}% confidence</span></div>
        <div className="uc-gesture"><span>{ctx.emoji}</span> {ctx.gesture}</div>
        <p className="uc-english">{lang==='english'?ctx.english:ctx.hindi}</p>
        {lang==='english' && <p className="uc-hindi">{ctx.hindi}</p>}
        <div className="uc-footer">
          <span className="conf-badge"><CircleDot size={8}/> {ctx.confidence}%</span>
          <div className="uc-wave" style={{marginLeft:'auto'}}>
            {wBars.map((h,i)=><i key={i} style={{height:h,width:2,borderRadius:2,background:'var(--accent)',opacity:.6}}/>)}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function Page() {
  const [selectedId, setSelectedId] = useState<CtxId>('hospital')
  const [lang, setLang] = useState<'english'|'hindi'>('english')
  const [isRunning, setIsRunning] = useState(false)
  const [speaking, setSpeaking] = useState(false)
  const [activeDataset, setActiveDataset] = useState<string|null>(null)
  const [uploadState, setUploadState] = useState<'idle'|'processing'|'ready'>('idle')
  const [fileName, setFileName] = useState('')
  const [replays, setReplays] = useState<Replay[]>([])
  const [replayPlaying, setReplayPlaying] = useState<number|null>(null)
  const [cameraGranted, setCameraGranted] = useState(false)
  const [pinned, setPinned] = useState(false)
  const [techExpanded, setTechExpanded] = useState(false)
  const [chipsVisible, setChipsVisible] = useState<boolean[]>([])
  const [mobileOpen, setMobileOpen] = useState(false)

  const fileRef = useRef<HTMLInputElement>(null)
  const heroVideoRef = useRef<HTMLVideoElement|null>(null)
  const liveVideoRef = useRef<HTMLVideoElement|null>(null)

  const ctx = useMemo(()=>contexts.find(c=>c.id===selectedId)??contexts[0],[selectedId])
  const confidence = isRunning ? ctx.confidence : 0
  const waveHeights = useMemo(()=>Array.from({length:34},(_,i)=>8+((i*13)%28)),[])

  useEffect(()=>{
    if (typeof window==='undefined') return
    const dot = document.querySelector<HTMLElement>('.cursor-dot')
    if (!dot) return
    let mx=0,my=0,cx=0,cy=0,raf:number
    const onMove=(e:MouseEvent)=>{
      mx=e.clientX; my=e.clientY
      document.body.style.setProperty('--mx',`${e.clientX}px`)
      document.body.style.setProperty('--my',`${e.clientY}px`)
    }
    function tick(){ cx+=(mx-cx)*.1; cy+=(my-cy)*.1; dot.style.transform=`translate(${cx-20}px,${cy-20}px)`; raf=requestAnimationFrame(tick) }
    raf=requestAnimationFrame(tick)
    window.addEventListener('mousemove',onMove,{passive:true})
    return ()=>{ window.removeEventListener('mousemove',onMove); cancelAnimationFrame(raf) }
  },[])

  useEffect(()=>{
    if (typeof window==='undefined') return
    let lenis:any, rafId:number
    const init=async()=>{
      const [{ default: Lenis }, gsapMod, stMod] = await Promise.all([
        import('lenis'),
        import('gsap'),
        import('gsap/ScrollTrigger'),
      ])
      const { gsap } = gsapMod
      const { ScrollTrigger } = stMod
      gsap.registerPlugin(ScrollTrigger)
      lenis=new Lenis({duration:1.2,easing:(t:number)=>Math.min(1,1.001-Math.pow(2,-10*t)),smoothWheel:true})
      lenis.on('scroll',ScrollTrigger.update)
      function tick(time:number){lenis.raf(time);rafId=requestAnimationFrame(tick)}
      rafId=requestAnimationFrame(tick)
      const headline=document.querySelector('.hero-h1')
      if(headline){
        const words=(headline.textContent??'').trim().split(/\s+/)
        headline.innerHTML=words.map((w,i)=>`<span class="hero-word-wrap"><span class="hero-word">${w}${i<words.length-1?'\u00A0':''}</span></span>`).join('')
        gsap.from('.hero-word',{yPercent:110,duration:0.9,stagger:0.07,ease:'power3.out',delay:0.1})
      }
      gsap.from(['.hero-eyebrow','.hero-orb','.hero-sub','.hero-actions','.hero-note','.scroll-cue'],{opacity:0,y:20,duration:0.8,stagger:0.1,ease:'power3.out',delay:0.5})
      gsap.from('.hero-device-col',{opacity:0,x:36,duration:1.1,ease:'power3.out',delay:0.35})
      gsap.from('.float-word',{opacity:0,duration:1.2,stagger:0.15,ease:'power2.out',delay:0.8})
      ScrollTrigger.create({
        trigger:'.hero',start:'top top',end:'bottom top',
        onUpdate:(self:any)=>{
          const p=self.progress
          gsap.set('.hero-text',{opacity:Math.max(0,1-p*1.9),y:p*-70,scale:1-p*0.045})
          setPinned(p>0.88)
        }
      })
      document.querySelectorAll('.reveal-h').forEach(el=>{
        gsap.from(el,{opacity:0,y:34,duration:0.9,ease:'power3.out',scrollTrigger:{trigger:el,start:'top 88%'}})
      })
      document.querySelectorAll('.reveal-p').forEach(el=>{
        gsap.from(el,{opacity:0,y:18,duration:0.8,ease:'power3.out',scrollTrigger:{trigger:el,start:'top 90%'}})
      })
      gsap.from('.stat-card',{opacity:0,y:44,stagger:0.12,duration:0.85,ease:'power3.out',scrollTrigger:{trigger:'.stats-grid',start:'top 82%'}})
      gsap.from('.universe-map',{opacity:0,y:36,duration:1.0,ease:'power3.out',scrollTrigger:{trigger:'.universe-map',start:'top 82%'}})
      gsap.from('.universe-context-card',{opacity:0,y:22,duration:0.85,ease:'power3.out',delay:0.15,scrollTrigger:{trigger:'.universe-context-card',start:'top 85%'}})
      if(document.querySelector('.demo-story')){
        gsap.set(['.ds-scene-2','.ds-scene-4'],{opacity:0,scale:0.9})
        gsap.set('.ds-scene-3',{opacity:0,scale:0.9,clipPath:'inset(0 100% 0 0)'})
        gsap.set('.ds-scene-1',{opacity:1,scale:1})
        const tl=gsap.timeline({scrollTrigger:{trigger:'.demo-story',pin:true,scrub:1.6,start:'top top',end:'+=320%'}})
        tl
          .to('.ds-waveform i',{scaleY:2.8,stagger:{each:0.025,from:'center'},duration:0.2,ease:'power2.inOut'},0.05)
          .to('.ds-scene-1',{opacity:0,scale:0.93,duration:0.2},0.3)
          .to('.ds-scene-2',{opacity:1,scale:1,duration:0.28},0.3)
          .to('.ds-ring',{rotation:720,duration:0.38,ease:'none'},0.3)
          .to('.ds-scene-2',{opacity:0,scale:0.93,duration:0.2},0.68)
          .to('.ds-scene-3',{opacity:1,scale:1,clipPath:'inset(0 0% 0 0)',duration:0.32},0.68)
          .to('.ds-scene-3',{opacity:0,duration:0.2},1.05)
          .to('.ds-scene-4',{opacity:1,scale:1,duration:0.3},1.05)
          .to('.ds-dot-1',{width:'24px',background:'var(--accent)',duration:0.01},0)
          .to('.ds-dot-2',{width:'24px',background:'var(--accent)',duration:0.01},0.3)
          .to('.ds-dot-3',{width:'24px',background:'var(--accent)',duration:0.01},0.68)
          .to('.ds-dot-4',{width:'24px',background:'var(--accent)',duration:0.01},1.05)
      }
      gsap.from('.live-panel',{opacity:0,y:44,duration:1.0,ease:'power3.out',scrollTrigger:{trigger:'.live-panel',start:'top 82%'}})
      gsap.from('.replay-layout > *',{opacity:0,y:28,stagger:0.15,duration:0.85,ease:'power3.out',scrollTrigger:{trigger:'.replay-layout',start:'top 82%'}})
      gsap.from('.upload-layout > *',{opacity:0,y:28,stagger:0.15,duration:0.85,ease:'power3.out',scrollTrigger:{trigger:'.upload-layout',start:'top 82%'}})
      gsap.from('.tech-step',{opacity:0,scale:0.9,y:18,stagger:0.14,duration:0.85,ease:'power3.out',scrollTrigger:{trigger:'.tech-simple',start:'top 82%'}})
      gsap.from('.r-card',{opacity:0,y:30,rotation:2.5,stagger:0.09,duration:0.85,ease:'power3.out',scrollTrigger:{trigger:'.research-cards',start:'top 82%'}})
      gsap.from('.stair',{opacity:0,y:32,stagger:0.08,duration:0.75,ease:'power3.out',scrollTrigger:{trigger:'.staircase',start:'top 82%'}})
      gsap.from('.cta-headline',{opacity:0,y:54,duration:1.1,ease:'power3.out',scrollTrigger:{trigger:'.cta-section',start:'top 80%'}})
      gsap.from(['.cta-sub','.cta-btn-wrap'],{opacity:0,y:28,stagger:0.15,duration:0.9,ease:'power3.out',scrollTrigger:{trigger:'.cta-section',start:'top 75%'}})
    }
    init()
    return ()=>{
      lenis?.destroy()
      cancelAnimationFrame(rafId)
      if(typeof window!=='undefined'){
        import('gsap/ScrollTrigger').then(({ScrollTrigger})=>{
          ScrollTrigger.getAll().forEach((t:any)=>t.kill())
        }).catch(()=>{})
      }
    }
  },[])

  useEffect(()=>{
    if(typeof window==='undefined') return
    const fns:Array<()=>void>=[]
    const init=async()=>{
      const {gsap}=await import('gsap')
      const els=document.querySelectorAll<HTMLElement>('.btn-primary,.nav-cta,.run-btn,.cta-btn')
      els.forEach(el=>{
        const onMove=(e:MouseEvent)=>{
          const r=el.getBoundingClientRect()
          gsap.to(el,{x:(e.clientX-r.left-r.width/2)*0.22,y:(e.clientY-r.top-r.height/2)*0.22-1,duration:0.35,ease:'power3.out',overwrite:'auto'})
        }
        const onLeave=()=>gsap.to(el,{x:0,y:0,duration:0.65,ease:'elastic.out(1,0.5)',overwrite:'auto'})
        el.addEventListener('mousemove',onMove); el.addEventListener('mouseleave',onLeave)
        fns.push(()=>{el.removeEventListener('mousemove',onMove);el.removeEventListener('mouseleave',onLeave)})
      })
    }
    init()
    return ()=>fns.forEach(f=>f())
  },[])

  useEffect(()=>{
    if(typeof window==='undefined') return
    import('gsap').then(({gsap})=>{
      gsap.from('.uc-english,.uc-hindi',{opacity:0,y:8,duration:0.4,ease:'power3.out'})
    }).catch(()=>{})
  },[selectedId])

  const requestCamera=useCallback(async()=>{
    try{
      const s1=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480,facingMode:'user'}})
      if(heroVideoRef.current){heroVideoRef.current.srcObject=s1;heroVideoRef.current.play()}
      const s2=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480,facingMode:'user'}})
      if(liveVideoRef.current){liveVideoRef.current.srcObject=s2;liveVideoRef.current.play()}
      setCameraGranted(true)
    }catch{}
  },[])

  function addReplay(c:Ctx){
    const t=new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})
    setReplays(r=>[{time:t,gesture:c.gesture,english:c.english,hindi:c.hindi,confidence:c.confidence},...r].slice(0,6))
  }
  function toggleRunning(){const next=!isRunning;setIsRunning(next);if(next)addReplay(ctx)}
  function handleSelect(id:CtxId){setSelectedId(id);if(isRunning)addReplay(contexts.find(c=>c.id===id)??ctx)}

  function handleFile(file?:File){
    if(!file) return
    setFileName(file.name);setUploadState('processing');setChipsVisible([])
    const chips=['Open Palm','Handshake','Point','Fist','Five']
    chips.forEach((_,i)=>setTimeout(()=>setChipsVisible(v=>[...v,true]),400+i*350))
    setTimeout(()=>setUploadState('ready'),2300)
  }

  return (
    <div className="shell">
      <div className="cursor-dot" aria-hidden="true"/>

      <nav className="nav" aria-label="Main navigation">
        <a href="#top" className="brand"><Mark/><span>Silent Interpreter</span></a>
        <div className={`nav-links ${mobileOpen?'nav-links-open':''}`}>
          <a href="#universe" onClick={()=>setMobileOpen(false)}>Universe</a>
          <a href="#live" onClick={()=>setMobileOpen(false)}>Live Demo</a>
          <a href="#technology" onClick={()=>setMobileOpen(false)}>Technology</a>
          <a href="#research" onClick={()=>setMobileOpen(false)}>Research</a>
          <a href="#roadmap" onClick={()=>setMobileOpen(false)}>Roadmap</a>
        </div>
        <a href="#live" className="nav-cta">Try the demo <ArrowRight size={13}/></a>
        <button className="mobile-toggle" onClick={()=>setMobileOpen(!mobileOpen)} aria-label="Toggle menu">
          {mobileOpen?<X size={18}/>:<Menu size={18}/>}
        </button>
      </nav>

      <section className="hero" id="top" aria-labelledby="hero-h1">
        <div className="hero-gradient" aria-hidden="true"/>
        <div className="hero-float-layer" aria-hidden="true">
          {FLOAT_WORDS.map((fw,i)=>(
            <span key={i} className="float-word" style={{left:`${fw.x}%`,top:`${fw.y}%`,animationDelay:`${fw.delay}s`}}>
              {fw.word}
            </span>
          ))}
        </div>
        <div className="hero-text">
          <div className="hero-eyebrow">
            <span className="hero-eyebrow-dot" aria-hidden="true"/>
            Indian Sign Language · Real-Time Translation · Hackathon 2026
          </div>
          <h1 className="hero-h1" id="hero-h1">Communication Should Never Need A Translator.</h1>
          <div className="hero-orb" aria-hidden="true">
            <div className="hero-orb-ring hero-orb-ring-2"/>
            <div className="hero-orb-ring"/>
            <div className="hero-orb-inner"><Mic size={18} strokeWidth={1.8}/></div>
          </div>
          <p className="hero-sub">Silent Interpreter bridges Indian Sign Language and spoken language — in hospitals, restaurants, airports, classrooms, and everywhere in between.</p>
          <div className="hero-actions">
            <button className="btn-primary" onClick={requestCamera} aria-label="Activate webcam">
              <Video size={16}/> Activate camera
            </button>
            <a href="#live" className="btn-ghost"><Play size={15} fill="currentColor"/> Try the demo</a>
          </div>
          <p className="hero-note"><span className="hero-note-dot" aria-hidden="true"/>Demo simulation · No AI backend connected · All processing is simulated</p>
          <div className="scroll-cue" aria-hidden="true"><ArrowDown size={14} className="scroll-cue-arrow"/><span>Scroll to begin the story</span></div>
        </div>
        <div className="hero-device-col" aria-hidden="true">
          <div className={`hero-device-wrap ${pinned?'fading':''}`}>
            <DeviceScreen ctx={ctx} lang={lang} onLang={setLang} isRunning={isRunning} cameraGranted={cameraGranted} videoRef={heroVideoRef}/>
          </div>
          <span className="depth-tag depth-tag-tl"><ScanFace size={11}/> context-aware</span>
          <span className="depth-tag depth-tag-bl"><Sparkles size={11}/> {ctx.label}</span>
          <div className="hero-device-shadow" aria-hidden="true"/>
        </div>
      </section>

      <div className={`device-pinned ${pinned?'show':''}`} aria-label="Pinned device" aria-hidden={!pinned}>
        <DeviceScreen ctx={ctx} lang={lang} onLang={setLang} isRunning={isRunning} cameraGranted={false} videoRef={{current:null}} mini/>
      </div>

      <section className="problem-section" id="problem" aria-labelledby="problem-h2">
        <div className="wrap">
          <div className="problem-inner">
            <div>
              <SectionLabel>THE PROBLEM WE ARE SOLVING</SectionLabel>
              <h2 className="problem-h2 reveal-h" id="problem-h2">One in seventy million.<br/><em>Every single day.</em></h2>
              <p className="problem-body reveal-p">India has the world's largest deaf and hard-of-hearing community. Most face communication barriers at hospitals, schools, and public spaces — not because of their disability, but because of ours.</p>
            </div>
            <div className="stats-grid">
              <div className="stat-card"><p className="stat-num"><em>70M+</em></p><p className="stat-label">deaf and hard-of-hearing people in India — the world's largest such community</p></div>
              <div className="stat-card"><p className="stat-num">300<em>+</em></p><p className="stat-label">unique hand shapes in Indian Sign Language</p></div>
              <div className="stat-card"><p className="stat-num">1<em>in</em>6</p><p className="stat-label">deaf Indians with access to a trained sign language interpreter</p></div>
            </div>
          </div>
        </div>
      </section>

      <section className="universe-section" id="universe" aria-labelledby="universe-h2">
        <div className="wrap">
          <div className="universe-top">
            <div>
              <SectionLabel>FOUND EVERYWHERE YOU LOOK</SectionLabel>
              <h2 id="universe-h2" className="reveal-h">Every place is a new<br/><em>conversation.</em></h2>
            </div>
            <p className="reveal-p">Hover a location to change the simulated conversation. Silent Interpreter adapts to the context — not just the gesture.</p>
          </div>
          <CommunicationUniverse selectedId={selectedId} onSelect={handleSelect} lang={lang}/>
        </div>
      </section>

      <section className="demo-story" id="story" aria-label="How Silent Interpreter works">
        <div className="ds-inner">
          <div className="ds-label">
            <SectionLabel>THE SILENT MOMENT</SectionLabel>
            <h2 className="ds-headline">One gesture.<br/><em>One conversation.</em></h2>
          </div>
          <div className="ds-stage">
            <div className="ds-scene ds-scene-1">
              <div className="ds-visual">
                <svg className="ds-hand-svg" viewBox="0 0 100 110" aria-hidden="true">
                  {LM_CONN.map(([a,b],i)=>(
                    <line key={i} stroke="rgba(43,181,168,0.45)" strokeWidth="1.2" fill="none"
                      x1={LM_PTS[a][0]} y1={LM_PTS[a][1]} x2={LM_PTS[b][0]} y2={LM_PTS[b][1]}/>
                  ))}
                  {LM_PTS.map(([x,y],i)=>(
                    <circle key={i} cx={x} cy={y} r="2.2" fill="#2BB5A8" style={{filter:'drop-shadow(0 0 5px #2BB5A8)'}}/>
                  ))}
                </svg>
                <div className="ds-waveform">
                  {Array.from({length:20}).map((_,i)=><i key={i} style={{animationDelay:`${i*0.065}s`}}/>)}
                </div>
              </div>
              <p className="ds-caption">You gesture in Indian Sign Language</p>
            </div>
            <div className="ds-scene ds-scene-2">
              <div className="ds-visual">
                <div className="ds-ring-wrap">
                  <svg className="ds-ring" viewBox="0 0 100 100" aria-hidden="true">
                    <circle cx="50" cy="50" r="38" fill="none" stroke="rgba(43,181,168,0.12)" strokeWidth="3"/>
                    <circle cx="50" cy="50" r="38" fill="none" stroke="#2BB5A8" strokeWidth="3"
                      strokeDasharray="55 185" strokeLinecap="round" style={{filter:'drop-shadow(0 0 8px #2BB5A8)'}}/>
                    <circle cx="50" cy="50" r="24" fill="none" stroke="rgba(43,181,168,0.07)" strokeWidth="2"/>
                    <text x="50" y="56" textAnchor="middle" fill="#2BB5A8" fontSize="11" fontFamily="var(--font-mono)" fontWeight="700">AI</text>
                  </svg>
                  <div className="ds-ring-dots">
                    {['MediaPipe','Landmarks','LSTM','Encoder'].map((label)=>(
                      <span key={label} className="ds-ring-dot">{label}</span>
                    ))}
                  </div>
                </div>
              </div>
              <p className="ds-caption">MediaPipe + LSTM analyzes every landmark</p>
            </div>
            <div className="ds-scene ds-scene-3">
              <div className="ds-visual">
                <div className="ds-translation-wrap">
                  <div className="ds-translate-from">
                    <span className="ds-t-label">ISL Gesture</span>
                    <strong className="ds-t-word">Open Palm</strong>
                  </div>
                  <div className="ds-translate-arrow"><ArrowRight size={28}/></div>
                  <div className="ds-translate-to">
                    <span className="ds-t-label">English</span>
                    <strong className="ds-t-word">I need help</strong>
                  </div>
                </div>
              </div>
              <p className="ds-caption">Translated with 96% confidence</p>
            </div>
            <div className="ds-scene ds-scene-4">
              <div className="ds-visual">
                <div className="ds-bubble-wrap">
                  <div className="ds-bubble">
                    <div className="ds-bubble-lang en">
                      <span className="ds-bubble-flag">🇬🇧</span>
                      <span>I need help finding my appointment.</span>
                    </div>
                    <div className="ds-bubble-divider"/>
                    <div className="ds-bubble-lang hi">
                      <span className="ds-bubble-flag">🇮🇳</span>
                      <span>मुझे अपनी अपॉइंटमेंट में मदद चाहिए।</span>
                    </div>
                  </div>
                  <div className="ds-bubble-wave">
                    {Array.from({length:14}).map((_,i)=><i key={i} style={{animationDelay:`${i*0.06}s`}}/>)}
                  </div>
                </div>
              </div>
              <p className="ds-caption">IndicTrans2 speaks in your language</p>
            </div>
          </div>
          <div className="ds-dots" aria-hidden="true">
            <span className="ds-dot ds-dot-1"/>
            <span className="ds-dot ds-dot-2"/>
            <span className="ds-dot ds-dot-3"/>
            <span className="ds-dot ds-dot-4"/>
          </div>
        </div>
      </section>

      <section className="live-section" id="live" aria-labelledby="live-h2">
        <div className="wrap">
          <div className="live-head">
            <div>
              <SectionLabel>A MOMENT YOU CAN TRY</SectionLabel>
              <h2 id="live-h2" className="reveal-h">The live mode <em>within reach.</em></h2>
              <p className="reveal-p">Start a local simulation to see gesture recognition, language output, and speech move together.</p>
            </div>
            <DemoBadge>Demo Simulation</DemoBadge>
          </div>
          <div className="live-panel">
            <div className="cam-pane">
              <div className="cam-pane-top">
                <span style={{display:'flex',alignItems:'center',gap:7}}>
                  <span className={`sig-dot ${isRunning?'on':''}`}/>
                  {isRunning?'gesture recognition active':'camera simulation ready'}
                </span>
                <span>{isRunning?'00:08':'00:00'}</span>
              </div>
              <div className="live-stage">
                {cameraGranted && <video ref={liveVideoRef} autoPlay muted playsInline aria-hidden="true"/>}
                <div className="live-grid" aria-hidden="true"/>
                <div className="live-vignette" aria-hidden="true"/>
                {!cameraGranted && (
                  <div className="live-signer" aria-hidden="true">
                    <div className="signer" style={{transform:'scale(1.3)'}}>
                      <i className="sh"/><i className="sb"/><i className="sa sa-l"/><i className="sa sa-r"/><i className="shand"/>
                    </div>
                  </div>
                )}
                <svg className="live-lm-svg" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                  {isRunning&&LM_CONN.map(([a,b],i)=>(
                    <line key={i} stroke="rgba(43,181,168,.35)" strokeWidth=".8"
                      x1={LM_PTS[a][0]} y1={LM_PTS[a][1]} x2={LM_PTS[b][0]} y2={LM_PTS[b][1]}/>
                  ))}
                  {isRunning&&LM_PTS.map(([x,y],i)=>(
                    <circle key={i} cx={x} cy={y} r="1.4" fill="#2BB5A8" style={{filter:'drop-shadow(0 0 3px #2BB5A8)'}}/>
                  ))}
                </svg>
                {isRunning&&<div className="live-detect-chip"><ScanFace size={10}/> {ctx.gesture}</div>}
                <span className="live-foot">No webcam required · visual simulation</span>
              </div>
              <div className="cam-controls">
                <button className={`run-btn ${isRunning?'stop':''}`} onClick={toggleRunning} aria-pressed={isRunning}>
                  {isRunning?<><Square size={14} fill="currentColor"/> Stop</>:<><Radio size={14}/> Start translation</>}
                </button>
                <div className="lang-toggle" role="group" aria-label="Output language">
                  <button className={lang==='english'?'on':''} onClick={()=>setLang('english')}>EN</button>
                  <button className={lang==='hindi'?'on':''} onClick={()=>setLang('hindi')}>HI</button>
                </div>
              </div>
            </div>
            <div className="out-pane">
              <div className="out-top">
                <span>SIMULATED TRANSCRIPT</span>
                <span className="out-conf">{isRunning?`${confidence}% confidence`:'awaiting gesture'}</span>
              </div>
              <div className="out-content" aria-live="polite">
                {isRunning?(<>
                  <div className="out-gesture-chip"><span>{ctx.emoji}</span> {ctx.gesture}</div>
                  <p className="out-en">{lang==='english'?ctx.english:ctx.hindi}</p>
                  {lang==='english'&&<p className="out-hi">{ctx.hindi}</p>}
                </>):(
                  <p className="out-placeholder">Start the simulation to see a conversation appear here.</p>
                )}
              </div>
              <div className="out-meter"><div className="out-meter-fill" style={{width:`${confidence}%`}}/></div>
              <div className={`out-wave ${speaking?'active':''}`} aria-hidden="true">
                {waveHeights.map((h,i)=><i key={i} style={{height:h*(speaking?1:.28),animationDelay:`${i*.04}s`}}/>)}
              </div>
              <div className="out-actions">
                <button className={`speak-btn ${speaking?'on':''}`}
                  onClick={()=>setSpeaking(!speaking)} disabled={!isRunning}
                  aria-label={speaking?'Stop speech':'Play speech'}>
                  {speaking?<Pause size={13}/>:<Volume2 size={13}/>}
                  {speaking?'Stop speech':'Play speech'}
                </button>
                <span className="out-note">{speaking?'Playing simulated speech…':'Web Speech API · future connection'}</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="replay-section" id="replay" aria-labelledby="replay-h2">
        <div className="wrap">
          <div className="replay-layout">
            <div className="replay-copy">
              <SectionLabel>CONVERSATION REPLAY</SectionLabel>
              <h2 id="replay-h2" className="reveal-h">Nothing meaningful has to <em>disappear.</em></h2>
              <p className="reveal-p">Every context you explore stays as a local memory.</p>
            </div>
            <div>
              <div className="replay-list">
                {replays.length===0&&(
                  <div className="replay-empty">
                    <Waves size={26} style={{color:'var(--accent)'}}/>
                    <p>Start the live demo above to create your first replay.</p>
                    <DemoBadge/>
                  </div>
                )}
                {replays.map((r,i)=>(
                  <article className="replay-item" key={`${r.time}-${i}`}>
                    <span className="replay-t">{r.time}</span>
                    <div className="replay-stem"><span className="replay-node-dot"/><span/></div>
                    <div className="replay-info">
                      <div className="replay-meta-row"><strong>{r.gesture}</strong><span className="replay-pct">{r.confidence}%</span></div>
                      <p>{lang==='english'?r.english:r.hindi}</p>
                      <small>{r.hindi}</small>
                    </div>
                    <button className={`replay-btn ${replayPlaying===i?'active':''}`}
                      onClick={()=>{setReplayPlaying(i);setTimeout(()=>setReplayPlaying(null),1400)}}
                      aria-label={`Replay ${r.gesture}`}>
                      {replayPlaying===i?<Pause size={13}/>:<Play size={13} fill="currentColor"/>}
                    </button>
                  </article>
                ))}
              </div>
              <div className="replay-foot"><DemoBadge/><span>Replay controls change local UI state only.</span></div>
            </div>
          </div>
        </div>
      </section>

      <section className="upload-section" id="upload" aria-labelledby="upload-h2">
        <div className="wrap">
          <div className="upload-layout">
            <div className="upload-copy">
              <SectionLabel>CONTINUE A CONVERSATION</SectionLabel>
              <h2 id="upload-h2" className="reveal-h">Let a video become a <em>timeline.</em></h2>
              <p className="reveal-p">Drop a clip to explore the shape of a future upload flow.</p>
              <ul className="upload-list" aria-label="Upload features">
                <li><Check size={13}/> Local-only processing</li>
                <li><Check size={13}/> Confidence and waveform preview</li>
                <li><Check size={13}/> Transcript download preview</li>
              </ul>
            </div>
            <div className="drop-zone" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();handleFile(e.dataTransfer.files[0])}} aria-label="Video file drop area">
              <input ref={fileRef} type="file" accept="video/*" hidden onChange={e=>handleFile(e.target.files?.[0])}/>
              {uploadState==='idle'&&(
                <button className="drop-idle" onClick={()=>fileRef.current?.click()}>
                  <div className="drop-icon" aria-hidden="true"><CloudUpload size={26}/></div>
                  <strong>Drop a video here</strong>
                  <span>or browse for a local clip</span>
                  <DemoBadge/>
                </button>
              )}
              {uploadState==='processing'&&(
                <div className="drop-processing" role="status" aria-live="polite">
                  <Sparkles size={26} style={{color:'var(--accent)'}}/>
                  <strong>Building a simulated timeline…</strong>
                  <span>{fileName}</span>
                  <div className="upload-progress"><div className="upload-progress-fill"/></div>
                  <div className="upload-steps">
                    <span className="ustep done"><Check size={10}/> gesture chips</span>
                    <span className="ustep">transcript</span>
                    <span className="ustep">speech preview</span>
                  </div>
                </div>
              )}
              {uploadState==='ready'&&(
                <div className="drop-ready" aria-live="polite">
                  <div className="ready-header">
                    <span className="ready-check"><Check size={15}/></span>
                    <div><strong>Timeline ready</strong><span>Demo Simulation · local result</span></div>
                    <button onClick={()=>{setUploadState('idle');setFileName('');setChipsVisible([])}} aria-label="Reset upload"><X size={15}/></button>
                  </div>
                  <div className="gesture-chips-row" aria-label="Detected gestures">
                    {['👋 Open Palm','🤝 Handshake','☝️ Point','✊ Fist','🖐️ Five'].map((chip,i)=>(
                      <span key={chip} className={`gchip ${chipsVisible[i]?'show':''}`} style={{transitionDelay:`${i*.07}s`}}>{chip}</span>
                    ))}
                  </div>
                  <div className="transcript-box">
                    <small>RECOGNIZED TEXT</small>
                    <p>{ctx.english}</p>
                    <div className="upload-wave" aria-hidden="true">
                      {waveHeights.slice(0,20).map((h,i)=><i key={i} style={{height:h*.5,animationDelay:`${i*.05}s`}}/>)}
                    </div>
                  </div>
                  <button className="dl-btn" disabled aria-label="Download transcript (disabled)"><Upload size={13}/> Download transcript preview</button>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      <section className="tech-section" id="technology" aria-labelledby="tech-h2">
        <div className="wrap">
          <div className="tech-inner">
            <div className="tech-copy">
              <SectionLabel>HOW IT WORKS</SectionLabel>
              <h2 id="tech-h2" className="reveal-h">Technology is the proof, <em>not the story.</em></h2>
              <p className="reveal-p">Four simple steps — from a gesture you make to a word someone hears.</p>
              <button className={`tech-expand-btn ${techExpanded?'open':''}`} onClick={()=>setTechExpanded(!techExpanded)} aria-expanded={techExpanded} aria-controls="tech-pipeline">
                {techExpanded?'Hide technical pipeline':'See technical pipeline'}<ChevronDown size={14}/>
              </button>
            </div>
            <div>
              <div className="tech-simple" aria-label="Simplified technology steps">
                {PIPELINE_SIMPLE.map(step=>(
                  <div key={step.step} className={`tech-step ${step.highlight?'highlight':''}`}>
                    <span className="tech-step-num">{step.step}</span>
                    <div className="tech-step-body"><strong>{step.label}</strong><span>{step.sub}</span></div>
                  </div>
                ))}
              </div>
              <div id="tech-pipeline" className={`tech-full ${techExpanded?'open':''}`} aria-hidden={!techExpanded}>
                <div className="pipeline-row">
                  {PIPELINE_FULL.slice(0,5).map(n=>(
                    <div className="pipe-node" key={n.label}><div className="pipe-dot"/><span className="pipe-label">{n.label}</span><span className="pipe-sub">{n.sub}</span></div>
                  ))}
                </div>
                <div className="pipeline-row" style={{marginTop:8}}>
                  {PIPELINE_FULL.slice(5).map(n=>(
                    <div className="pipe-node" key={n.label}><div className="pipe-dot"/><span className="pipe-label">{n.label}</span><span className="pipe-sub">{n.sub}</span></div>
                  ))}
                </div>
                <div style={{paddingTop:16,display:'flex',alignItems:'center',gap:10}}>
                  <DemoBadge/>
                  <span style={{color:'var(--muted)',fontFamily:'var(--font-mono)',fontSize:10}}>Future inference pipeline — simulated visualization only</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="research-section" id="research" aria-labelledby="research-h2">
        <div className="wrap">
          <div className="research-head">
            <div>
              <SectionLabel>TRAINED ON INDIAN SIGN LANGUAGE RESEARCH</SectionLabel>
              <h2 id="research-h2" className="reveal-h">Open knowledge, <em>carefully held.</em></h2>
            </div>
            <p className="reveal-p">These datasets represent the research roots that teach a future interpreter the depth of Indian Sign Language.</p>
          </div>
          <div className="research-cards">
            {DATASETS.map(ds=>{
              const open=activeDataset===ds.name
              return (
                <article className={`r-card ${open?'open':''}`} key={ds.name}>
                  <button className="r-trigger" onClick={()=>setActiveDataset(open?null:ds.name)} aria-expanded={open}>
                    <span className="r-tab">{ds.mark}</span>
                    <span><strong>{ds.name}</strong><small>{ds.meta}</small></span>
                    <ChevronDown size={14}/>
                  </button>
                  <div className="r-body" aria-hidden={!open}>
                    <div className="r-inner"><p>{ds.purpose}</p><span>{ds.role}</span><div className="r-bars"><i/><i/><i/><i/><i/></div></div>
                  </div>
                </article>
              )
            })}
          </div>
        </div>
      </section>

      <section className="roadmap-section" id="roadmap" aria-labelledby="roadmap-h2">
        <div className="wrap">
          <div className="roadmap-layout">
            <div className="roadmap-copy">
              <SectionLabel>THE ROAD AHEAD</SectionLabel>
              <h2 id="roadmap-h2" className="reveal-h">From a first step to <em>universal communication.</em></h2>
              <p className="reveal-p">A staircase, not a promise. Each level is a direction for future research.</p>
              <div style={{marginTop:22}}><DemoBadge>Roadmap Simulation</DemoBadge></div>
            </div>
            <div className="staircase" aria-label="Silent Interpreter roadmap staircase">
              {ROADMAP.map((item,i)=>(
                <div key={item} className={`stair ${i===0?'current':''}`} style={{height:`${82+i*28}px`}}>
                  <span>{String(i+1).padStart(2,'0')}</span>
                  <strong>{item}</strong>
                  {i===0&&<small>now</small>}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="cta-section" aria-labelledby="cta-h2">
        <div className="cta-bg-gradient" aria-hidden="true"/>
        <div className="wrap cta-wrap">
          <SectionLabel>JOIN THE CONVERSATION</SectionLabel>
          <h2 className="cta-headline" id="cta-h2">Every person deserves<br/>to be <em>understood.</em></h2>
          <p className="cta-sub">Silent Interpreter is built for the moments when being heard matters most. A future where language is never a barrier.</p>
          <div className="cta-btn-wrap">
            <button className="cta-btn btn-primary" onClick={requestCamera}><Video size={16}/> Experience the demo</button>
            <a href="#live" className="btn-ghost"><Play size={15} fill="currentColor"/> See how it works</a>
            <div className="cta-glow" aria-hidden="true"/>
          </div>
        </div>
      </section>

      <footer className="footer" aria-label="Site footer">
        <div className="wrap">
          <div className="footer-inner">
            <a href="#top" className="brand" aria-label="Back to top"><Mark/><span>Silent Interpreter</span></a>
            <p className="footer-center">Built for the moments when being understood matters.</p>
            <div className="footer-links">
              <a href="https://github.com" target="_blank" rel="noreferrer">GitHub <ArrowUpRight size={12}/></a>
              <a href="#live">Contact</a>
              <a href="#top">Accessibility</a>
            </div>
          </div>
          <p className="footer-end">Built for hackathons · 2026 · Demo Simulation — No AI backend connected · Indian Sign Language Research Project</p>
        </div>
      </footer>
    </div>
  )
}
