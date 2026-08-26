'use client'

import {
  useCallback, useEffect, useMemo, useRef, useState,
} from 'react'
import {
  ArrowRight, ArrowUpRight, Check, ChevronDown,
  CloudUpload, Menu, Pause, Play, Radio, ScanFace, Sparkles,
  Square, Upload, Video, Volume2, Waves, X,
} from 'lucide-react'
import { useISLRecognition, CONFIDENCE_THRESHOLD } from '@/lib/isl/useISLRecognition'
import { getDisplayLabel } from '@/lib/isl/displayLabels'

/* ═══════════════════════════════════════════════════════
   SIMULATION DATA  —  replace simulateRecognition() with
   the real model later without touching any UI component
═══════════════════════════════════════════════════════ */

export type RecognitionResult = {
  id: string
  gesture: string
  emoji: string
  english: string
  hindi: string
  confidence: number
  context: string
  signature?: any
}

/* Legacy presentation-only data. Never used by camera or live interpretation. */
const LEGACY_DEMO_GESTURES: RecognitionResult[] = [
  {
    id: 'open-palm',
    gesture: 'Open Palm',
    emoji: '🤚',
    english: 'I need help finding my appointment.',
    hindi: 'मुझे अपनी अपॉइंटमेंट में मदद चाहिए।',
    confidence: 96,
    context: 'Hospital',
  },
  {
    id: 'raised-hand',
    gesture: 'Raised Hand',
    emoji: '✋',
    english: 'Could you explain this again?',
    hindi: 'क्या आप यह फिर से समझा सकते हैं?',
    confidence: 93,
    context: 'Classroom',
  },
  {
    id: 'point',
    gesture: 'Point',
    emoji: '☝️',
    english: 'Where is Gate 12?',
    hindi: 'गेट 12 कहाँ है?',
    confidence: 91,
    context: 'Airport',
  },
  {
    id: 'thumbs-up',
    gesture: 'Thumbs Up',
    emoji: '👍',
    english: 'Yes, thank you.',
    hindi: 'जी, शुक्रिया।',
    confidence: 94,
    context: 'General',
  },
  {
    id: 'wave',
    gesture: 'Wave',
    emoji: '👋',
    english: 'Hello, nice to meet you.',
    hindi: 'नमस्ते, आपसे मिलकर अच्छा लगा।',
    confidence: 89,
    context: 'General',
  },
]

/* ═══════════════════════════════════════════════════════
   LANDMARK GEOMETRY  (for decorative SVG overlay)
═══════════════════════════════════════════════════════ */
const LM_PTS: [number, number][] = [
  [50,78],[48,65],[45,55],[44,47],[42,41],
  [52,44],[53,32],[54,26],[55,21],
  [57,45],[59,31],[60,24],[61,19],
  [62,47],[64,35],[65,29],[66,25],
  [66,50],[67,41],[67,36],[68,32],
]
const LM_CONN: [number,number][] = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
]

/* ═══════════════════════════════════════════════════════
   DATASETS / ROADMAP CONSTANTS
═══════════════════════════════════════════════════════ */
const DATASETS = [
  { name: 'INCLUDE',     meta: '4,287 videos',    purpose: 'Isolated ISL words',             role: 'A clear baseline for recognizing individual gestures.',             mark: 'IN' },
  { name: 'ISLTranslate',meta: '31,000+ samples', purpose: 'Continuous sentence translation', role: 'Teaches models how meaning flows across sequences.',                mark: 'IS' },
  { name: 'ISL-CSLTR',  meta: '5,000 sentences',  purpose: 'Continuous sign recognition',    role: 'Supports temporal modeling and natural signing.',                   mark: 'CS' },
  { name: 'CISLR',      meta: '700+ glosses',     purpose: 'Large-vocabulary ISL',            role: 'Adds real-world breadth to the research desk.',                    mark: 'CI' },
  { name: 'ISL-FS',     meta: '36 classes',        purpose: 'Alphabet and numbers',           role: 'Helps spell the words that gesture vocabularies miss.',            mark: 'FS' },
]

const ROADMAP = [
  { label: 'Hackathon MVP',         active: true  },
  { label: 'Continuous ISL',        active: false },
  { label: 'Transformer Models',    active: false },
  { label: 'Offline Edge AI',       active: false },
  { label: '3D Avatar Output',      active: false },
  { label: 'Smart Glasses',         active: false },
  { label: 'Universal Communication', active: false },
]

const PIPELINE_FULL = [
  { label: 'Camera',        sub: 'input',    cat: 'input'    },
  { label: 'MediaPipe',     sub: 'vision',   cat: 'vision'   },
  { label: 'Landmarks',     sub: 'vision',   cat: 'vision'   },
  { label: 'Sliding Window',sub: 'model',    cat: 'model'    },
  { label: 'LSTM',          sub: 'model',    cat: 'model'    },
  { label: 'English',       sub: 'language', cat: 'language' },
  { label: 'IndicTrans2',   sub: 'language', cat: 'language' },
  { label: 'Hindi',         sub: 'language', cat: 'language' },
  { label: 'Speech API',    sub: 'output',   cat: 'output'   },
  { label: 'Replay',        sub: 'output',   cat: 'output'   },
]

/* ═══════════════════════════════════════════════════════
   SMALL REUSABLE COMPONENTS
═══════════════════════════════════════════════════════ */

function NavMark() {
  return (
    <span className="nav-mark" aria-hidden="true">
      <span /><span /><span />
    </span>
  )
}

function SectionLabel({ children }: { children: string }) {
  return <p className="section-label">{children}</p>
}

function Dot({ color = '', pulse = false }: { color?: string; pulse?: boolean }) {
  return (
    <span
      className={['dot', color, pulse ? 'pulse' : ''].filter(Boolean).join(' ')}
      aria-hidden="true"
    />
  )
}

function Badge({ children, sim = false }: { children: React.ReactNode; sim?: boolean }) {
  return <span className={`badge ${sim ? 'sim' : ''}`}>{children}</span>
}

/* ═══════════════════════════════════════════════════════
   PROCESSING PIPELINE COMPONENT
═══════════════════════════════════════════════════════ */
const PIPE_STAGES = ['Camera', 'Vision', 'Gesture', 'Meaning', 'Voice']

function ProcessingPipeline({ activeStage }: { activeStage: number }) {
  return (
    <div className="pipeline-row-container">
      {PIPE_STAGES.map((stage, i) => (
        <div key={stage} className="pipe-node-wrap">
          <div className="pipe-node">
            <div className={`pipe-dot ${i <= activeStage ? 'active' : ''}`} />
            <span className={`pipe-label ${i <= activeStage ? 'active' : ''}`}>{stage}</span>
          </div>
          {i < PIPE_STAGES.length - 1 && (
            <div className="pipe-connector">
              <div className={`pipe-connector-fill ${i < activeStage ? 'active' : ''}`} />
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════
   WORD STREAM COMPONENT
═══════════════════════════════════════════════════════ */
function WordStream({
  isRunning,
  result,
  lang,
  isLiveModel,
}: {
  isRunning: boolean
  result: RecognitionResult | null
  lang: 'english' | 'hindi'
  isLiveModel?: boolean
}) {
  // Live model mode: display the real label directly without fake typing delays.
  // Demo mode: retain the existing animated word-by-word reveal.
  const [words, setWords] = useState<string[]>([])
  const [latestIdx, setLatestIdx] = useState(-1)
  const timers = useRef<ReturnType<typeof setTimeout>[]>([])

  useEffect(() => {
    timers.current.forEach(clearTimeout)
    timers.current = []
    setWords([])
    setLatestIdx(-1)
    if (!isRunning || !result) return

    // Live model: show label immediately, no fake word-by-word animation
    if (isLiveModel) {
      setWords(result.english.split(' '))
      setLatestIdx(result.english.split(' ').length - 1)
      return
    }

    // Demo mode: animated word reveal
    const sentence = result.english
    const ws = sentence.split(' ')
    ws.forEach((_, i) => {
      const t = setTimeout(() => {
        setWords(prev => [...prev, ws[i]])
        setLatestIdx(i)
      }, 360 + i * 240)
      timers.current.push(t)
    })
    return () => timers.current.forEach(clearTimeout)
  }, [isRunning, result?.id, isLiveModel])

  const isTyping = !isLiveModel && isRunning && result && words.length < result.english.split(' ').length

  if (!isRunning || !result) {
    return (
      <div className="word-stream">
        <p className="word-stream-idle">
          {isLiveModel ? 'Waiting for signs…' : 'Start simulation to see language generated live'}
        </p>
      </div>
    )
  }

  // Hindi is generated by the live model now
  const showHindi = lang === 'hindi'

  return (
    <>
      <div className="word-stream" aria-live="polite" aria-label="Generated text">
        {lang === 'english' || isLiveModel
          ? words.map((w, i) => (
              <span key={`${result.id}-${i}`} className={`word-token ${i === latestIdx ? 'new' : ''}`}>
                {w}
              </span>
            ))
          : (showHindi
              ? <span className="word-token">{result.hindi}</span>
              : null
            )
        }
        {isTyping && <span className="word-cursor" aria-hidden="true" />}
      </div>
      {lang === 'english' && (
        <p className="interp-hindi">
          {result.hindi}
        </p>
      )}
    </>
  )
}

/* ═══════════════════════════════════════════════════════
   SPEECH CONTROLS COMPONENT
═══════════════════════════════════════════════════════ */
function SpeechControls({
  text,
  disabled,
}: {
  text: string
  disabled: boolean
}) {
  const [speaking, setSpeaking] = useState(false)
  const [ready, setReady] = useState(false)
  const uttRef = useRef<SpeechSynthesisUtterance | null>(null)
  const waveHeights = useMemo(() => Array.from({ length: 18 }, (_, i) => 8 + ((i * 13) % 28)), [])

  useEffect(() => {
    if (typeof window === 'undefined') return
    const onVoices = () => setReady(speechSynthesis.getVoices().length > 0)
    speechSynthesis.addEventListener('voiceschanged', onVoices)
    onVoices()
    return () => {
      speechSynthesis.removeEventListener('voiceschanged', onVoices)
      speechSynthesis.cancel()
    }
  }, [])

  useEffect(() => {
    setSpeaking(false)
    speechSynthesis.cancel()
  }, [text])

  function speak() {
    if (!text || disabled) return
    speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(text)
    const voices = speechSynthesis.getVoices().filter(v => v.lang.startsWith('en'))
    if (voices.length) u.voice = voices[0]
    u.rate = 0.92
    u.onend = () => setSpeaking(false)
    u.onerror = () => setSpeaking(false)
    uttRef.current = u
    speechSynthesis.speak(u)
    setSpeaking(true)
  }

  function stop() {
    speechSynthesis.cancel()
    setSpeaking(false)
  }

  return (
    <div className="speech-row">
      {speaking ? (
        <button className="btn btn-sm btn-primary" onClick={stop} aria-label="Stop speech">
          <Square size={11} fill="currentColor" /> Stop
        </button>
      ) : (
        <button
          className="btn btn-sm btn-teal"
          onClick={speak}
          disabled={disabled}
          aria-label="Speak generated text"
        >
          <Volume2 size={13} /> Speak
        </button>
      )}
      <div className={`waveform ${speaking ? 'active' : ''}`} aria-hidden="true">
        {waveHeights.map((h, i) => (
          <i
            key={i}
            style={{
              height: h,
              animationDelay: `${i * 0.055}s`,
              animationDuration: `${0.55 + (i % 3) * 0.12}s`,
            }}
          />
        ))}
      </div>
      <span className="speak-status">
        {disabled ? 'Speech coming soon' : speaking ? '🔊 Speaking…' : ready ? 'Ready to speak' : 'Web Speech API'}
      </span>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════
   DEMO GESTURE SELECTOR
═══════════════════════════════════════════════════════ */
function DemoGestureSelector({
  selected,
  onChange,
}: {
  selected: string
  onChange: (id: string) => void
}) {
  return (
    <div className="gesture-selector-card">
      <p className="gesture-selector-label">
        <ScanFace size={11} />
        Demo gestures · simulation
      </p>
      <div className="gesture-pills" role="group" aria-label="Select demo gesture">
        {LEGACY_DEMO_GESTURES.map(g => (
          <button
            key={g.id}
            className={`gesture-pill ${selected === g.id ? 'active' : ''}`}
            onClick={() => onChange(g.id)}
            aria-pressed={selected === g.id}
            aria-label={`Demo gesture: ${g.gesture}`}
          >
            {g.emoji} {g.gesture}
          </button>
        ))}
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════
   CAMERA PREVIEW + CONTROLS COMPONENT
═══════════════════════════════════════════════════════ */
type CamState = 'idle' | 'requesting' | 'active' | 'recording' | 'stopped' | 'error'
type CamError = 'denied' | 'notfound' | 'busy' | 'unsupported' | 'unknown' | null

/* ── Helper: pick best supported MediaRecorder MIME type ── */
function getSupportedMimeType(): string {
  const candidates = [
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
    'video/mp4',
  ]
  for (const t of candidates) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(t)) return t
  }
  return ''
}

function CameraPreview({
  isRunning,
  result,
  camState,
  setCamState,
  onCamera,
  videoRef,
  streamRef,
  elapsed,
  camError,
}: {
  isRunning: boolean
  result: RecognitionResult | null
  camState: CamState
  setCamState: (s: CamState) => void
  onCamera: () => Promise<void>
  videoRef: React.RefObject<HTMLVideoElement | null>
  streamRef: React.RefObject<MediaStream | null>
  elapsed: number
  camError: CamError
}) {
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const [replayUrl, setReplayUrl] = useState<string | null>(null)
  const [replayMime, setReplayMime] = useState('video/webm')
  const replayRef = useRef<HTMLVideoElement | null>(null)
  const [replayPlaying, setReplayPlaying] = useState(false)
  const [recorderError, setRecorderError] = useState<string | null>(null)

  /* Attach stream to video element whenever camState becomes active/recording.
     The video element is always in the DOM (just hidden), so videoRef.current
     is available when this effect runs. */
  useEffect(() => {
    if ((camState === 'active' || camState === 'recording') && streamRef.current && videoRef.current) {
      const vid = videoRef.current
      if (vid.srcObject !== streamRef.current) {
        vid.srcObject = streamRef.current
        vid.play().catch(() => {})
      }
    }
  }, [camState, streamRef, videoRef])

  /* Clean up replay URL when we get a new one */
  function updateReplayUrl(blob: Blob, mime: string) {
    setReplayUrl(prev => {
      if (prev) URL.revokeObjectURL(prev)
      return URL.createObjectURL(blob)
    })
    setReplayMime(mime)
  }

  function formatTime(s: number) {
    const m = Math.floor(s / 60)
    const sec = s % 60
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
  }

  function startRecording() {
    setRecorderError(null)
    if (typeof MediaRecorder === 'undefined') {
      setRecorderError('Recording not supported in this browser.')
      return
    }
    const stream = streamRef.current
    if (!stream) return
    const mimeType = getSupportedMimeType()
    chunksRef.current = []
    let rec: MediaRecorder
    try {
      rec = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
    } catch {
      setRecorderError('Could not start recorder — try a different browser.')
      return
    }
    rec.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
    rec.onstop = () => {
      const effectiveMime = rec.mimeType || mimeType || 'video/webm'
      const blob = new Blob(chunksRef.current, { type: effectiveMime })
      updateReplayUrl(blob, effectiveMime)
      setCamState('stopped')
    }
    rec.onerror = () => {
      setRecorderError('Recording failed unexpectedly.')
      setCamState('active')
    }
    recorderRef.current = rec
    rec.start(250) // timeslice = collect data every 250 ms
    setCamState('recording')
  }

  function stopRecording() {
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop()
    }
  }

  function handleDownload() {
    if (!replayUrl) return
    const ext = replayMime.includes('mp4') ? 'mp4' : 'webm'
    const a = document.createElement('a')
    a.href = replayUrl
    a.download = `silent-interpreter-recording.${ext}`
    a.click()
  }

  const errorMessages: Record<NonNullable<CamError>, string> = {
    denied:      'Camera access denied. Allow camera in browser settings.',
    notfound:    'No camera found on this device.',
    busy:        'Camera is in use by another application.',
    unsupported: 'getUserMedia is not supported in this browser.',
    unknown:     'Unable to access camera.',
  }

  const statusLabel =
    camState === 'idle'       ? 'Camera ready' :
    camState === 'requesting' ? 'Requesting…'  :
    camState === 'active'     ? 'Camera active' :
    camState === 'recording'  ? `Recording ${formatTime(elapsed)}` :
    camState === 'error'      ? 'Camera error' :
                                'Recording stopped'

  const dotColor =
    camState === 'idle'      ? '' :
    camState === 'active'    ? 'green' :
    camState === 'recording' ? 'record' :
    camState === 'error'     ? 'record' : 'amber'

  const showLiveVideo = camState === 'active' || camState === 'recording'

  return (
    <div className="camera-card">
      {/* Header */}
      <div className="camera-card-header">
        <div className="camera-card-title">
          <Video size={11} />
          Live Camera
        </div>
        <div className="camera-card-meta">
          <Dot color={dotColor} pulse={camState === 'recording'} />
          {statusLabel}
          {showLiveVideo && (
            <span>· LIVE</span>
          )}
          <Badge>UI DEMO</Badge>
        </div>
      </div>

      {/* Viewport */}
      <div className={`camera-viewport ${camState === 'recording' ? 'recording' : ''}`}>
        {/* Corner marks */}
        <span className="camera-corner tl" /><span className="camera-corner tr" />
        <span className="camera-corner bl" /><span className="camera-corner br" />

        {/* Scan line — always visible for polish */}
        <div className="camera-scanline" aria-hidden="true" />

        {/* Camera grid */}
        {showLiveVideo && (
          <div className="camera-grid" aria-hidden="true" />
        )}

        {/* ── LIVE VIDEO element — always in DOM so videoRef is always valid ── */}
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="camera-video"
          aria-label="Live camera feed"
          style={{ display: showLiveVideo ? 'block' : 'none' }}
        />

        {/* Overlays on top of live feed */}
        {showLiveVideo && (
          <>
            {/* Signing guide */}
            <div className="camera-guide" aria-hidden="true">
              <span className="camera-guide-label">Signing Area</span>
            </div>
            {/* LM overlay when running */}
            {isRunning && (
              <svg
                className="camera-lm-svg"
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
                aria-hidden="true"
              >
                {LM_CONN.map(([a, b], i) => (
                  <line
                    key={i}
                    stroke="rgba(43,181,168,.32)"
                    strokeWidth=".7"
                    x1={LM_PTS[a][0]} y1={LM_PTS[a][1]}
                    x2={LM_PTS[b][0]} y2={LM_PTS[b][1]}
                  />
                ))}
                {LM_PTS.map(([x, y], i) => (
                  <circle
                    key={i}
                    cx={x} cy={y} r="1.3"
                    fill="#2BB5A8"
                    style={{ filter: 'drop-shadow(0 0 2px #2BB5A8)' }}
                  />
                ))}
              </svg>
            )}
            {/* Detect chip */}
            {isRunning && result && (
              <div className="camera-detect-chip">
                <ScanFace size={9} /> {result.gesture}
              </div>
            )}
            {/* Timer chip */}
            {camState === 'recording' && (
              <div className="camera-timer-chip">
                <Dot color="record" pulse /> {formatTime(elapsed)}
              </div>
            )}
          </>
        )}

        {/* Replay video (shown when stopped) */}
        {camState === 'stopped' && replayUrl && (
          <video
            ref={replayRef}
            src={replayUrl}
            className="camera-video"
            controls
            playsInline
            aria-label="Recording replay"
            style={{ display: 'block' }}
          />
        )}

        {/* Error state */}
        {camState === 'error' && (
          <div className="camera-idle">
            <div className="camera-idle-icon" style={{ color: 'var(--record)' }}>
              <Video size={22} strokeWidth={1.5} />
            </div>
            <span className="camera-idle-label" style={{ color: 'var(--record)', textAlign: 'center', padding: '0 16px' }}>
              {camError ? errorMessages[camError] : 'Camera error'}
            </span>
          </div>
        )}

        {/* Idle placeholder */}
        {(camState === 'idle' || camState === 'requesting') && (
          <div className="camera-idle">
            <div className="camera-idle-icon">
              <Video size={22} strokeWidth={1.5} />
            </div>
            <span className="camera-idle-label">
              {camState === 'requesting' ? 'Requesting camera…' : 'Enable camera to begin'}
            </span>
          </div>
        )}
      </div>

      {/* Recorder error banner */}
      {recorderError && (
        <div style={{ padding: '6px 14px', background: 'rgba(229,62,82,.08)', borderTop: '1px solid rgba(229,62,82,.15)', fontSize: 11, color: 'var(--record)' }}>
          {recorderError}
        </div>
      )}

      {/* Footer controls */}
      <div className="camera-card-footer">
        <div className="camera-footer-info">
          <Dot color="" />
          <span>LOCAL CAMERA · SIMULATION</span>
          {showLiveVideo && (
            <span>· {result?.context ?? 'General'}</span>
          )}
        </div>
        <div className="camera-controls-row">
          {(camState === 'idle' || camState === 'error') && (
            <button
              className="btn btn-sm btn-teal"
              onClick={onCamera}
              aria-label="Enable camera"
            >
              <Video size={13} /> {camState === 'error' ? 'Try Again' : 'Enable Camera'}
            </button>
          )}
          {camState === 'requesting' && (
            <button className="btn btn-sm btn-ghost" disabled>
              <Radio size={13} /> Requesting…
            </button>
          )}
          {camState === 'active' && (
            <button
              className="btn btn-sm btn-primary"
              onClick={startRecording}
              aria-label="Start recording"
            >
              <Radio size={13} /> Record
            </button>
          )}
          {camState === 'recording' && (
            <button
              className="btn btn-sm"
              style={{ background: '#e64050', color: '#fff', borderRadius: 'var(--r-full)', padding: '7px 15px', fontSize: 12, fontWeight: 700 }}
              onClick={stopRecording}
              aria-label="Stop recording"
            >
              <Square size={11} fill="currentColor" /> Stop Recording
            </button>
          )}
          {camState === 'stopped' && replayUrl && (
            <>
              <button
                className="btn btn-sm btn-ghost"
                onClick={() => {
                  if (replayRef.current) {
                    if (replayRef.current.paused) {
                      replayRef.current.play()
                      setReplayPlaying(true)
                    } else {
                      replayRef.current.pause()
                      setReplayPlaying(false)
                    }
                  }
                }}
                aria-label={replayPlaying ? 'Pause replay' : 'Play recording'}
              >
                {replayPlaying ? <Pause size={13} /> : <Play size={13} fill="currentColor" />}
                {replayPlaying ? 'Pause' : 'Play Recording'}
              </button>
              <button
                className="btn btn-sm btn-ghost"
                onClick={handleDownload}
                aria-label="Download recording"
              >
                <CloudUpload size={13} /> Download
              </button>
              <button
                className="btn btn-sm btn-ghost"
                onClick={() => {
                  setReplayUrl(prev => { if (prev) URL.revokeObjectURL(prev); return null })
                  setReplayPlaying(false)
                  setCamState('active')
                }}
                aria-label="Retake recording"
              >
                Retake
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════
   INTERPRETATION CARD COMPONENT
═══════════════════════════════════════════════════════ */
function InterpretationCard({
  isRunning,
  result,
  lang,
  onLang,
  pipelineStage,
  modelState,
  isLiveModel,
  bufferFill,
}: {
  isRunning: boolean
  result: RecognitionResult | null
  lang: 'english' | 'hindi'
  onLang: (l: 'english' | 'hindi') => void
  pipelineStage: number
  modelState?: string
  isLiveModel?: boolean
  bufferFill?: number
}) {
  // Confidence: real model returns 0-1 float; demo uses 0-100 integer
  const rawConf = isRunning && result ? result.confidence : 0
  const confidencePct = isLiveModel ? Math.round(rawConf * 100) : rawConf

  const modelBadge =
    modelState === 'loading' ? 'MODEL LOADING' :
    modelState === 'ready'   ? (isLiveModel ? 'LIVE MODEL' : 'MODEL READY') :
    modelState === 'error'   ? 'MODEL ERROR' :
                               'UI DEMO'

  const headerDotColor =
    modelState === 'ready' && isLiveModel ? 'green' :
    modelState === 'error'                ? 'record' : ''

  return (
    <div className="interp-card">
      <div className="interp-header">
        <span className="interp-title">Live Interpretation</span>
        <div className="interp-meta">
          <Dot color={headerDotColor} pulse={modelState === 'ready' && isLiveModel && isRunning} />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--muted)', letterSpacing: '.05em' }}>
            {isLiveModel ? 'ISL MODEL' : 'SIMULATION'}
          </span>
          <Badge sim={!isLiveModel || modelState !== 'ready'}>{modelBadge}</Badge>
        </div>
      </div>

      {/* Pipeline */}
      <ProcessingPipeline activeStage={isRunning ? pipelineStage : -1} />

      <div className="interp-body">
        {/* Buffer fill indicator when accumulating frames */}
        {isLiveModel && isRunning && result === null && (bufferFill ?? 0) > 0 && (bufferFill ?? 0) < 1 && (
          <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 4 }}>
            Accumulating frames… {Math.round((bufferFill ?? 0) * 100)}%
          </div>
        )}

        {/* Gesture + Confidence */}
        <div className="gesture-row">
          <div className="gesture-label-group">
            <span className="gesture-sublabel">Gesture Detected</span>
            <span className="gesture-name">
              {isRunning && result
                ? (isLiveModel ? `🤙 ${result.gesture}` : `${result.emoji} ${result.gesture}`)
                : (isLiveModel && isRunning
                    ? ((bufferFill ?? 0) < 1 ? 'Collecting frames…' : 'Waiting for sign…')
                    : '—')
              }
            </span>
          </div>
          <div className="conf-pill">
            <span className="conf-pill-label">Confidence</span>
            <span className="conf-pill-val">
              {isRunning && result && confidencePct > 0
                ? `${confidencePct}%`
                : (isLiveModel ? '—' : '—')}
            </span>
          </div>
        </div>

        {/* Confidence bar */}
        <div className="conf-bar">
          <div className="conf-bar-fill" style={{ width: `${confidencePct}%` }} />
        </div>

        {/* Language toggle — hidden/simplified in live model mode */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className="word-stream-label">
            {isLiveModel ? 'Recognized Sign' : 'Generated Language'}
          </span>
          {!isLiveModel && (
            <div style={{ marginLeft: 'auto', display: 'flex', gap: 4 }} role="group" aria-label="Output language">
            {(['english', 'hindi'] as const).map(l => (
              <button
                key={l}
                onClick={() => onLang(l)}
                aria-pressed={lang === l}
                className={`lang-btn ${lang === l ? 'active' : ''}`}
              >
                {l === 'english' ? 'EN' : 'HI'}
              </button>
            ))}
          </div>
          )}
        </div>

        {/* Live model: show raw label note */}
        {isLiveModel && (
          <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 2, fontFamily: 'var(--font-mono)', letterSpacing: '.04em' }}>
            RAW ISL LABEL • labels.json index: {isRunning && result ? String(result.id).replace('ml-', '') : '—'}
          </div>
        )}

        {/* Word stream */}
        <WordStream isRunning={isRunning} result={result} lang={lang} isLiveModel={isLiveModel} />

        {/* Dictionary / Gesture Signature context (vague answer fallback) */}
        {isLiveModel && isRunning && result?.signature && (
          <div style={{ marginTop: 12, padding: 10, background: 'rgba(255,255,255,0.03)', borderRadius: 8, border: '1px solid rgba(255,255,255,0.1)' }}>
            <div style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', marginBottom: 6, fontWeight: 600 }}>
              Dictionary Context
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 12 }}>
              {result.signature.category && <div><span style={{color: 'var(--muted)'}}>Category:</span> {result.signature.category}</div>}
              {result.signature.hands && <div><span style={{color: 'var(--muted)'}}>Hands:</span> {result.signature.hands}</div>}
              {result.signature.handshape && <div><span style={{color: 'var(--muted)'}}>Handshape:</span> {result.signature.handshape.join(', ')}</div>}
              {result.signature.location && <div><span style={{color: 'var(--muted)'}}>Location:</span> {result.signature.location.join(', ')}</div>}
            </div>
          </div>
        )}

        {/* Speech controls */}
        <SpeechControls
          text={isRunning && result ? result.english : ''}
          disabled={false}
        />
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════
   HERO SECTION
═══════════════════════════════════════════════════════ */
function HeroSection({
  result,
  isRunning,
  lang,
  onLang,
  camState,
  setCamState,
  onCamera,
  videoRef,
  streamRef,
  onStartTranslation,
  pipelineStage,
  selectedGesture,
  onGestureChange,
  camError,
  modelState,
  isLiveModel,
  bufferFill,
}: {
  result: RecognitionResult | null
  isRunning: boolean
  lang: 'english' | 'hindi'
  onLang: (l: 'english' | 'hindi') => void
  camState: CamState
  setCamState: (s: CamState) => void
  onCamera: () => Promise<void>
  videoRef: React.RefObject<HTMLVideoElement | null>
  streamRef: React.RefObject<MediaStream | null>
  onStartTranslation: () => void
  pipelineStage: number
  selectedGesture: string
  onGestureChange: (id: string) => void
  camError: CamError
  modelState?: string
  isLiveModel?: boolean
  bufferFill?: number
}) {
  const [elapsed, setElapsed] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (camState === 'recording') {
      setElapsed(0)
      timerRef.current = setInterval(() => setElapsed(e => e + 1), 1000)
    } else {
      if (timerRef.current) clearInterval(timerRef.current)
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [camState])

  return (
    <section className="hero" id="top" aria-labelledby="hero-h1">
      <div className="hero-inner">
        {/* Left: Copy */}
        <div className="hero-copy">
          {/* Eyebrow */}
          <div className="hero-eyebrow" id="hero-eyebrow">
            <span className="hero-eyebrow-dot" aria-hidden="true" />
            <span>Real-time Sign Language Interpretation</span>
          </div>

          {/* Headline — two lines, individually animated */}
          <h1 className="hero-headline" id="hero-h1" aria-label="When hands speak, technology should listen.">
            <span className="hero-line" id="hero-line-1">
              When hands <em>speak,</em>
            </span>
            <span className="hero-line" id="hero-line-2">
              technology should listen.
            </span>
          </h1>

          <p className="hero-sub" id="hero-sub">
            Silent Interpreter transforms sign language into meaningful text and spoken language,
            helping conversations move naturally between people.
          </p>

          <div className="hero-actions" id="hero-actions">
            <button
              className="btn btn-hero btn-primary"
              onClick={onStartTranslation}
              aria-label="Start live translation demo"
              id="hero-cta-primary"
            >
              <Radio size={16} />
              Start Live Translation
            </button>
            <a href="#upload" className="btn btn-hero btn-ghost" aria-label="Upload a video">
              <Video size={15} />
              Upload a Video
            </a>
          </div>

          {/* Status */}
          <div className="hero-status" id="hero-status">
            <div style={{ paddingTop: 2 }}>
              <Dot
                color={
                  modelState === 'ready' ? 'green' :
                  modelState === 'error' ? 'record' :
                  modelState === 'loading' ? 'amber' : ''
                }
                pulse={modelState === 'ready' && isLiveModel && isRunning}
              />
            </div>
            <div className="hero-status-text">
              <strong>
                {modelState === 'ready' ? 'MODEL READY' :
                 modelState === 'loading' ? 'MODEL LOADING…' :
                 modelState === 'error' ? 'MODEL ERROR' :
                 'INTERPRETER READY'}
              </strong>
              {modelState === 'ready'
                ? (isLiveModel && isRunning ? 'Live ISL recognition active' : 'ISL model loaded · 261 signs')
                : (modelState === 'loading' ? 'Loading TensorFlow.js model…' :
                   modelState === 'error' ? 'Failed to load model — check network' :
                   'Starting…')}
            </div>
          </div>
        </div>

        {/* Right: Product */}
        <div className="hero-product" id="hero-product">
          <CameraPreview
            isRunning={isRunning}
            result={result}
            camState={camState}
            setCamState={setCamState}
            onCamera={onCamera}
            videoRef={videoRef}
            streamRef={streamRef}
            elapsed={elapsed}
            camError={camError}
          />
          <InterpretationCard
            isRunning={isRunning}
            result={result}
            lang={lang}
            onLang={onLang}
            pipelineStage={pipelineStage}
            modelState={modelState}
            isLiveModel={isLiveModel}
            bufferFill={bufferFill}
          />
          {/* DemoGestureSelector only shown in demo mode */}
          {!isLiveModel && (
            <DemoGestureSelector selected={selectedGesture} onChange={onGestureChange} />
          )}
        </div>
      </div>
    </section>
  )
}

/* ═══════════════════════════════════════════════════════
   BELOW-FOLD SECTIONS
═══════════════════════════════════════════════════════ */

function ProblemSection() {
  return (
    <section id="problem" aria-labelledby="problem-h2">
      <div className="wrap">
        <div className="problem-layout">
          <div>
            <SectionLabel>THE PROBLEM WE ARE SOLVING</SectionLabel>
            <h2 className="problem-h2 reveal" id="problem-h2">
              One in seventy million.<br />
              <em>Every single day.</em>
            </h2>
            <p className="problem-body reveal reveal-delay-1">
              India has the world&apos;s largest deaf and hard-of-hearing community.
              Most face communication barriers at hospitals, schools, and public
              spaces — not because of their disability, but because of ours.
            </p>
          </div>
          <div className="stats-grid reveal reveal-delay-2">
            <div className="stat-card">
              <p className="stat-num"><em>70M+</em></p>
              <p className="stat-label">deaf and hard-of-hearing people in India — the world's largest such community</p>
            </div>
            <div className="stat-card">
              <p className="stat-num">300<em>+</em></p>
              <p className="stat-label">unique hand shapes in Indian Sign Language</p>
            </div>
            <div className="stat-card">
              <p className="stat-num">1<em>in</em>6</p>
              <p className="stat-label">deaf Indians with access to a trained sign language interpreter</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

/* ─── MISSION / STATEMENT SECTION ──────────────────── */
function MissionSection() {
  return (
    <section id="mission" className="statement-section" aria-labelledby="mission-h2">
      <div className="wrap">
        <span className="section-label reveal" style={{ textAlign: 'center', display: 'block' }}>THE HUMAN MISSION</span>
        <div className="statement-lines" aria-label="Communication should never depend on who can speak.">
          <span className="statement-line reveal" id="mission-h2">Communication</span>
          <span className="statement-line reveal reveal-delay-1" style={{ color: 'var(--accent)' }}>should never</span>
          <span className="statement-line reveal reveal-delay-2">depend on who</span>
          <span className="statement-line reveal reveal-delay-3">can speak.</span>
        </div>
        <p className="statement-sub reveal reveal-delay-4">
          Silent Interpreter is built to make communication more accessible, natural and immediate —
          so that language is never a barrier between people.
        </p>
      </div>
    </section>
  )
}

function HowItWorksSection({
  activeDataset,
  setActiveDataset,
  techExpanded,
  setTechExpanded,
}: {
  activeDataset: string | null
  setActiveDataset: (s: string | null) => void
  techExpanded: boolean
  setTechExpanded: (b: boolean) => void
}) {
  const STEPS = [
    { num: '01', label: 'Capture',     sub: 'Your camera captures the signing in real time.',           highlight: false },
    { num: '02', label: 'Understand',  sub: 'MediaPipe extracts 21 hand landmarks per frame.',          highlight: false },
    { num: '03', label: 'Interpret',   sub: 'LSTM network maps the gesture sequence to meaning.',       highlight: true  },
    { num: '04', label: 'Generate',    sub: 'Meaning becomes readable English and Hindi text.',         highlight: false },
    { num: '05', label: 'Speak',       sub: 'IndicTrans2 + Web Speech API delivers the spoken word.',  highlight: false },
  ]

  return (
    <section id="how-it-works" aria-labelledby="hiw-h2">
      <div className="wrap">
        <div className="hiw-layout">
          <div>
            <SectionLabel>HOW IT WORKS</SectionLabel>
            <div className="section-head">
              <h2 id="hiw-h2" className="reveal">
                Technology is the proof,<br /><em>not the story.</em>
              </h2>
              <p className="reveal reveal-delay-1">
                Five stages from a gesture you make to a word someone hears.
              </p>
            </div>
            <button
              className={`tech-expand-btn ${techExpanded ? 'open' : ''}`}
              onClick={() => setTechExpanded(!techExpanded)}
              aria-expanded={techExpanded}
              aria-controls="tech-pipeline"
            >
              {techExpanded ? 'Hide technical pipeline' : 'See technical pipeline'}
              <ChevronDown size={14} />
            </button>
            <div
              id="tech-pipeline"
              className={`tech-full-panel reveal reveal-delay-2 ${techExpanded ? 'open' : ''}`}
              aria-hidden={!techExpanded}
            >
              <div className="full-pipeline">
                {PIPELINE_FULL.map(n => (
                  <div className="full-pipe-item" key={n.label}>
                    <span className={`full-pipe-item-dot ${n.cat}`} />
                    <span className="full-pipe-item-label">{n.label}</span>
                    <span className="full-pipe-item-sub">{n.sub}</span>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: 12, paddingTop: 10, borderTop: '1px solid var(--line)', display: 'flex', gap: 8 }}>
                <Badge sim>Future inference pipeline</Badge>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--muted)', letterSpacing: '.03em' }}>Simulated visualization only</span>
              </div>
            </div>
          </div>
          <div>
            <div className="how-pipeline-steps reveal reveal-delay-1">
              {STEPS.map(s => (
                <div key={s.num} className="how-step">
                  <span className="how-step-num">{s.num}</span>
                  <div className="how-step-body">
                    <span className="how-step-title" style={{ color: s.highlight ? 'var(--accent)' : undefined }}>{s.label}</span>
                    <span className="how-step-desc">{s.sub}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function ResearchSection({
  activeDataset,
  setActiveDataset,
}: {
  activeDataset: string | null
  setActiveDataset: (s: string | null) => void
}) {
  return (
    <section id="research" aria-labelledby="research-h2">
      <div className="wrap">
        <div className="section-head">
          <SectionLabel>TRAINED ON ISL RESEARCH</SectionLabel>
          <h2 id="research-h2" className="reveal">Open knowledge, <em>carefully held.</em></h2>
          <p className="reveal reveal-delay-1">
            These datasets form the research foundation that will teach a future interpreter the depth of Indian Sign Language.
          </p>
        </div>
        <div className="research-cards reveal reveal-delay-2">
          {DATASETS.map(ds => {
            const open = activeDataset === ds.name
            return (
              <article className={`r-card ${open ? 'open' : ''}`} key={ds.name}>
                <button
                  className="r-trigger"
                  onClick={() => setActiveDataset(open ? null : ds.name)}
                  aria-expanded={open}
                  aria-label={`${open ? 'Collapse' : 'Expand'} ${ds.name} dataset`}
                >
                  <span className="r-tab">{ds.mark}</span>
                  <span className="r-trigger-text">
                    <strong>{ds.name}</strong>
                    <small>{ds.meta}</small>
                  </span>
                  <ChevronDown size={14} />
                </button>
                <div className="r-body" aria-hidden={!open}>
                  <div className="r-inner">
                    <p>{ds.purpose}</p>
                    <span>{ds.role}</span>
                  </div>
                </div>
              </article>
            )
          })}
        </div>
      </div>
    </section>
  )
}

function RoadmapSection() {
  return (
    <section id="roadmap" aria-labelledby="roadmap-h2">
      <div className="wrap">
        <div className="roadmap-layout">
          <div>
            <SectionLabel>THE ROAD AHEAD</SectionLabel>
            <div className="section-head">
              <h2 id="roadmap-h2" className="reveal">
                From a first step to<br /><em>universal communication.</em>
              </h2>
              <p className="reveal reveal-delay-1">
                A staircase, not a promise. Each level is a direction for future research.
              </p>
            </div>
            <div style={{ marginTop: 16 }}>
              <Badge sim>Roadmap Simulation</Badge>
            </div>
          </div>
          <div className="roadmap-steps reveal reveal-delay-1">
            {ROADMAP.map((item, i) => (
              <div key={item.label} className={`roadmap-step ${item.active ? 'active' : ''}`}>
                <span className="roadmap-step-num">{String(i + 1).padStart(2, '0')}</span>
                <span className="roadmap-step-text">{item.label}</span>
                {item.active && <span className="roadmap-step-badge">NOW</span>}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

function ReplaySection({
  replays,
  lang,
  replayPlaying,
  setReplayPlaying,
}: {
  replays: Array<{ time: string; gesture: string; english: string; hindi: string; confidence: number; emoji: string }>
  lang: 'english' | 'hindi'
  replayPlaying: number | null
  setReplayPlaying: (i: number | null) => void
}) {
  return (
    <section id="replay" aria-labelledby="replay-h2">
      <div className="wrap">
        <div className="replay-layout">
          <div>
            <SectionLabel>CONVERSATION REPLAY</SectionLabel>
            <div className="section-head">
              <h2 id="replay-h2" className="reveal">
                Nothing meaningful has to<br /><em>disappear.</em>
              </h2>
              <p className="reveal reveal-delay-1">
                Every gesture you simulate stays as a local session memory.
              </p>
            </div>
          </div>
          <div className="replay-list reveal reveal-delay-2">
            {replays.length === 0 ? (
              <div className="replay-empty">
                <Waves size={24} style={{ color: 'var(--accent)' }} />
                <p>Start the live demo to create your first replay.</p>
                <Badge sim>Demo Simulation</Badge>
              </div>
            ) : replays.map((r, i) => (
              <div className="replay-item" key={`${r.time}-${i}`}>
                <span className="replay-time">{r.time}</span>
                <div className="replay-info">
                  <strong>{r.emoji} {r.gesture}</strong>
                  <p>{lang === 'english' ? r.english : r.hindi}</p>
                </div>
                <span className="replay-pct">{r.confidence}%</span>
                <button
                  className="btn btn-icon"
                  style={{ marginLeft: 4 }}
                  onClick={() => { setReplayPlaying(i); setTimeout(() => setReplayPlaying(null), 1400) }}
                  aria-label={`Replay ${r.gesture}`}
                >
                  {replayPlaying === i ? <Pause size={12} /> : <Play size={12} fill="currentColor" />}
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

function UploadSection({
  ctx,
  uploadState,
  setUploadState,
  fileName,
  setFileName,
  chipsVisible,
  setChipsVisible,
  waveHeights,
}: {
  ctx: RecognitionResult | null
  uploadState: 'idle' | 'processing' | 'ready'
  setUploadState: (s: 'idle' | 'processing' | 'ready') => void
  fileName: string
  setFileName: (s: string) => void
  chipsVisible: boolean[]
  setChipsVisible: (v: boolean[] | ((prev: boolean[]) => boolean[])) => void
  waveHeights: number[]
}) {
  const fileRef = useRef<HTMLInputElement>(null)

  function handleFile(file?: File) {
    if (!file) return
    setFileName(file.name)
    setUploadState('processing')
    setChipsVisible([])
    const chips = ['Open Palm', 'Handshake', 'Point', 'Fist', 'Five']
    chips.forEach((_, i) => setTimeout(() => setChipsVisible(v => [...v, true]), 400 + i * 350))
    setTimeout(() => setUploadState('ready'), 2400)
  }

  return (
    <section id="upload" aria-labelledby="upload-h2">
      <div className="wrap">
        <div className="upload-layout">
          <div>
            <SectionLabel>CONTINUE A CONVERSATION</SectionLabel>
            <div className="section-head">
              <h2 id="upload-h2" className="reveal">Let a video become a <em>timeline.</em></h2>
              <p className="reveal reveal-delay-1">Drop a local clip to explore the shape of a future upload flow.</p>
            </div>
            <ul style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 8 }}>
              {['Local-only processing', 'Confidence and waveform preview', 'Transcript download preview'].map(item => (
                <li key={item} style={{ display: 'flex', alignItems: 'center', gap: 9, fontSize: 14, color: 'var(--muted)' }}>
                  <Check size={13} style={{ color: 'var(--accent)', flexShrink: 0 }} /> {item}
                </li>
              ))}
            </ul>
          </div>
          <div
            className="drop-zone reveal reveal-delay-2"
            onDragOver={e => e.preventDefault()}
            onDrop={e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]) }}
            aria-label="Video file drop area"
          >
            <input ref={fileRef} type="file" accept="video/*" hidden onChange={e => handleFile(e.target.files?.[0])} />
            {uploadState === 'idle' && (
              <button className="drop-idle" onClick={() => fileRef.current?.click()}>
                <div style={{ width: 48, height: 48, border: '1px dashed var(--accent)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent)' }}>
                  <CloudUpload size={24} />
                </div>
                <strong>Drop a video here</strong>
                <span>or browse for a local clip</span>
                <Badge sim>UI Demo</Badge>
              </button>
            )}
            {uploadState === 'processing' && (
              <div className="drop-processing" role="status" aria-live="polite">
                <Sparkles size={24} style={{ color: 'var(--accent)' }} />
                <strong style={{ fontSize: 15, color: 'var(--text)' }}>Building simulated timeline…</strong>
                <span style={{ fontSize: 13, color: 'var(--muted)' }}>{fileName}</span>
                <div className="upload-progress"><div className="upload-progress-fill" /></div>
              </div>
            )}
            {uploadState === 'ready' && (
              <div className="drop-ready" aria-live="polite">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ width: 24, height: 24, borderRadius: '50%', background: 'var(--success)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Check size={13} color="#fff" />
                  </span>
                  <strong style={{ fontSize: 14, color: 'var(--text)' }}>Timeline ready</strong>
                  <button onClick={() => { setUploadState('idle'); setFileName(''); setChipsVisible([]) }} aria-label="Reset upload" style={{ marginLeft: 'auto', color: 'var(--muted)', cursor: 'pointer', background: 'none', border: 'none', display: 'flex' }}>
                    <X size={14} />
                  </button>
                </div>
                <div className="gesture-chips-row">
                  {['👋 Open Palm', '🤝 Handshake', '☝️ Point', '✊ Fist', '🖐️ Five'].map((chip, i) => (
                    <span key={chip} className={`gchip ${chipsVisible[i] ? 'show' : ''}`} style={{ transitionDelay: `${i * .07}s` }}>{chip}</span>
                  ))}
                </div>
                <div className="transcript-box">
                  <small>RECOGNIZED TEXT</small>
                  <p style={{ fontSize: 14, color: 'var(--text)' }}>{ctx?.english ?? 'Video upload recognition is not implemented.'}</p>
                </div>
                <button className="btn btn-sm btn-ghost" disabled aria-label="Download transcript (disabled)">
                  <Upload size={12} /> Download transcript preview
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}

function CtaSection({ onCamera }: { onCamera: () => void }) {
  return (
    <section className="cta-section" aria-labelledby="cta-h2">
      <div className="wrap" style={{ textAlign: 'center' }}>
        <SectionLabel>LET COMMUNICATION MOVE FREELY</SectionLabel>
        <div className="cta-headline-wrap" aria-label="Every person deserves to be understood.">
          <span className="cta-line" id="cta-h2">Every person</span>
          <span className="cta-line">deserves</span>
          <span className="cta-line">to be <em>understood.</em></span>
        </div>
        <p className="cta-sub">
          Silent Interpreter is built for the moments when being heard matters most.
          A future where language is never a barrier.
        </p>
        <div className="cta-actions">
          <button className="btn btn-hero btn-primary" onClick={onCamera} aria-label="Experience the demo">
            <Video size={16} /> Start Live Translation
          </button>
          <a href="#upload" className="btn btn-hero btn-ghost">
            <Play size={15} fill="currentColor" /> Upload a Video
          </a>
        </div>
      </div>
    </section>
  )
}

/* ═══════════════════════════════════════════════════════
   NAVIGATION
═══════════════════════════════════════════════════════ */
function Navigation({
  mobileOpen,
  setMobileOpen,
  onStart,
}: {
  mobileOpen: boolean
  setMobileOpen: (b: boolean) => void
  onStart: () => void
}) {
  return (
    <nav className="nav" aria-label="Main navigation">
      <a href="#top" className="nav-brand" aria-label="Silent Interpreter home">
        <NavMark />
        Silent Interpreter
      </a>
      <div className={`nav-links ${mobileOpen ? 'open' : ''}`}>
        <a href="#problem" onClick={() => setMobileOpen(false)}>Product</a>
        <a href="#how-it-works" onClick={() => setMobileOpen(false)}>How It Works</a>
        <a href="#research" onClick={() => setMobileOpen(false)}>Research</a>
        <a href="#roadmap" onClick={() => setMobileOpen(false)}>About</a>
        <a href="#top" className="nav-cta" onClick={() => { setMobileOpen(false); onStart() }}>
          Start Translation <ArrowRight size={13} />
        </a>
      </div>
      <div className="nav-right">
        <button className="nav-cta" onClick={onStart} aria-label="Start translation">
          Start Translation <ArrowRight size={13} />
        </button>
        <button
          className="nav-mobile-btn"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label={mobileOpen ? 'Close menu' : 'Open menu'}
          aria-expanded={mobileOpen}
        >
          {mobileOpen ? <X size={18} /> : <Menu size={18} />}
        </button>
      </div>
    </nav>
  )
}

/* ═══════════════════════════════════════════════════════
   ROOT PAGE COMPONENT
═══════════════════════════════════════════════════════ */
export default function Page() {
  useEffect(() => {
    if (typeof window === 'undefined') return
    const originalLog = console.log
    const originalError = console.error
    console.log = function(...args) {
      if (args[0] && typeof args[0] === 'string' && args[0].includes('[ISL]')) {
        fetch('http://localhost:4000', { method: 'POST', body: args.join(' ') }).catch(() => {})
      }
      originalLog.apply(console, args)
    }
    console.error = function(...args) {
      if (args[0] && typeof args[0] === 'string' && args[0].includes('[ISL]')) {
        fetch('http://localhost:4000', { method: 'POST', body: 'ERROR: ' + args.join(' ') }).catch(() => {})
      }
      originalError.apply(console, args)
    }
    setTimeout(() => {
      const btn = document.querySelector('.btn-primary')
      if (btn) (btn as HTMLButtonElement).click()
    }, 2000)
  }, [])
  const [selectedGesture, setSelectedGesture] = useState('open-palm')
  const [isRunning, setIsRunning] = useState(false)
  const [lang, setLang] = useState<'english' | 'hindi'>('english')
  const [camState, setCamState] = useState<CamState>('idle')
  const [camError, setCamError] = useState<CamError>(null)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [activeDataset, setActiveDataset] = useState<string | null>(null)
  const [techExpanded, setTechExpanded] = useState(false)
  const [uploadState, setUploadState] = useState<'idle' | 'processing' | 'ready'>('idle')
  const [fileName, setFileName] = useState('')
  const [chipsVisible, setChipsVisible] = useState<boolean[]>([])
  const [replays, setReplays] = useState<Array<{ time: string; gesture: string; english: string; hindi: string; confidence: number; emoji: string }>>([])
  const [replayPlaying, setReplayPlaying] = useState<number | null>(null)
  const [pipelineStage, setPipelineStage] = useState(-1)

  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const pipeTimers = useRef<ReturnType<typeof setTimeout>[]>([])

  const {
    prediction: mlPrediction,
    mvpPrediction,
    modelState,
    mvpModelState,
    bufferFill,
  } = useISLRecognition(isRunning, videoRef)

  // isLiveModel = model is ready AND camera has been granted at any point.
  // Includes 'stopped' so that after recording ends we don't revert to demo sentences.
  // Does NOT include 'idle'/'requesting'/'error' — camera must have been successfully opened.
  const isLiveModel = (modelState === 'ready' || mvpModelState === 'ready') && (
    camState === 'active' || camState === 'recording' || camState === 'stopped'
  )

  // ─── Adapt real ML prediction to RecognitionResult shape ──
  // The live result comes from the ML pipeline when available;
  // falls back to null (not demo data) in live mode.
  // english field = the raw ISL label from labels.json (cleaned of numeric prefix).
  const liveResult = useMemo((): RecognitionResult | null => {
    // Prefer MVP prediction if available and valid
    if (mvpPrediction && mvpPrediction.label !== 'UNKNOWN' && mvpPrediction.label !== 'Uncertain sign') {
      const display = getDisplayLabel(mvpPrediction.label)
      return {
        id: `ml-mvp`,
        gesture: display.gesture,
        emoji: display.emoji,
        english: display.english === 'UNCERTAIN' ? '' : display.english,
        hindi: display.hindi === 'UNCERTAIN' ? '' : display.hindi,
        confidence: mvpPrediction.confidence,
        context: 'Live (MVP)',
      }
    }
    
    // Fall back to INCLUDE prediction
    if (!mlPrediction) return null
    const display = getDisplayLabel(mlPrediction.label)
    return {
      id: `ml-${mlPrediction.index}`,
      gesture: display.gesture,
      emoji: display.emoji,
      english: display.english === 'UNCERTAIN' ? '' : display.english,
      hindi: display.hindi === 'UNCERTAIN' ? '' : display.hindi,
      confidence: mlPrediction.confidence,
      context: 'Live',
    }
  }, [mlPrediction, mvpPrediction])

  // ─── Demo result (for demo selector — NEVER used in live mode) ────
  // ─── Displayed result depends on mode ──────────────────────────────
  // LIVE MODE (model ready + camera active/stopped): always use real ML prediction.
  //   If mlPrediction is null (buffer filling), result=null → UI shows 'Waiting for sign…'
  //   NEVER fall back to demoResult in live mode.
  // DEMO MODE (camera idle or model not ready): use demoResult.
  // A live session never falls back to simulated recognition data.
  const result: RecognitionResult | null = isRunning && isLiveModel ? liveResult : null
  const waveHeights = useMemo(() => Array.from({ length: 20 }, (_, i) => 8 + ((i * 13) % 28)), [])

  // ─── Scroll-reveal via IntersectionObserver ───────────
  useEffect(() => {
    if (typeof window === 'undefined') return
    const els = document.querySelectorAll<HTMLElement>('.reveal')
    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible')
            observer.unobserve(entry.target)
          }
        })
      },
      { threshold: 0.1 }
    )
    els.forEach(el => observer.observe(el))
    return () => observer.disconnect()
  }, [])

  // ─── GSAP Cinematic Entrance ──────────────────────────
  useEffect(() => {
    if (typeof window === 'undefined') return
    const init = async () => {
      const { gsap } = await import('gsap')
      const { ScrollTrigger } = await import('gsap/ScrollTrigger')
      gsap.registerPlugin(ScrollTrigger)

      // Respect prefers-reduced-motion
      const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
      if (prefersReduced) {
        // Just make everything visible immediately
        gsap.set([
          '.nav', '#hero-eyebrow', '#hero-line-1', '#hero-line-2',
          '#hero-sub', '#hero-actions', '#hero-status', '#hero-product',
          '.bg-grid-v', '.bg-grid-h',
        ], { opacity: 1, y: 0, scaleY: 1, scaleX: 1 })
        return
      }

      // ── Background grid wipe-in ────────────────────────
      gsap.set('.bg-grid-v', { scaleY: 0, transformOrigin: 'top' })
      gsap.set('.bg-grid-h', { scaleX: 0, transformOrigin: 'left' })
      gsap.to('.bg-grid-v', { scaleY: 1, duration: 1.6, ease: 'power2.out', delay: 0.1 })
      gsap.to('.bg-grid-h', { scaleX: 1, duration: 1.8, ease: 'power2.out', delay: 0.2 })

      // ── Hero entrance choreography ──────────────────────
      const tl = gsap.timeline({ defaults: { ease: 'power3.out' } })

      // Nav
      tl.fromTo('.nav',
        { opacity: 0, y: -12 },
        { opacity: 1, y: 0, duration: 0.6 },
        0.15
      )

      // Eyebrow
      tl.fromTo('#hero-eyebrow',
        { opacity: 0, y: 16 },
        { opacity: 1, y: 0, duration: 0.65 },
        0.4
      )

      // Headline line 1
      tl.fromTo('#hero-line-1',
        { opacity: 0, y: 32 },
        { opacity: 1, y: 0, duration: 0.75 },
        0.55
      )

      // Headline line 2
      tl.fromTo('#hero-line-2',
        { opacity: 0, y: 32 },
        { opacity: 1, y: 0, duration: 0.75 },
        0.7
      )

      // Sub text
      tl.fromTo('#hero-sub',
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.65 },
        0.85
      )

      // CTA row
      tl.fromTo('#hero-actions',
        { opacity: 0, y: 16 },
        { opacity: 1, y: 0, duration: 0.6 },
        0.98
      )

      // Status
      tl.fromTo('#hero-status',
        { opacity: 0 },
        { opacity: 1, duration: 0.5 },
        1.1
      )

      // Product panel (camera + interp + selector)
      tl.fromTo('#hero-product',
        { opacity: 0, y: 28 },
        { opacity: 1, y: 0, duration: 0.9 },
        0.85
      )

      // ── ScrollTrigger for below-fold sections ──────────
      // Statement section lines (special — staggered within)
      const statementLines = document.querySelectorAll('.statement-line')
      if (statementLines.length) {
        gsap.fromTo(statementLines,
          { opacity: 0, y: 36 },
          {
            opacity: 1, y: 0,
            duration: 0.75,
            stagger: 0.12,
            ease: 'power3.out',
            scrollTrigger: {
              trigger: '#mission',
              start: 'top 72%',
              toggleActions: 'play none none none',
            }
          }
        )
      }

      // CTA section lines
      const ctaLines = document.querySelectorAll('.cta-line')
      if (ctaLines.length) {
        gsap.fromTo(ctaLines,
          { opacity: 0, y: 32 },
          {
            opacity: 1, y: 0,
            duration: 0.7,
            stagger: 0.1,
            ease: 'power3.out',
            scrollTrigger: {
              trigger: '.cta-section',
              start: 'top 75%',
              toggleActions: 'play none none none',
            }
          }
        )
        gsap.fromTo('.cta-sub, .cta-actions',
          { opacity: 0 },
          {
            opacity: 1, duration: 0.6,
            stagger: 0.12,
            ease: 'power2.out',
            scrollTrigger: {
              trigger: '.cta-section',
              start: 'top 65%',
              toggleActions: 'play none none none',
            }
          }
        )
      }

      // How-it-works steps — stagger reveal
      const howSteps = document.querySelectorAll('.how-step')
      if (howSteps.length) {
        gsap.fromTo(howSteps,
          { opacity: 0, x: 24 },
          {
            opacity: 1, x: 0,
            duration: 0.55,
            stagger: 0.1,
            ease: 'power2.out',
            scrollTrigger: {
              trigger: '#how-it-works',
              start: 'top 65%',
              toggleActions: 'play none none none',
            }
          }
        )
      }

      // Stat cards stagger
      const statCards = document.querySelectorAll('.stat-card')
      if (statCards.length) {
        gsap.fromTo(statCards,
          { opacity: 0, y: 20 },
          {
            opacity: 1, y: 0,
            duration: 0.6,
            stagger: 0.1,
            ease: 'power2.out',
            scrollTrigger: {
              trigger: '#problem',
              start: 'top 65%',
              toggleActions: 'play none none none',
            }
          }
        )
      }
    }
    init().catch(() => {})
  }, [])

  // ─── Pipeline animation when isRunning changes ────────
  useEffect(() => {
    pipeTimers.current.forEach(clearTimeout)
    pipeTimers.current = []
    setPipelineStage(-1)
    if (!isRunning) return
    // Animate pipeline stages with real events where possible
    const stages = [0, 1, 2, 3, 4]
    stages.forEach((stage, i) => {
      const t = setTimeout(() => setPipelineStage(stage), i * 420 + 200)
      pipeTimers.current.push(t)
    })
    return () => pipeTimers.current.forEach(clearTimeout)
  }, [isRunning])

  // ─── Camera cleanup on unmount ────────────────────────
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop())
        streamRef.current = null
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null
      }
    }
  }, [])

  // ─── Camera enable ─────────────────────────────────────
  const requestCamera = useCallback(async () => {
    // Allow requesting from idle or error state
    if (camState !== 'idle' && camState !== 'error') return
    // Stop any existing stream before re-requesting
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setCamError(null)
    setCamState('requesting')
    // Check getUserMedia support
    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      setCamError('unsupported')
      setCamState('error')
      return
    }
    try {
      let stream: MediaStream
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
          audio: true,
        })
      } catch {
        // Fall back to video-only if audio fails
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        })
      }
      streamRef.current = stream
      // The <video> element is always in the DOM — assign directly.
      // The useEffect in CameraPreview will also attach it when camState -> active.
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play().catch(() => {})
      }
      setCamState('active')
    } catch (err: unknown) {
      let code: CamError = 'unknown'
      if (err instanceof DOMException) {
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') code = 'denied'
        else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') code = 'notfound'
        else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') code = 'busy'
      }
      setCamError(code)
      setCamState('error')
    }
  }, [camState])

  // ─── Start translation ─────────────────────────────────
  function startTranslation() {
    if (!isRunning) {
      setIsRunning(true)
    }
  }

  function handleStartTranslation() {
    if (camState === 'idle') requestCamera()
    startTranslation()
    document.getElementById('hero-product')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }

  function handleGestureChange(id: string) {
    // Demo gesture selector only affects demo mode, NOT live model mode
    setSelectedGesture(id)
    // In live mode: do nothing — real model predictions control the UI
  }

  // ─── Add live prediction to replay list ────────────────
  const lastReplayedPrediction = useRef<string>('')
  useEffect(() => {
    if (!isLiveModel || !mlPrediction || !isRunning) return
    if (mlPrediction.label === lastReplayedPrediction.current) return
    if (mlPrediction.confidence < CONFIDENCE_THRESHOLD) return
    if (mlPrediction.label === 'UNCERTAIN') return
    lastReplayedPrediction.current = mlPrediction.label
    const t = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    setReplays(prev => [
      {
        time: t,
        gesture: mlPrediction.label,
        english: mlPrediction.label,
        hindi: '',
        confidence: Math.round(mlPrediction.confidence * 100),
        emoji: '🤙',
      },
      ...prev,
    ].slice(0, 8))
  }, [mlPrediction, isLiveModel, isRunning])

  return (
    <div style={{ position: 'relative' }}>
      {/* Animated background grid */}
      <div className="bg-canvas" aria-hidden="true">
        <div className="bg-grid-v" />
        <div className="bg-grid-h" />
      </div>

      <Navigation mobileOpen={mobileOpen} setMobileOpen={setMobileOpen} onStart={handleStartTranslation} />

      <main style={{ paddingTop: 0 }}>
        <HeroSection
          result={result}
          isRunning={isRunning}
          lang={lang}
          onLang={setLang}
          camState={camState}
          setCamState={setCamState}
          onCamera={requestCamera}
          videoRef={videoRef}
          streamRef={streamRef}
          onStartTranslation={handleStartTranslation}
          pipelineStage={pipelineStage}
          selectedGesture={selectedGesture}
          onGestureChange={handleGestureChange}
          camError={camError}
          modelState={modelState}
          isLiveModel={isLiveModel}
          bufferFill={bufferFill}
        />

        <ProblemSection />

        <MissionSection />

        <HowItWorksSection
          activeDataset={activeDataset}
          setActiveDataset={setActiveDataset}
          techExpanded={techExpanded}
          setTechExpanded={setTechExpanded}
        />

        <ResearchSection
          activeDataset={activeDataset}
          setActiveDataset={setActiveDataset}
        />

        <RoadmapSection />

        <ReplaySection
          replays={replays}
          lang={lang}
          replayPlaying={replayPlaying}
          setReplayPlaying={setReplayPlaying}
        />

        <UploadSection
          ctx={liveResult}
          uploadState={uploadState}
          setUploadState={setUploadState}
          fileName={fileName}
          setFileName={setFileName}
          chipsVisible={chipsVisible}
          setChipsVisible={setChipsVisible}
          waveHeights={waveHeights}
        />

        <CtaSection onCamera={handleStartTranslation} />
      </main>

      <footer className="footer" aria-label="Site footer">
        <div className="wrap">
          <div className="footer-inner">
            <a href="#top" className="footer-brand" aria-label="Back to top">
              <NavMark />
              Silent Interpreter
            </a>
            <div className="footer-links">
              <a href="https://github.com" target="_blank" rel="noreferrer">
                GitHub <ArrowUpRight size={11} />
              </a>
              <a href="#how-it-works">How It Works</a>
              <a href="#research">Research</a>
            </div>
          </div>
          <p className="footer-note">
            Built for hackathons · 2026 · UI Demo — No AI backend connected · Indian Sign Language Research Project
          </p>
        </div>
      </footer>
    </div>
  )
}
