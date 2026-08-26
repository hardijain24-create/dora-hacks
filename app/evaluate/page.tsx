'use client'

import { useState, useRef, useEffect } from 'react'
import { loadModel, runInference } from '@/lib/isl/model'
import { loadLabels, getLabels } from '@/lib/isl/labels'
import { initLandmarkers, detectLandmarks, getLandmarkState } from '@/lib/isl/landmarks'
import { extractFeatureVector, normalizeWindow, FEATURE_DIM } from '@/lib/isl/preprocessing'
import { WINDOW_SIZE } from '@/lib/isl/predictor'

interface TestResult {
  filename: string
  expected?: string
  predictedIndex?: number
  predictedLabel?: string
  confidence?: number
  correct?: boolean
  error?: string
}

export default function EvaluatePage() {
  const [results, setResults] = useState<TestResult[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [status, setStatus] = useState('Idle')
  const videoRef = useRef<HTMLVideoElement | null>(null)

  useEffect(() => {
    async function setup() {
      setStatus('Loading ML models...')
      await Promise.all([loadModel(), loadLabels(), initLandmarkers()])
      setStatus('Models loaded. Ready.')
    }
    setup()
  }, [])

  const processVideo = async (file: File): Promise<TestResult> => {
    return new Promise((resolve) => {
      const video = videoRef.current
      if (!video) return resolve({ filename: file.name, error: 'No video element' })

      const url = URL.createObjectURL(file)
      video.src = url
      video.muted = true
      
      const frames: Float32Array[] = []

      video.onloadeddata = () => {
        video.play()
      }

      const processFrame = () => {
        if (video.paused || video.ended) return

        if (getLandmarkState() === 'ready') {
          // Force a monotonic timestamp
          const ts = performance.now()
          const lmResult = detectLandmarks(video, ts)
          const feature = extractFeatureVector(lmResult)
          frames.push(feature)
        }

        if (video.currentTime < video.duration) {
          // Use requestVideoFrameCallback if available, fallback to requestAnimationFrame
          if ('requestVideoFrameCallback' in video) {
            (video as any).requestVideoFrameCallback(processFrame)
          } else {
            requestAnimationFrame(processFrame)
          }
        }
      }

      video.onplay = () => {
        frames.length = 0
        if ('requestVideoFrameCallback' in video) {
          (video as any).requestVideoFrameCallback(processFrame)
        } else {
          requestAnimationFrame(processFrame)
        }
      }

      video.onended = async () => {
        URL.revokeObjectURL(url)
        
        if (frames.length === 0) {
          return resolve({ filename: file.name, error: 'No frames extracted' })
        }

        // Resample to EXACTLY 40 frames
        const sampledFrames: Float32Array[] = []
        for (let i = 0; i < WINDOW_SIZE; i++) {
          const idx = Math.floor((i / (WINDOW_SIZE - 1)) * (frames.length - 1))
          sampledFrames.push(frames[idx || 0])
        }

        try {
          const normalized = normalizeWindow(sampledFrames)
          const probabilities = await runInference(normalized)
          
          let maxIdx = 0
          let maxVal = probabilities[0]
          for (let i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxVal) {
              maxVal = probabilities[i]
              maxIdx = i
            }
          }

          const labels = getLabels()
          const predictedLabel = labels ? labels[maxIdx] : 'Unknown'
          
          // Guess expected label from filename if it starts with a number, e.g., "50. Yellow.mp4"
          // Or we just let the user see what it is
          const expectedMatch = file.name.match(/^(\d+)/)
          let expectedStr = expectedMatch ? `${expectedMatch[1]}` : file.name

          // In this simple test, we just output the raw predicted index and string.
          // True/False correctness depends on if the user named the file correctly.
          const isCorrect = file.name.toLowerCase().includes(predictedLabel.split('.')[1]?.trim().toLowerCase() || 'impossible_match')

          resolve({
            filename: file.name,
            expected: expectedStr,
            predictedIndex: maxIdx,
            predictedLabel: predictedLabel,
            confidence: maxVal,
            correct: isCorrect
          })

        } catch (err) {
          resolve({ filename: file.name, error: String(err) })
        }
      }

      video.onerror = (e) => {
        resolve({ filename: file.name, error: 'Video load error' })
      }
    })
  }

  const handleFiles = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.length) return
    const files = Array.from(e.target.files)
    
    setIsProcessing(true)
    setResults([])
    
    for (const file of files) {
      setStatus(`Processing ${file.name}...`)
      const res = await processVideo(file)
      setResults(prev => [...prev, res])
    }
    
    setStatus('Evaluation complete.')
    setIsProcessing(false)
  }

  return (
    <div style={{ padding: '40px', fontFamily: 'sans-serif' }}>
      <h1>INCLUDE Model Evaluation</h1>
      <p>Select 5 real INCLUDE dataset videos to verify model/dataset parity.</p>
      
      <div style={{ marginBottom: '20px' }}>
        <input 
          type="file" 
          multiple 
          accept="video/mp4,video/webm" 
          onChange={handleFiles} 
          disabled={isProcessing || status !== 'Models loaded. Ready.' && status !== 'Evaluation complete.'}
        />
      </div>

      <p><strong>Status:</strong> {status}</p>

      {/* Hidden video element for processing */}
      <video 
        ref={videoRef} 
        style={{ display: 'none' }} 
        playsInline 
        muted 
      />

      {results.length > 0 && (
        <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '20px' }}>
          <thead>
            <tr style={{ textAlign: 'left', borderBottom: '2px solid #ccc' }}>
              <th style={{ padding: '8px' }}>Filename</th>
              <th style={{ padding: '8px' }}>Predicted Index</th>
              <th style={{ padding: '8px' }}>Predicted Label</th>
              <th style={{ padding: '8px' }}>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #eee' }}>
                <td style={{ padding: '8px' }}>{r.filename}</td>
                {r.error ? (
                  <td colSpan={3} style={{ padding: '8px', color: 'red' }}>{r.error}</td>
                ) : (
                  <>
                    <td style={{ padding: '8px' }}>{r.predictedIndex}</td>
                    <td style={{ padding: '8px' }}>{r.predictedLabel}</td>
                    <td style={{ padding: '8px' }}>{r.confidence?.toFixed(4)}</td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
