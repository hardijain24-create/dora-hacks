import { FrameLandmarks } from './types'

// TEMPORARY FILE FOR DEMO SUBMISSION ONLY
// Implements a deterministic heuristic for 5 gestures:
// HELLO, THANK_YOU, SORRY, YES, NO

const HISTORY_SIZE = 40; // ~1.3 seconds at 30fps
let history: any[] = [];
let lastRecognizedTime = 0;

function dist2d(p1: [number, number], p2: [number, number]) {
  return Math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2);
}

function extractHandState(handData: Float32Array | null, isRight: boolean) {
  if (!handData) return null;
  // Wrist is index 0 (x,y,z) -> 0,1,2
  const wrist = [handData[0], handData[1]] as [number, number];
  const index = [handData[8*3], handData[8*3+1]] as [number, number];
  const middle = [handData[12*3], handData[12*3+1]] as [number, number];
  const ring = [handData[16*3], handData[16*3+1]] as [number, number];
  const pinky = [handData[20*3], handData[20*3+1]] as [number, number];

  const avgDist = (dist2d(wrist, index) + dist2d(wrist, middle) + dist2d(wrist, ring) + dist2d(wrist, pinky)) / 4.0;
  // Normalized approx: hand spans ~0.15 of screen width. Closed fist is < 0.08
  const isOpen = avgDist > 0.08;
  const isClosed = avgDist < 0.06;

  return { wrist, isOpen, isClosed };
}

export function evaluateHeuristic(lm: FrameLandmarks, now: number) {
  if (now - lastRecognizedTime < 2000) {
    return { gesture: 'UNCERTAIN', confidence: 0 }; // 2-second debounce
  }

  // Parse current frame
  let hand = null;
  let pose = null;

  if (lm.pose) {
    // 33 pose landmarks x 4 (x,y,z,vis)
    pose = {
      nose: [lm.pose[0], lm.pose[1]] as [number, number],
      leftShoulder: [lm.pose[11*4], lm.pose[11*4+1]] as [number, number],
      rightShoulder: [lm.pose[12*4], lm.pose[12*4+1]] as [number, number]
    };
  }

  // Use the dominant/visible hand
  const rightState = extractHandState(lm.rightHand, true);
  const leftState = extractHandState(lm.leftHand, false);
  
  if (rightState) hand = rightState;
  else if (leftState) hand = leftState;

  history.push({ hand, pose, t: now });
  if (history.length > HISTORY_SIZE) {
    history.shift();
  }

  if (history.length < 15) return { gesture: 'UNCERTAIN', confidence: 0 };

  // Analyze history
  let openFrames = 0;
  let closedFrames = 0;
  let minX = 999, maxX = -999;
  let minY = 999, maxY = -999;
  let totalDx = 0, totalDy = 0;
  
  // Track movement reversals (oscillations)
  let reversalsX = 0;
  let reversalsY = 0;
  let lastDirX = 0;
  let lastDirY = 0;

  let validFrames = 0;
  let avgWristX = 0, avgWristY = 0;

  for (let i = 0; i < history.length; i++) {
    const h = history[i].hand;
    if (h) {
      validFrames++;
      if (h.isOpen) openFrames++;
      if (h.isClosed) closedFrames++;
      
      avgWristX += h.wrist[0];
      avgWristY += h.wrist[1];
      
      minX = Math.min(minX, h.wrist[0]);
      maxX = Math.max(maxX, h.wrist[0]);
      minY = Math.min(minY, h.wrist[1]);
      maxY = Math.max(maxY, h.wrist[1]);

      if (i > 0 && history[i-1].hand) {
        const dx = h.wrist[0] - history[i-1].hand.wrist[0];
        const dy = h.wrist[1] - history[i-1].hand.wrist[1];
        totalDx += Math.abs(dx);
        totalDy += Math.abs(dy);
        
        // Track reversals
        if (Math.abs(dx) > 0.01) {
          const dirX = Math.sign(dx);
          if (lastDirX !== 0 && dirX !== lastDirX) reversalsX++;
          lastDirX = dirX;
        }
        if (Math.abs(dy) > 0.01) {
          const dirY = Math.sign(dy);
          if (lastDirY !== 0 && dirY !== lastDirY) reversalsY++;
          lastDirY = dirY;
        }
      }
    }
  }

  if (validFrames < 15) return { gesture: 'UNCERTAIN', confidence: 0 };

  avgWristX /= validFrames;
  avgWristY /= validFrames;

  const rangeX = maxX - minX;
  const rangeY = maxY - minY;
  const pathRatioX = totalDx / (rangeX + 0.001); // High if back-and-forth
  const pathRatioY = totalDy / (rangeY + 0.001);

  const isOpen = openFrames > validFrames * 0.6;
  const isClosed = closedFrames > validFrames * 0.6;
  
  const currentPose = history[history.length - 1].pose;
  if (!currentPose) return { gesture: 'UNCERTAIN', confidence: 0 };

  const faceY = currentPose.nose[1];
  const chinY = faceY + 0.1;
  const chestY = (currentPose.leftShoulder[1] + currentPose.rightShoulder[1]) / 2.0;

  let bestMatch = 'UNCERTAIN';
  let conf = 0;

  // 1. HELLO: Open hand + near face + lateral waving
  const isNearFace = avgWristY < chinY;
  if (isOpen && isNearFace && rangeX > 0.15 && reversalsX >= 1 && pathRatioX > 1.5) {
    bestMatch = 'HELLO';
    conf = 95;
  }
  // 2. THANK YOU: Open hand + near chin/mouth moving outward (forward/down)
  // Usually translates to moving downward or scaling. We'll use Y movement away from face.
  // Actually, outward in 2D often looks like moving down/away from center.
  // Let's check if started near chin and moved down.
  else if (isOpen && rangeY > 0.1) {
    const firstValid = history.find(h => h.hand);
    const lastValid = history.slice().reverse().find(h => h.hand);
    if (firstValid && lastValid) {
      const startY = firstValid.hand.wrist[1];
      const endY = lastValid.hand.wrist[1];
      if (startY < chinY + 0.05 && endY > startY + 0.1) {
        bestMatch = 'THANK_YOU';
        conf = 90;
      }
    }
  }
  // 3. SORRY: Closed hand + front of chest + circular rubbing
  else if (isClosed && avgWristY > chinY && avgWristY < chestY + 0.3) {
    // Circular means movement in both X and Y, and high path ratio
    if (rangeX > 0.08 && rangeY > 0.08 && reversalsX >= 1 && reversalsY >= 1) {
      bestMatch = 'SORRY';
      conf = 92;
    }
  }
  // 4. YES: Closed hand + vertical up/down movement
  else if (isClosed && rangeY > 0.1 && reversalsY >= 1 && pathRatioY > 1.5) {
    // Make sure it's primarily vertical
    if (rangeY > rangeX * 1.5) {
      bestMatch = 'YES';
      conf = 88;
    }
  }
  // 5. NO: Hand + horizontal left/right movement
  // NO is typically index finger up or open hand, but user says "Hand + horizontal left/right"
  else if (rangeX > 0.15 && reversalsX >= 1 && pathRatioX > 1.5) {
    // Make sure it's primarily horizontal, and not HELLO (not near face)
    if (rangeX > rangeY * 1.5 && avgWristY > chinY) {
      bestMatch = 'NO';
      conf = 89;
    }
  }

  if (bestMatch !== 'UNCERTAIN') {
    lastRecognizedTime = now;
    // Clear history to debounce effectively
    history = [];
    return { gesture: bestMatch, confidence: conf / 100.0 };
  }

  return { gesture: 'UNCERTAIN', confidence: 0 };
}
