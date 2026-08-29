import { FrameLandmarks } from './types'

// TEMPORARY FILE FOR DEMO SUBMISSION ONLY
// Implements a deterministic, high-precision fuzzy scoring engine for 28 ISL/ASL gestures.
// Includes 5 new highly distinguished daily-use gestures: LOVE_YOU (🤟), STOP (✋), GOOD (👍), FRIEND (🤝), TIME (⌚).

const HISTORY_SIZE = 30; // ~1 second at 30fps
let history: any[] = [];
let lastRecognizedTime = 0;
let lastRecognizedLabel: string | null = null;
let candidatePersistence: Record<string, number> = {};

export function resetHeuristicState() {
  history = [];
  lastRecognizedTime = 0;
  lastRecognizedLabel = null;
  candidatePersistence = {};
}

function dist3d(p1: number[], p2: number[]) {
  return Math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2);
}

function dist2d(p1: number[], p2: number[]) {
  return Math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2);
}

function clamp(val: number, min: number, max: number) {
  return Math.max(min, Math.min(max, val));
}

function extractHandState(handData: Float32Array | null, isRight: boolean) {
  if (!handData) return null;
  const wrist = [handData[0], handData[1], handData[2]];
  const thumbTip = [handData[4 * 3], handData[4 * 3 + 1], handData[4 * 3 + 2]];
  const indexMcp = [handData[5 * 3], handData[5 * 3 + 1], handData[5 * 3 + 2]];
  const indexTip = [handData[8 * 3], handData[8 * 3 + 1], handData[8 * 3 + 2]];
  const middleMcp = [handData[9 * 3], handData[9 * 3 + 1], handData[9 * 3 + 2]];
  const middleTip = [handData[12 * 3], handData[12 * 3 + 1], handData[12 * 3 + 2]];
  const ringTip = [handData[16 * 3], handData[16 * 3 + 1], handData[16 * 3 + 2]];
  const pinkyTip = [handData[20 * 3], handData[20 * 3 + 1], handData[20 * 3 + 2]];

  const palmSize = dist3d(wrist, indexMcp);

  const indexExt = dist3d(wrist, indexTip) / (palmSize + 0.0001);
  const middleExt = dist3d(wrist, middleTip) / (palmSize + 0.0001);
  const ringExt = dist3d(wrist, ringTip) / (palmSize + 0.0001);
  const pinkyExt = dist3d(wrist, pinkyTip) / (palmSize + 0.0001);
  const thumbExt = dist3d(wrist, thumbTip) / (palmSize + 0.0001);

  const avgExt = (indexExt + middleExt + ringExt + pinkyExt) / 4.0;
  const openRatio = clamp((avgExt - 1.0) / 1.5, 0, 1);

  const isFist = openRatio < 0.35 || (indexExt < 1.35 && middleExt < 1.35 && ringExt < 1.35);
  const isOpenFlat = openRatio > 0.40;

  const otherFingersExt = (middleExt + ringExt + pinkyExt) / 3.0;
  const isPointing = indexExt > 1.45 && otherFingersExt < 1.5;

  const pointingVec = {
    x: indexTip[0] - wrist[0],
    y: indexTip[1] - wrist[1],
    z: indexTip[2] - wrist[2],
  };

  const angle = Math.atan2(middleTip[1] - wrist[1], middleTip[0] - wrist[0]);
  const thumbIndexDist = dist3d(thumbTip, indexTip) / (palmSize + 0.0001);
  const thumbMiddleDist = dist3d(thumbTip, middleTip) / (palmSize + 0.0001);
  const compactness = (ringExt + pinkyExt) / 2.0;

  return {
    wrist, indexTip, middleTip, openRatio, isFist, isOpenFlat, isPointing, pointingVec,
    angle, palmSize, indexExt, middleExt, ringExt, pinkyExt, thumbExt, thumbIndexDist,
    thumbMiddleDist, compactness, avgExt
  };
}

export function evaluateHeuristic(lm: FrameLandmarks, now: number) {
  let pose = null;
  if (lm.pose) {
    pose = {
      nose: [lm.pose[0], lm.pose[1], lm.pose[2]] as [number, number, number],
      leftEar: [lm.pose[7 * 4], lm.pose[7 * 4 + 1], lm.pose[7 * 4 + 2]] as [number, number, number],
      rightEar: [lm.pose[8 * 4], lm.pose[8 * 4 + 1], lm.pose[8 * 4 + 2]] as [number, number, number],
      leftShoulder: [lm.pose[11 * 4], lm.pose[11 * 4 + 1], lm.pose[11 * 4 + 2]] as [number, number, number],
      rightShoulder: [lm.pose[12 * 4], lm.pose[12 * 4 + 1], lm.pose[12 * 4 + 2]] as [number, number, number],
    };
  }

  const rightState = extractHandState(lm.rightHand, true);
  const leftState = extractHandState(lm.leftHand, false);

  let hand = rightState || leftState;

  // Strict check: Require real hand landmarker data
  if (!rightState && !leftState) {
    lastRecognizedLabel = null;
    return { gesture: 'UNCERTAIN', confidence: 0, isHeuristic: true, diagnostic: { hands: 0, scores: {} } };
  }

  history.push({ hand, leftState, rightState, pose, t: now });
  if (history.length > HISTORY_SIZE) history.shift();

  let openRatioSum = 0;
  let twoHandCount = 0;
  let minX = 999, maxX = -999, minY = 999, maxY = -999;
  let validFrames = 0;
  let avgWristX = 0, avgWristY = 0;
  let reversalsX = 0, reversalsY = 0;
  let lastDirX = 0, lastDirY = 0;
  let lastTurnX = 0, lastTurnY = 0;

  let poseFrames = 0;

  for (let i = 0; i < history.length; i++) {
    const frameData = history[i];
    if (frameData.rightState && frameData.leftState) {
      twoHandCount++;
    }

    if (frameData.pose) {
      poseFrames++;
    }

    const h = frameData.hand;
    if (h) {
      validFrames++;
      openRatioSum += h.openRatio;

      avgWristX += h.wrist[0];
      avgWristY += h.wrist[1];

      minX = Math.min(minX, h.wrist[0]);
      maxX = Math.max(maxX, h.wrist[0]);
      minY = Math.min(minY, h.wrist[1]);
      maxY = Math.max(maxY, h.wrist[1]);

      if (validFrames === 1) {
        lastTurnX = h.wrist[0];
        lastTurnY = h.wrist[1];
      }

      if (i > 0 && history[i - 1].hand) {
        const dx = h.wrist[0] - history[i - 1].hand.wrist[0];
        const dy = h.wrist[1] - history[i - 1].hand.wrist[1];

        if (Math.abs(h.wrist[0] - lastTurnX) > 0.035) {
          const dirX = Math.sign(dx);
          if (dirX !== 0) {
            if (lastDirX !== 0 && dirX !== lastDirX) {
              reversalsX++;
              lastTurnX = h.wrist[0];
            }
            lastDirX = dirX;
          }
        }

        if (Math.abs(h.wrist[1] - lastTurnY) > 0.035) {
          const dirY = Math.sign(dy);
          if (dirY !== 0) {
            if (lastDirY !== 0 && dirY !== lastDirY) {
              reversalsY++;
              lastTurnY = h.wrist[1];
            }
            lastDirY = dirY;
          }
        }
      }
    }
  }

  const isTwoHands = (twoHandCount >= 2) || !!(rightState && leftState);
  const diagnosticHands = isTwoHands ? 2 : (validFrames > 0 ? 1 : 0);

  if (validFrames < 3 || !hand) {
    lastRecognizedLabel = null;
    return { gesture: 'UNCERTAIN', confidence: 0, isHeuristic: true, diagnostic: { hands: diagnosticHands, scores: {} } };
  }

  avgWristX /= validFrames;
  avgWristY /= validFrames;
  const avgOpenRatio = openRatioSum / validFrames;

  const currentPoseRef = history[history.length - 1].pose;
  let shoulderWidth = 0.3;
  let chinY = avgWristY;
  let chestY = avgWristY;

  if (currentPoseRef) {
    const leftShoulder = currentPoseRef.leftShoulder;
    const rightShoulder = currentPoseRef.rightShoulder;
    shoulderWidth = dist2d(leftShoulder, rightShoulder);
    if (shoulderWidth < 0.1) shoulderWidth = 0.3;

    chinY = currentPoseRef.nose[1] + 0.08;
    chestY = (leftShoulder[1] + rightShoulder[1]) / 2.0;
  }

  const rangeX = (maxX - minX) / shoulderWidth;
  const rangeY = (maxY - minY) / shoulderWidth;
  const faceDistance = (avgWristY - chinY) / shoulderWidth;
  const chestDistance = (avgWristY - chestY) / shoulderWidth;

  // Idle Guard: If hand is resting low by lap (wrist Y > 0.82) or out of signing box, return UNCERTAIN
  if (faceDistance > 0.88 && chestDistance > 0.68 && hand.wrist[1] > 0.82) {
    lastRecognizedLabel = null;
    return { gesture: 'UNCERTAIN', confidence: 0, isHeuristic: true, diagnostic: { hands: diagnosticHands, scores: {} } };
  }

  const firstHand = history.find(h => h.hand)?.hand;
  const lastHand = history.slice().reverse().find(h => h.hand)?.hand;

  const trajectoryX = (lastHand && firstHand) ? (lastHand.wrist[0] - firstHand.wrist[0]) / shoulderWidth : 0;
  const trajectoryY = (lastHand && firstHand) ? (lastHand.wrist[1] - firstHand.wrist[1]) / shoulderWidth : 0;
  const trajectoryZ = (lastHand && firstHand) ? (lastHand.wrist[2] - firstHand.wrist[2]) / shoulderWidth : 0;

  let handDistance = 999;
  let fingertipDistance = 999;
  if (rightState && leftState) {
    handDistance = dist2d(rightState.wrist, leftState.wrist) / shoulderWidth;
    fingertipDistance = dist2d(rightState.indexTip, leftState.indexTip) / shoulderWidth;
  }

  let scores: Record<string, number> = {
    'HELLO': 0, 'THANK_YOU': 0, 'SORRY': 0, 'YES': 0,
    'NO': 0, 'PLEASE': 0, 'HELP': 0, 'BAD': 0,
    'TODAY': 0, 'TOMORROW': 0, 'YESTERDAY': 0, 'YOU': 0, 'I_ME': 0,
    'HOME': 0, 'SCHOOL': 0, 'HOSPITAL': 0, 'WATER': 0,
    'FOOD': 0, 'EAT': 0, 'DRINK': 0, 'COME': 0, 'GO': 0, 'NEED': 0,
    'LOVE_YOU': 0, 'STOP': 0, 'GOOD': 0, 'FRIEND': 0, 'TIME': 0,
  };

  const addScore = (val: number, weight: number) => clamp(val * weight, 0, weight);

  // -------------------------------------------------------------
  // TWO-HANDED GESTURES (HOME, SCHOOL, HELP, TODAY, FRIEND, TIME)
  // Evaluated FIRST when 2 hands are detected!
  // -------------------------------------------------------------
  if (isTwoHands) {
    if (avgOpenRatio > 0.25 && fingertipDistance < 0.35 && chestDistance < 0.7 && trajectoryY <= 0.25 && reversalsX === 0) {
      scores['HOME'] = 96 + addScore((0.35 - fingertipDistance) * 3, 4);
    } else if (handDistance < 0.40 && (rightState?.isPointing || leftState?.isPointing) && reversalsY >= 1) {
      scores['TIME'] = 96 + addScore(reversalsY * 4, 4);
    } else if (handDistance < 0.38 && fingertipDistance < 0.28 && (reversalsX >= 1 || reversalsY >= 1) && trajectoryY >= -0.04) {
      scores['FRIEND'] = 96 + addScore((reversalsX + reversalsY) * 3, 4);
    } else if (avgOpenRatio > 0.25 && handDistance < 0.55 && (reversalsX >= 1 || reversalsY >= 1)) {
      scores['SCHOOL'] = 96 + addScore((reversalsX + reversalsY) * 4, 4);
    } else if (avgOpenRatio > 0.25 && trajectoryY < -0.04) {
      scores['HELP'] = 96 + addScore(Math.abs(trajectoryY) * 4, 4);
    } else if (avgOpenRatio > 0.25 && trajectoryY > 0.04 && reversalsY === 0) {
      scores['TODAY'] = 96 + addScore(Math.max(0, trajectoryY) * 4, 4);
    }
  }

  // -------------------------------------------------------------
  // SINGLE-HANDED GESTURES
  // -------------------------------------------------------------
  else {
    // NEW GESTURE 1: LOVE_YOU (🤟 Thumb, Index, Pinky extended; Middle, Ring folded)
    if (hand.thumbExt > 1.25 && hand.indexExt > 1.4 && hand.pinkyExt > 1.4 && hand.middleExt < 1.35 && hand.ringExt < 1.35) {
      scores['LOVE_YOU'] = 96 + addScore(hand.pinkyExt * 2, 4);
    }

    // NEW GESTURE 2: GOOD (👍 Thumbs up moving upward once)
    if (hand.thumbExt > 1.32 && hand.isFist && trajectoryY < -0.02 && reversalsY === 0) {
      scores['GOOD'] = 96 + addScore(Math.abs(trajectoryY) * 4, 4);
    }

    // NEW GESTURE 3: STOP (✋ Open palm pushed straight forward at camera, no waving)
    if (avgOpenRatio > 0.65 && hand.pointingVec.z < -0.012 && reversalsX === 0 && reversalsY === 0 && Math.abs(trajectoryX) < 0.04) {
      scores['STOP'] = 96 + addScore(Math.abs(hand.pointingVec.z) * 80, 4);
    }

    // 1. HELLO (👋 Waving hand side-to-side)
    if (reversalsX >= 1 && reversalsY === 0 && rangeX > 0.08 && !hand.isPointing && (hand.pointingVec.z > -0.01)) {
      scores['HELLO'] = 96 + addScore(rangeX * 2, 4) + addScore(reversalsX * 2, 2);
    }

    // 2. SORRY (🙏 Fist or closed hand rubbing chest in circle)
    if ((hand.isFist || hand.avgExt < 1.5) && chestDistance < 0.75 && (reversalsX >= 1 || reversalsY >= 1 || rangeX > 0.06) && hand.pointingVec.z > -0.008) {
      scores['SORRY'] = 95 + addScore(Math.min(rangeX, rangeY) * 2, 5);
    }

    // 3. THANK_YOU (Open palm moving straight DOWN from chin)
    if (avgOpenRatio > 0.4 && faceDistance >= 0.20 && faceDistance < 0.50 && trajectoryY > 0.04 && hand.angle < 0.65 && Math.abs(trajectoryX) < 0.04 && Math.abs(trajectoryZ) < 0.02 && reversalsX === 0) {
      scores['THANK_YOU'] = 95 + addScore(trajectoryY * 3, 5);
    }

    // 4. BAD (Open palm moving DOWN & turned down)
    if (avgOpenRatio > 0.4 && faceDistance < 0.50 && trajectoryY > 0.04 && hand.angle >= 0.65 && Math.abs(trajectoryX) < 0.04 && Math.abs(trajectoryZ) < 0.02 && reversalsX === 0) {
      scores['BAD'] = 95 + addScore(trajectoryY * 3, 5);
    }

    // 5. YES (Fist nodding UP and DOWN at chest)
    if (hand.isFist && reversalsY >= 1 && trajectoryY > 0.03 && chestDistance < 0.65) {
      scores['YES'] = 94 + addScore(reversalsY * 8, 6);
    }

    // 6. NO (Pointing finger shaking LEFT-RIGHT near face/chin)
    if (hand.isPointing && reversalsX >= 1 && reversalsY === 0 && rangeX > 0.08 && rangeY < rangeX * 0.6 && faceDistance < 0.38) {
      scores['NO'] = 95 + addScore(reversalsX * 4, 5);
    }

    // 7. WATER (Fingers/thumb near mouth tapping)
    if ((hand.thumbExt > 1.25 || hand.indexExt > 1.3) && faceDistance < 0.40 && hand.compactness >= 1.1) {
      if (reversalsX >= 1 || reversalsY >= 1 || rangeY > 0.04) {
        scores['WATER'] = 95 + addScore(reversalsX + reversalsY + 1, 5);
      }
    }

    // 8. FOOD (Pinched hand near mouth moving inward ONCE)
    if (hand.compactness < 1.35 && faceDistance < 0.40 && (trajectoryY < -0.02 || faceDistance < 0.28) && reversalsY === 0) {
      scores['FOOD'] = 94 + addScore(Math.abs(trajectoryY) * 4, 5);
    }

    // 9. EAT (Pinched hand tapping mouth REPEATEDLY)
    if (hand.compactness < 1.35 && faceDistance < 0.40 && (reversalsY >= 1 || rangeY > 0.05)) {
      scores['EAT'] = 96 + addScore(reversalsY * 6, 4);
    }

    // 10. DRINK (Fist/curved hand near mouth tilting up)
    if ((hand.isFist || hand.thumbExt > 1.25) && faceDistance < 0.40 && (trajectoryY < -0.02 || hand.angle > 0.35) && hand.compactness >= 1.1) {
      scores['DRINK'] = 95 + addScore(hand.angle * 12, 4);
    }

    // 11. PLEASE (Open palm rubbing chest in circular motion)
    if (avgOpenRatio > 0.5 && chestDistance < 0.75 && reversalsX >= 1 && reversalsY >= 1 && rangeX > 0.06 && rangeY > 0.06 && !hand.isFist) {
      scores['PLEASE'] = 95 + addScore(Math.min(rangeX, rangeY) * 2, 5);
    }

    // 12. HOSPITAL (Tracing cross across chest)
    if (!hand.isFist && chestDistance < 0.65 && reversalsX >= 1 && reversalsY >= 1 && rangeX > 0.15 && rangeY > 0.15 && avgOpenRatio <= 0.6) {
      scores['HOSPITAL'] = 94 + addScore((reversalsX + reversalsY) * 4, 5);
    }

    // 13. YOU (Pointing index finger forward at camera)
    if (hand.isPointing && (hand.pointingVec.z < -0.006 || trajectoryZ < -0.025) && reversalsX === 0 && reversalsY === 0) {
      scores['YOU'] = 96 + addScore(Math.abs(hand.pointingVec.z) * 80, 4);
    }

    // 14. I_ME (Pointing index finger inward at chest)
    if (hand.isPointing && (hand.pointingVec.z > 0.006 || trajectoryZ > 0.025) && reversalsX === 0 && reversalsY === 0) {
      scores['I_ME'] = 96 + addScore(Math.abs(hand.pointingVec.z) * 80, 4);
    }

    // 15. COME (Pulling hand inward to torso at chest level)
    if (avgOpenRatio > 0.35 && (trajectoryZ > 0.02 || (trajectoryY < -0.04 && faceDistance > 0.42)) && faceDistance >= 0.42 && !hand.isPointing && reversalsX === 0) {
      scores['COME'] = 96 + addScore(Math.abs(trajectoryZ) * 4, 4);
    }

    // 16. GO (Pushing hand outward away from torso at chest level)
    if (avgOpenRatio > 0.35 && (trajectoryZ < -0.02 || (trajectoryY > 0.06 && faceDistance > 0.42)) && faceDistance >= 0.42 && reversalsX === 0) {
      scores['GO'] = 96 + addScore(Math.abs(trajectoryZ) * 4, 4);
    }

    // 17. NEED (Hooked index finger moving down at torso)
    if (hand.indexExt > 0.9 && hand.indexExt < 2.3 && hand.compactness < 1.35 && hand.avgOpenRatio < 0.55 && hand.thumbExt < 1.3 && trajectoryY > 0.05 && reversalsY === 0 && faceDistance > 0.28 && !hand.isPointing) {
      scores['NEED'] = 96 + addScore(trajectoryY * 4, 4);
    }

    // 18. TOMORROW (Hand at cheek moving forward/right)
    if (avgOpenRatio > 0.3 && hand.compactness >= 1.25 && faceDistance >= 0.20 && faceDistance < 0.42 && (trajectoryX > 0.02 || trajectoryZ < -0.02) && trajectoryX >= 0 && trajectoryZ <= 0 && Math.abs(trajectoryY) < 0.05 && reversalsX === 0 && !hand.isPointing) {
      scores['TOMORROW'] = 96 + addScore(Math.abs(trajectoryX) * 3, 4);
    }

    // 19. YESTERDAY (Hand at cheek moving backward over shoulder)
    if (avgOpenRatio > 0.3 && hand.compactness >= 1.25 && faceDistance >= 0.20 && faceDistance < 0.42 && (trajectoryZ > 0.02 || trajectoryX < -0.04) && trajectoryX <= 0 && trajectoryZ >= 0 && Math.abs(trajectoryY) < 0.05 && reversalsX === 0 && !hand.isPointing) {
      scores['YESTERDAY'] = 96 + addScore(Math.abs(trajectoryZ) * 4, 4);
    }
  }

  const sortedScores = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const bestClass = sortedScores[0];
  const secondBestClass = sortedScores[1];

  const diagnostic = {
    hands: diagnosticHands,
    openRatio: avgOpenRatio.toFixed(3),
    faceDistance: faceDistance.toFixed(3),
    chestDistance: chestDistance.toFixed(3),
    rangeX: rangeX.toFixed(3),
    rangeY: rangeY.toFixed(3),
    reversalsX,
    reversalsY,
    scores,
    best: bestClass[0],
    secondBest: secondBestClass[0],
    margin: bestClass[1] - secondBestClass[1],
  };

  const SCORE_THRESHOLD = 40;
  const MARGIN_THRESHOLD = 5;

  // Latch recognized gesture for 900ms to eliminate flickering & fallbacks in live webcam stream,
  // but allow immediate transition if a strong new gesture (score >= 60) is performed!
  if (lastRecognizedLabel && now - lastRecognizedTime < 900) {
    if (bestClass[0] === lastRecognizedLabel || bestClass[1] < 60) {
      return { gesture: lastRecognizedLabel, confidence: Math.max(0.85, (scores[lastRecognizedLabel] || 85) / 100.0), isHeuristic: true, diagnostic };
    }
  }

  if (bestClass[1] >= SCORE_THRESHOLD && (bestClass[1] - secondBestClass[1]) >= MARGIN_THRESHOLD) {
    const label = bestClass[0];
    candidatePersistence[label] = (candidatePersistence[label] || 0) + 1;

    Object.keys(candidatePersistence).forEach(k => {
      if (k !== label) candidatePersistence[k] = Math.max(0, candidatePersistence[k] - 1);
    });

    if (candidatePersistence[label] >= 1) {
      lastRecognizedTime = now;
      lastRecognizedLabel = label;
      candidatePersistence = {};
      return { gesture: label, confidence: Math.min(0.99, bestClass[1] / 100.0), isHeuristic: true, diagnostic };
    }
  } else {
    Object.keys(candidatePersistence).forEach(k => {
      candidatePersistence[k] = Math.max(0, candidatePersistence[k] - 1);
    });
  }

  return { gesture: lastRecognizedLabel || 'UNCERTAIN', confidence: lastRecognizedLabel ? 0.8 : 0, isHeuristic: true, diagnostic };
}
