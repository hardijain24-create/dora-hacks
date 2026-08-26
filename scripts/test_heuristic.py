import cv2
import mediapipe as mp
import numpy as np
import sys
import math

mp_holistic = mp.solutions.holistic
HISTORY_SIZE = 40

history = []

def dist2d(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def extract_hand_state(hand_landmarks):
    if not hand_landmarks:
        return None
    lm = hand_landmarks.landmark
    wrist = (lm[0].x, lm[0].y)
    index = (lm[8].x, lm[8].y)
    middle = (lm[12].x, lm[12].y)
    ring = (lm[16].x, lm[16].y)
    pinky = (lm[20].x, lm[20].y)

    avg_dist = (dist2d(wrist, index) + dist2d(wrist, middle) + dist2d(wrist, ring) + dist2d(wrist, pinky)) / 4.0
    is_open = avg_dist > 0.08
    is_closed = avg_dist < 0.06

    return {
        'wrist': wrist,
        'isOpen': is_open,
        'isClosed': is_closed,
        'avgDist': avg_dist
    }

def evaluate_heuristic(pose, left_hand, right_hand, frame_idx):
    global history

    hand_state = None
    pose_state = None

    if pose:
        lm = pose.landmark
        pose_state = {
            'nose': (lm[0].x, lm[0].y),
            'leftShoulder': (lm[11].x, lm[11].y),
            'rightShoulder': (lm[12].x, lm[12].y)
        }

    right_state = extract_hand_state(right_hand)
    left_state = extract_hand_state(left_hand)

    if right_state:
        hand_state = right_state
    elif left_state:
        hand_state = left_state

    history.append({ 'hand': hand_state, 'pose': pose_state, 't': frame_idx })
    if len(history) > HISTORY_SIZE:
        history.pop(0)

    if len(history) < 15:
        return None

    openFrames = 0
    closedFrames = 0
    minX, maxX = 999, -999
    minY, maxY = 999, -999
    totalDx, totalDy = 0, 0
    reversalsX, reversalsY = 0, 0
    lastDirX, lastDirY = 0, 0
    validFrames = 0
    avgWristX, avgWristY = 0, 0

    for i in range(len(history)):
        h = history[i]['hand']
        if h:
            validFrames += 1
            if h['isOpen']: openFrames += 1
            if h['isClosed']: closedFrames += 1
            
            avgWristX += h['wrist'][0]
            avgWristY += h['wrist'][1]
            
            minX = min(minX, h['wrist'][0])
            maxX = max(maxX, h['wrist'][0])
            minY = min(minY, h['wrist'][1])
            maxY = max(maxY, h['wrist'][1])

            if i > 0 and history[i-1]['hand']:
                dx = h['wrist'][0] - history[i-1]['hand']['wrist'][0]
                dy = h['wrist'][1] - history[i-1]['hand']['wrist'][1]
                totalDx += abs(dx)
                totalDy += abs(dy)
                
                if abs(dx) > 0.01:
                    dirX = 1 if dx > 0 else -1
                    if lastDirX != 0 and dirX != lastDirX: reversalsX += 1
                    lastDirX = dirX
                if abs(dy) > 0.01:
                    dirY = 1 if dy > 0 else -1
                    if lastDirY != 0 and dirY != lastDirY: reversalsY += 1
                    lastDirY = dirY

    if validFrames < 15:
        return None

    avgWristX /= validFrames
    avgWristY /= validFrames

    rangeX = maxX - minX
    rangeY = maxY - minY
    pathRatioX = totalDx / (rangeX + 0.001)
    pathRatioY = totalDy / (rangeY + 0.001)

    isOpen = openFrames > validFrames * 0.6
    isClosed = closedFrames > validFrames * 0.6
    
    currentPose = history[-1]['pose']
    if not currentPose:
        return None

    faceY = currentPose['nose'][1]
    chinY = faceY + 0.1
    chestY = (currentPose['leftShoulder'][1] + currentPose['rightShoulder'][1]) / 2.0

    bestMatch = 'UNCERTAIN'
    
    firstValid = next((h for h in history if h['hand']), None)
    lastValid = next((h for h in reversed(history) if h['hand']), None)
    
    startY = firstValid['hand']['wrist'][1] if firstValid else 0
    endY = lastValid['hand']['wrist'][1] if lastValid else 0

    scores = {}
    
    isNearFace = avgWristY < chinY
    if isOpen and isNearFace and rangeX > 0.15 and reversalsX >= 1 and pathRatioX > 1.5:
        bestMatch = 'HELLO'
        scores['HELLO'] = 95
    else:
        scores['HELLO'] = 0

    if isOpen and rangeY > 0.1 and startY < chinY + 0.05 and endY > startY + 0.1:
        if bestMatch == 'UNCERTAIN': bestMatch = 'THANK_YOU'
        scores['THANK_YOU'] = 90
    else:
        scores['THANK_YOU'] = 0

    if isClosed and avgWristY > chinY and avgWristY < chestY + 0.3 and rangeX > 0.08 and rangeY > 0.08 and reversalsX >= 1 and reversalsY >= 1:
        if bestMatch == 'UNCERTAIN': bestMatch = 'SORRY'
        scores['SORRY'] = 92
    else:
        scores['SORRY'] = 0

    if isClosed and rangeY > 0.1 and reversalsY >= 1 and pathRatioY > 1.5 and rangeY > rangeX * 1.5:
        if bestMatch == 'UNCERTAIN': bestMatch = 'YES'
        scores['YES'] = 88
    else:
        scores['YES'] = 0

    if rangeX > 0.15 and reversalsX >= 1 and pathRatioX > 1.5 and rangeX > rangeY * 1.5 and avgWristY > chinY:
        if bestMatch == 'UNCERTAIN': bestMatch = 'NO'
        scores['NO'] = 89
    else:
        scores['NO'] = 0

    # Dump diagnostic every 10 frames
    if frame_idx % 10 == 0 or bestMatch != 'UNCERTAIN':
        print("\n==================================================")
        print(f"[ISL CHECK] Frame {frame_idx}")
        print(f"Hand visible: {'YES' if validFrames > 0 else 'NO'} ({validFrames}/{len(history)})")
        print(f"Open hand: {'YES' if isOpen else 'NO'} (Openness avg: {lastValid['hand']['avgDist']:.3f} if hand)")
        print(f"Closed fist: {'YES' if isClosed else 'NO'}")
        print(f"Hand near face: {'YES' if avgWristY < chinY else 'NO'} (wristY={avgWristY:.3f}, chinY={chinY:.3f})")
        print(f"Hand in front of chest: {'YES' if (avgWristY > chinY and avgWristY < chestY + 0.3) else 'NO'} (chestY={chestY:.3f})")
        print("Start position:")
        print(f"x={firstValid['hand']['wrist'][0]:.3f}" if firstValid else "x=N/A")
        print(f"y={startY:.3f}")
        print("Current position:")
        print(f"x={lastValid['hand']['wrist'][0]:.3f}" if lastValid else "x=N/A")
        print(f"y={endY:.3f}")
        print(f"RangeX: {rangeX:.3f}, RangeY: {rangeY:.3f}")
        print(f"TotalDx: {totalDx:.3f}, TotalDy: {totalDy:.3f}")
        print(f"PathRatioX: {pathRatioX:.3f}, PathRatioY: {pathRatioY:.3f}")
        print(f"ReversalsX: {reversalsX}, ReversalsY: {reversalsY}")
        print("Heuristic score:")
        for k,v in scores.items():
            print(f"{k} = {v}")
        print(f"Winner: {bestMatch}")
        print("==================================================")
        
        if bestMatch != 'UNCERTAIN':
            # clear history just like TS does
            history.clear()

def main():
    video_path = 'isl_reference_videos/06_gestures_short.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video.")
        return

    frame_idx = 0
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            
            evaluate_heuristic(results.pose_landmarks, results.left_hand_landmarks, results.right_hand_landmarks, frame_idx)

if __name__ == "__main__":
    main()
