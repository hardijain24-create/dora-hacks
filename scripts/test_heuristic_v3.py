import os
import cv2
import mediapipe as mp
import math
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

POSE_MODEL_PATH = 'isl_tfjs_export (1)/pose_landmarker_lite.task'
HAND_MODEL_PATH = 'isl_tfjs_export (1)/hand_landmarker.task'
HISTORY_SIZE = 40

history = []

def dist2d(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def extract_hand_state(hand_landmarks):
    if not hand_landmarks: return None
    lm = hand_landmarks
    wrist = (lm[0].x, lm[0].y, lm[0].z)
    index = (lm[8].x, lm[8].y, lm[8].z)
    middle = (lm[12].x, lm[12].y, lm[12].z)
    ring = (lm[16].x, lm[16].y, lm[16].z)
    pinky = (lm[20].x, lm[20].y, lm[20].z)

    avg_dist = (dist2d(wrist, index) + dist2d(wrist, middle) + dist2d(wrist, ring) + dist2d(wrist, pinky)) / 4.0
    is_open = avg_dist > 0.08
    is_closed = avg_dist < 0.06

    # Orientation: angle of middle finger relative to wrist (0 is straight right, -pi/2 is straight up)
    angle = math.atan2(middle[1] - wrist[1], middle[0] - wrist[0])

    return {
        'wrist': wrist,
        'isOpen': is_open,
        'isClosed': is_closed,
        'avgDist': avg_dist,
        'angle': angle
    }

def evaluate_heuristic(pose_landmarks, left_hand, right_hand, frame_idx):
    global history

    hand_state = None
    pose_state = None
    two_hands = False

    if pose_landmarks:
        lm = pose_landmarks
        pose_state = {
            'nose': (lm[0].x, lm[0].y, lm[0].z),
            'leftShoulder': (lm[11].x, lm[11].y, lm[11].z),
            'rightShoulder': (lm[12].x, lm[12].y, lm[12].z),
            'leftWrist': (lm[15].x, lm[15].y, lm[15].z),
            'rightWrist': (lm[16].x, lm[16].y, lm[16].z)
        }

    right_state = extract_hand_state(right_hand)
    left_state = extract_hand_state(left_hand)

    if right_state and left_state:
        two_hands = True
        hand_state = right_state # track dominant for single-hand stats
    elif right_state:
        hand_state = right_state
    elif left_state:
        hand_state = left_state

    history.append({ 'hand': hand_state, 'pose': pose_state, 'two_hands': two_hands, 't': frame_idx })
    if len(history) > HISTORY_SIZE:
        history.pop(0)

    if len(history) < 15:
        return None

    openFrames = 0
    closedFrames = 0
    twoHandFrames = 0
    minX, maxX = 999, -999
    minY, maxY = 999, -999
    totalDx, totalDy, totalDz = 0, 0, 0
    reversalsX, reversalsY = 0, 0
    lastDirX, lastDirY = 0, 0
    validFrames = 0
    avgWristX, avgWristY = 0, 0

    for i in range(len(history)):
        h = history[i]['hand']
        if history[i]['two_hands']: twoHandFrames += 1
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
                dz = h['wrist'][2] - history[i-1]['hand']['wrist'][2]
                totalDx += abs(dx)
                totalDy += abs(dy)
                totalDz += abs(dz)
                
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
    isTwoHands = twoHandFrames > validFrames * 0.6
    
    currentPose = history[-1]['pose']
    if not currentPose: return None

    faceY = currentPose['nose'][1]
    chinY = faceY + 0.1
    chestY = (currentPose['leftShoulder'][1] + currentPose['rightShoulder'][1]) / 2.0

    bestMatch = 'UNCERTAIN'
    firstValid = next((h for h in history if h['hand']), None)
    lastValid = next((h for h in reversed(history) if h['hand']), None)
    
    startY = firstValid['hand']['wrist'][1] if firstValid else 0
    endY = lastValid['hand']['wrist'][1] if lastValid else 0
    startZ = firstValid['hand']['wrist'][2] if firstValid else 0
    endZ = lastValid['hand']['wrist'][2] if lastValid else 0
    startScale = firstValid['hand']['avgDist'] if firstValid else 0
    endScale = lastValid['hand']['avgDist'] if lastValid else 0
    endAngle = lastValid['hand']['angle'] if lastValid else 0

    scores = {'HELLO': 0, 'THANK_YOU': 0, 'SORRY': 0, 'YES': 0, 'NO': 0, 'PLEASE': 0, 'HELP': 0, 'STOP': 0, 'GOOD': 0, 'BAD': 0}
    
    # 1. HELLO
    isNearFace = avgWristY < chinY
    if isOpen and isNearFace and rangeX > 0.15 and reversalsX >= 1 and pathRatioX > 1.5:
        scores['HELLO'] = 95
        bestMatch = 'HELLO'

    # 2. THANK_YOU
    if isOpen and startY < chinY + 0.05 and endY > startY + 0.1:
        scores['THANK_YOU'] = 90
        if bestMatch == 'UNCERTAIN': bestMatch = 'THANK_YOU'

    # 3. SORRY
    if isClosed and avgWristY > chinY and avgWristY < chestY + 0.3 and rangeX > 0.08 and rangeY > 0.08 and reversalsX >= 1 and reversalsY >= 1:
        scores['SORRY'] = 92
        if bestMatch == 'UNCERTAIN': bestMatch = 'SORRY'

    # 4. YES
    if isClosed and rangeY > 0.1 and reversalsY >= 1 and pathRatioY > 1.5 and rangeY > rangeX * 1.5:
        scores['YES'] = 88
        if bestMatch == 'UNCERTAIN': bestMatch = 'YES'

    # 5. NO
    if rangeX > 0.15 and reversalsX >= 1 and pathRatioX > 1.5 and rangeX > rangeY * 1.5 and avgWristY > chinY:
        scores['NO'] = 89
        if bestMatch == 'UNCERTAIN': bestMatch = 'NO'

    # 6. PLEASE
    if isOpen and avgWristY > chinY and avgWristY < chestY + 0.3 and rangeX > 0.08 and rangeY > 0.08 and reversalsX >= 1 and reversalsY >= 1:
        scores['PLEASE'] = 93
        if bestMatch == 'UNCERTAIN' or bestMatch == 'SORRY': bestMatch = 'PLEASE'

    # 7. HELP (Two hands + upward)
    if isTwoHands and rangeY > 0.1 and endY < startY - 0.05:
        scores['HELP'] = 91
        if bestMatch == 'UNCERTAIN': bestMatch = 'HELP'

    # 8. STOP (Open palm facing outward + forward movement)
    # Forward movement implies bounding box size increases (avgDist increases)
    isForward = (endScale - startScale) > 0.02
    if isOpen and isForward and rangeX < 0.1 and rangeY < 0.1:
        scores['STOP'] = 87
        if bestMatch == 'UNCERTAIN': bestMatch = 'STOP'

    # 9. GOOD (Open hand near mouth/chin + downward/away)
    # This overlaps perfectly with THANK_YOU right now.
    if isOpen and startY < chinY + 0.05 and endY > startY + 0.1:
        scores['GOOD'] = 90
        # Wait, how to distinguish THANK_YOU and GOOD?
        # In this simplistic logic, they are almost identical unless we add specific angle checks, which might be flaky.
        
    # 10. BAD (Open hand near mouth/chin + downward/away + different orientation)
    # Palm turning down => angle changes.
    if isOpen and startY < chinY + 0.05 and endY > startY + 0.1:
        scores['BAD'] = 90

    # Dump diagnostic every 10 frames or if a winner exists
    if frame_idx % 10 == 0 or bestMatch != 'UNCERTAIN':
        print(f"\n[ISL CHECK] Frame {frame_idx}")
        print(f"Hand visible: {'YES' if validFrames > 0 else 'NO'} ({validFrames}/{len(history)})")
        print(f"Two hands: {'YES' if isTwoHands else 'NO'} ({twoHandFrames}/{validFrames})")
        print(f"Open hand: {'YES' if isOpen else 'NO'} (Openness avg: {lastValid['hand']['avgDist']:.3f} if hand)")
        print(f"Closed fist: {'YES' if isClosed else 'NO'}")
        print(f"Hand near face: {'YES' if avgWristY < chinY else 'NO'} (wristY={avgWristY:.3f}, chinY={chinY:.3f})")
        print(f"Hand in front of chest: {'YES' if (avgWristY > chinY and avgWristY < chestY + 0.3) else 'NO'} (chestY={chestY:.3f})")
        
        print("\nDiagnostics:")
        print(f"RangeX: {rangeX:.3f}, RangeY: {rangeY:.3f}")
        print(f"TotalDx: {totalDx:.3f}, TotalDy: {totalDy:.3f}, TotalDz: {totalDz:.3f}")
        print(f"PathRatioX: {pathRatioX:.3f}, PathRatioY: {pathRatioY:.3f}")
        print(f"ReversalsX: {reversalsX}, ReversalsY: {reversalsY}")
        print(f"StartScale: {startScale:.3f}, EndScale: {endScale:.3f}, DeltaScale: {(endScale-startScale):.3f}")
        print(f"EndAngle: {endAngle:.3f} rad")
        
        print("\nHeuristic score:")
        for k,v in scores.items():
            print(f"{k} = {v}")
            
        print(f"\nWinner: {bestMatch}")
        print(f"Threshold: 85")
        print(f"Result: {'PASS' if bestMatch != 'UNCERTAIN' else 'FAIL'}")
        
        if scores['THANK_YOU'] > 0 and scores['GOOD'] > 0 and scores['BAD'] > 0:
            print("\nWARNING: THANK_YOU, GOOD, and BAD are NOT SEPARABLE WITH CURRENT FEATURES.")

        if bestMatch != 'UNCERTAIN':
            history.clear()

def main():
    base_options_p = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options_p = mp_vision.PoseLandmarkerOptions(base_options=base_options_p, num_poses=1)
    pose_det = mp_vision.PoseLandmarker.create_from_options(options_p)

    base_options_h = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options_h = mp_vision.HandLandmarkerOptions(base_options=base_options_h, num_hands=2)
    hand_det = mp_vision.HandLandmarker.create_from_options(options_h)

    cap = cv2.VideoCapture('isl_reference_videos/06_gestures_short.mp4')
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_idx += 1
        print(f"Processing frame {frame_idx}...", flush=True)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pose_r = pose_det.detect(mp_image)
        hand_r = hand_det.detect(mp_image)

        p_lm = pose_r.pose_landmarks[0] if pose_r.pose_landmarks else None
        
        left_h = None
        right_h = None
        if hand_r.hand_landmarks:
            for i, handedness in enumerate(hand_r.handedness):
                cat = handedness[0].category_name
                if cat == 'Left': left_h = hand_r.hand_landmarks[i]
                else: right_h = hand_r.hand_landmarks[i]
        
        evaluate_heuristic(p_lm, left_h, right_h, frame_idx)

if __name__ == '__main__':
    main()
