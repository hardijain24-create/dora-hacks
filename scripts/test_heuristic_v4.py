"""
Full 28 ISL Gesture Precision Evaluation & Disambiguation Suite v8
"""
import math

def clamp(val, mn, mx):
    return max(mn, min(mx, val))

def addScore(val, weight):
    return clamp(val * weight, 0, weight)

def evaluate_heuristic_sim(data):
    isTwoHands = data.get('isTwoHands', False)
    avgOpenRatio = data.get('avgOpenRatio', 0.5)
    validFrames = data.get('validFrames', 20)
    
    poseFrames = data.get('poseFrames', 20)
    headXRange = data.get('headXRange', 0.0)
    headYRange = data.get('headYRange', 0.0)
    headReversalsX = data.get('headReversalsX', 0)
    headReversalsY = data.get('headReversalsY', 0)

    rangeX = data.get('rangeX', 0.0)
    rangeY = data.get('rangeY', 0.0)
    reversalsX = data.get('reversalsX', 0)
    reversalsY = data.get('reversalsY', 0)

    faceDistance = data.get('faceDistance', 1.0)
    chestDistance = data.get('chestDistance', 0.5)

    trajectoryX = data.get('trajectoryX', 0.0)
    trajectoryY = data.get('trajectoryY', 0.0)
    trajectoryZ = data.get('trajectoryZ', 0.0)

    handDistance = data.get('handDistance', 999.0)
    fingertipDistance = data.get('fingertipDistance', 999.0)
    
    isPointing = data.get('isPointing', False)
    isFist = data.get('isFist', False)
    pointingVecZ = data.get('pointingVecZ', 0.0)
    compactness = data.get('compactness', 1.5)
    thumbExt = data.get('thumbExt', 1.0)
    indexExt = data.get('indexExt', 1.0)
    angle = data.get('angle', 0.0)

    scores = {k: 0 for k in [
        'HELLO', 'THANK_YOU', 'SORRY', 'YES', 'NO', 'PLEASE', 'HELP', 'BAD',
        'HOW', 'WHERE', 'WHAT', 'WHY', 'WHEN', 'TODAY', 'TOMORROW', 'YESTERDAY',
        'YOU', 'I_ME', 'HOME', 'SCHOOL', 'HOSPITAL', 'WATER', 'FOOD', 'EAT',
        'DRINK', 'COME', 'GO', 'NEED'
    ]}

    # -------------------------------------------------------------
    # TWO-HANDED GESTURES
    # -------------------------------------------------------------
    if isTwoHands:
        # 1. HOME (roof shape: fingertips close < 0.45, wrists near chest)
        if avgOpenRatio > 0.3 and (fingertipDistance < 0.45 or handDistance < 0.45) and chestDistance < 0.6:
            s = 85 + addScore((0.45 - min(fingertipDistance, handDistance)) * 3, 10)
            if trajectoryY > 0.25 or reversalsX >= 2: s = 0
            scores['HOME'] = s

        # 2. SCHOOL (clapping/tapping palms together in front of chest)
        if avgOpenRatio > 0.3 and handDistance < 0.45 and (reversalsX >= 1 or reversalsY >= 1):
            s = 85 + addScore((reversalsX + reversalsY) * 5, 10)
            scores['SCHOOL'] = s

        # 3. HELP (two hands moving UP together)
        if avgOpenRatio > 0.3 and trajectoryY < -0.05:
            s = 85 + addScore(abs(trajectoryY) * 4, 10)
            scores['HELP'] = s

        # 4. TODAY (two hands moving DOWN together or held low at waist)
        if avgOpenRatio > 0.3 and (trajectoryY > 0.05 or chestDistance > 0.5) and reversalsY == 0:
            s = 85 + addScore(max(0, trajectoryY) * 4, 10)
            scores['TODAY'] = s

        # 5. HOW (two hands facing up turning/moving outward)
        if avgOpenRatio > 0.4 and rangeX > 0.15 and handDistance > 0.3:
            s = 85 + addScore(rangeX * 2, 10)
            if reversalsX >= 1: s = 0 # Waving side to side is HELLO, not HOW!
            scores['HOW'] = s

    # -------------------------------------------------------------
    # SINGLE-HANDED GESTURES
    # -------------------------------------------------------------
    else:
        # 6. HELLO (open hand waving side-to-side near upper face/head/shoulder level)
        if avgOpenRatio > 0.5 and (faceDistance < 0.4 or chestDistance < 0.15) and not isPointing and not isFist and reversalsY == 0 and thumbExt <= 1.4:
            if (rangeX > 0.12 or reversalsX >= 1):
                s = 95 + addScore(rangeX * 2, 5) + addScore(reversalsX * 2.5, 5)
                scores['HELLO'] = s

        # 7. THANK_YOU (open hand starting near chin/mouth moving down to chest, palm upright)
        if avgOpenRatio > 0.5 and faceDistance >= 0.25 and faceDistance < 0.45 and trajectoryY > 0.06 and angle < 0.6 and abs(trajectoryX) < 0.06 and not isPointing and not isFist:
            s = 85 + addScore(trajectoryY * 3, 10)
            if reversalsX >= 1: s = 0
            scores['THANK_YOU'] = s

        # 8. SORRY (fist rubbing chest in circular motion)
        if isFist and chestDistance < 0.7:
            if reversalsX >= 1 or reversalsY >= 1 or rangeX > 0.08:
                s = 90 + addScore(min(rangeX, rangeY) * 2, 5)
                scores['SORRY'] = s

        # 9. PLEASE (open palm rubbing chest in circular motion)
        if avgOpenRatio > 0.6 and chestDistance < 0.7 and not isFist and not isPointing:
            if (reversalsX >= 1 and reversalsY >= 1) or (rangeX > 0.08 and rangeY > 0.08):
                s = 90 + addScore(min(rangeX, rangeY) * 2, 5)
                scores['PLEASE'] = s

        # 10. BAD (open hand near chin moving down with palm flicking down/turned)
        if avgOpenRatio > 0.5 and faceDistance < 0.45 and trajectoryY > 0.06 and angle >= 0.6 and not isFist and not isPointing:
            s = 85 + addScore(trajectoryY * 3, 10)
            if reversalsX >= 1: s = 0
            scores['BAD'] = s

        # 11. WHERE (index finger pointing moving side to side at chest level)
        if isPointing and (rangeX > 0.15 or reversalsX >= 1) and faceDistance >= 0.35 and reversalsY == 0:
            s = 85 + addScore(rangeX * 2, 10)
            if rangeY > rangeX * 1.2: s = 0
            scores['WHERE'] = s

        # 12. WHAT (open palm(s) moving side to side horizontally at chest level)
        if avgOpenRatio > 0.6 and (rangeX > 0.18 or reversalsX >= 1) and faceDistance > 0.35 and reversalsY == 0 and not isPointing and not isFist:
            s = 93 + addScore(rangeX * 2, 5)
            scores['WHAT'] = s

        # 13. WHY (open hand near upper forehead/temple pulling outward ONCE)
        if avgOpenRatio > 0.4 and faceDistance < 0.25 and (trajectoryY > 0.04 or trajectoryX > 0.06) and not isFist and not isPointing:
            s = 90 + addScore(abs(trajectoryX) * 3, 10)
            if reversalsX >= 1: s = 0 # Waving side-to-side (reversals >= 1) is HELLO, not WHY!
            scores['WHY'] = s

        # 14. WHEN (index finger pointing making circular/arc motion)
        if isPointing and reversalsX >= 1 and reversalsY >= 1 and rangeX < 0.25:
            s = 87 + addScore(min(rangeX, rangeY) * 2, 10)
            scores['WHEN'] = s

        # 15. TOMORROW (open hand at cheek/temple moving forward/right)
        if avgOpenRatio > 0.3 and faceDistance >= 0.22 and faceDistance < 0.45 and (trajectoryX > 0.08 or trajectoryZ < -0.03) and not isPointing:
            s = 85 + addScore(abs(trajectoryX) * 3, 10)
            if reversalsX >= 1: s = 0
            scores['TOMORROW'] = s

        # 16. YESTERDAY (open hand at cheek/ear moving backward over shoulder)
        if avgOpenRatio > 0.3 and faceDistance < 0.45 and (trajectoryZ > 0.03 or trajectoryX < -0.08 or trajectoryY < -0.03) and not isPointing and thumbExt <= 1.3:
            s = 85 + addScore(abs(trajectoryZ) * 4 + abs(trajectoryX) * 2, 10)
            if reversalsX >= 1: s = 0
            scores['YESTERDAY'] = s

        # 17. YOU (pointing index finger forward at camera)
        if isPointing and (pointingVecZ < -0.01 or trajectoryZ < -0.03) and reversalsX == 0 and reversalsY == 0:
            s = 85 + addScore(abs(pointingVecZ) * 80, 10)
            scores['YOU'] = s

        # 18. I_ME (pointing index finger inward at chest)
        if isPointing and (pointingVecZ > 0.01 or trajectoryZ > 0.03) and reversalsX == 0 and reversalsY == 0:
            s = 85 + addScore(abs(pointingVecZ) * 80, 10)
            scores['I_ME'] = s

        # 19. HOSPITAL (hand/index forming cross across chest)
        if not isFist and chestDistance < 0.6:
            if reversalsX >= 1 and reversalsY >= 1 and rangeX > 0.20 and rangeY > 0.20:
                s = 80 + addScore((reversalsX + reversalsY) * 5, 15)
                if avgOpenRatio > 0.6 and min(rangeX, rangeY) < 0.20: s = 0
                scores['HOSPITAL'] = s

        # 20. WATER (fingers/thumb near mouth tapping)
        if (thumbExt > 1.3 or indexExt > 1.4) and faceDistance < 0.35 and not isPointing and compactness >= 1.2:
            if reversalsX >= 1 or reversalsY >= 1 or rangeY > 0.05:
                s = 94 + addScore(reversalsX + reversalsY + 1, 5)
                scores['WATER'] = s

        # 21. FOOD (pinched hand near mouth moving inward once)
        if compactness < 1.3 and faceDistance < 0.35 and (trajectoryY < -0.03 or faceDistance < 0.25) and reversalsY == 0:
            s = 85 + addScore(abs(trajectoryY) * 4, 10)
            scores['FOOD'] = s

        # 22. EAT (pinched hand tapping mouth repeatedly)
        if compactness < 1.3 and faceDistance < 0.35 and (reversalsY >= 1 or rangeY > 0.06):
            s = 85 + addScore(reversalsY * 10, 10)
            scores['EAT'] = s

        # 23. DRINK (fist/curved hand near mouth tilting up)
        if (isFist or thumbExt > 1.3) and faceDistance < 0.35 and (trajectoryY < -0.03 or angle > 0.4) and compactness >= 1.2:
            s = 88 + addScore(angle * 15, 10)
            scores['DRINK'] = s

        # 24. COME (open hand pulling inward towards torso)
        if avgOpenRatio > 0.4 and (trajectoryZ > 0.03 or (trajectoryY < -0.04 and faceDistance > 0.4)) and not isPointing and reversalsX == 0:
            s = 75 + addScore(abs(trajectoryZ) * 4, 20)
            scores['COME'] = s

        # 25. GO (open hand or index pushing outward away from torso)
        if avgOpenRatio > 0.4 and (trajectoryZ < -0.03 or (trajectoryY > 0.06 and faceDistance > 0.4)) and reversalsX == 0:
            s = 75 + addScore(abs(trajectoryZ) * 4, 20)
            scores['GO'] = s

        # 26. NEED (index finger hooked moving down at torso)
        if indexExt > 1.0 and indexExt < 2.2 and trajectoryY > 0.08 and reversalsY == 0 and faceDistance > 0.3 and not isPointing:
            s = 75 + addScore(trajectoryY * 4, 20)
            scores['NEED'] = s

        # 27. YES (Head nod OR Fist nod up and down)
        if isFist and reversalsY >= 1 and trajectoryY > 0.05 and chestDistance < 0.6:
            s = 85 + addScore(reversalsY * 10, 10)
            scores['YES'] = s
        elif headYRange > 0.06 and headReversalsY >= 1 and poseFrames > 5 and headYRange > headXRange * 1.5 and validFrames < 15:
            s = 75 + addScore(headYRange * 4, 20)
            scores['YES'] = s

        # 28. NO (Head shake OR Index finger shaking left-right near face/chin)
        if isPointing and reversalsX >= 1 and rangeX > 0.15 and rangeY < rangeX * 0.5 and faceDistance < 0.35:
            s = 85 + addScore(reversalsX * 5, 10)
            scores['NO'] = s
        elif headXRange > 0.06 and headReversalsX >= 1 and poseFrames > 5 and headXRange > headYRange * 1.5 and validFrames < 15:
            s = 75 + addScore(headXRange * 4, 20)
            scores['NO'] = s

    best = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if best[0][1] >= 50:
        return best[0][0]
    return 'UNCERTAIN'

classes = [
    'HELLO', 'THANK_YOU', 'SORRY', 'YES', 'NO', 'PLEASE', 'HELP', 'BAD',
    'HOW', 'WHERE', 'WHAT', 'WHY', 'WHEN', 'TODAY', 'TOMORROW', 'YESTERDAY',
    'YOU', 'I_ME', 'HOME', 'SCHOOL', 'HOSPITAL', 'WATER', 'FOOD', 'EAT',
    'DRINK', 'COME', 'GO', 'NEED'
]

dataset = [
    ('HOME', {'isTwoHands': True, 'avgOpenRatio': 0.8, 'fingertipDistance': 0.1, 'handDistance': 0.3, 'chestDistance': 0.3}),
    ('HOME', {'isTwoHands': True, 'avgOpenRatio': 0.7, 'fingertipDistance': 0.2, 'handDistance': 0.35, 'chestDistance': 0.4}),
    ('SCHOOL', {'isTwoHands': True, 'avgOpenRatio': 0.8, 'handDistance': 0.2, 'reversalsX': 2, 'chestDistance': 0.4}),
    ('SCHOOL', {'isTwoHands': True, 'avgOpenRatio': 0.7, 'handDistance': 0.25, 'reversalsY': 2, 'chestDistance': 0.3}),
    ('HELP', {'isTwoHands': True, 'avgOpenRatio': 0.8, 'trajectoryY': -0.15}),
    ('HELP', {'isTwoHands': True, 'avgOpenRatio': 0.7, 'trajectoryY': -0.12}),
    ('TODAY', {'isTwoHands': True, 'avgOpenRatio': 0.8, 'trajectoryY': 0.15}),
    ('TODAY', {'isTwoHands': True, 'avgOpenRatio': 0.7, 'chestDistance': 0.6}),
    ('HOW', {'isTwoHands': True, 'avgOpenRatio': 0.8, 'rangeX': 0.3, 'handDistance': 0.4}),
    ('HOW', {'isTwoHands': True, 'avgOpenRatio': 0.7, 'rangeX': 0.25, 'handDistance': 0.35}),
    ('HELLO', {'isTwoHands': False, 'avgOpenRatio': 0.8, 'rangeX': 0.3, 'reversalsX': 2, 'faceDistance': 0.2}),
    ('HELLO', {'isTwoHands': False, 'avgOpenRatio': 0.7, 'rangeX': 0.25, 'reversalsX': 3, 'faceDistance': 0.3}),
    ('THANK_YOU', {'isTwoHands': False, 'avgOpenRatio': 0.8, 'faceDistance': 0.3, 'trajectoryY': 0.15, 'angle': 0.1}),
    ('THANK_YOU', {'isTwoHands': False, 'avgOpenRatio': 0.7, 'faceDistance': 0.35, 'chestDistance': 0.4, 'trajectoryY': 0.10, 'angle': 0.2}),
    ('SORRY', {'isTwoHands': False, 'isFist': True, 'avgOpenRatio': 0.1, 'chestDistance': 0.3, 'reversalsX': 2, 'reversalsY': 2, 'rangeX': 0.12, 'rangeY': 0.12}),
    ('SORRY', {'isTwoHands': False, 'isFist': True, 'avgOpenRatio': 0.2, 'chestDistance': 0.4, 'reversalsX': 1, 'reversalsY': 1, 'rangeX': 0.1, 'rangeY': 0.1}),
    ('PLEASE', {'isTwoHands': False, 'avgOpenRatio': 0.8, 'chestDistance': 0.3, 'reversalsX': 2, 'reversalsY': 2, 'rangeX': 0.12, 'rangeY': 0.12}),
    ('PLEASE', {'isTwoHands': False, 'avgOpenRatio': 0.75, 'chestDistance': 0.4, 'reversalsX': 1, 'reversalsY': 1, 'rangeX': 0.1, 'rangeY': 0.1}),
    ('BAD', {'isTwoHands': False, 'avgOpenRatio': 0.8, 'faceDistance': 0.3, 'trajectoryY': 0.15, 'angle': 0.8}),
    ('BAD', {'isTwoHands': False, 'avgOpenRatio': 0.7, 'faceDistance': 0.35, 'trajectoryY': 0.10, 'angle': 0.7}),
    ('WHERE', {'isTwoHands': False, 'isPointing': True, 'rangeX': 0.3, 'reversalsX': 2, 'faceDistance': 0.5}),
    ('WHERE', {'isTwoHands': False, 'isPointing': True, 'rangeX': 0.25, 'reversalsX': 3, 'faceDistance': 0.4}),
    ('WHAT', {'isTwoHands': False, 'avgOpenRatio': 0.8, 'rangeX': 0.3, 'reversalsX': 2}),
    ('WHAT', {'isTwoHands': False, 'avgOpenRatio': 0.75, 'rangeX': 0.25, 'reversalsX': 3}),
    ('WHY', {'isTwoHands': False, 'avgOpenRatio': 0.8, 'faceDistance': 0.2, 'trajectoryY': 0.1}),
    ('WHY', {'isTwoHands': False, 'avgOpenRatio': 0.7, 'faceDistance': 0.18, 'trajectoryX': 0.1}),
    ('WHEN', {'isTwoHands': False, 'isPointing': True, 'rangeX': 0.15, 'rangeY': 0.15, 'reversalsX': 1, 'reversalsY': 1}),
    ('WHEN', {'isTwoHands': False, 'isPointing': True, 'rangeX': 0.18, 'rangeY': 0.12, 'reversalsX': 2, 'reversalsY': 1}),
    ('TOMORROW', {'isTwoHands': False, 'avgOpenRatio': 0.8, 'faceDistance': 0.3, 'trajectoryX': 0.15}),
    ('TOMORROW', {'isTwoHands': False, 'avgOpenRatio': 0.7, 'faceDistance': 0.35, 'trajectoryX': 0.12}),
    ('YESTERDAY', {'isTwoHands': False, 'avgOpenRatio': 0.8, 'faceDistance': 0.3, 'trajectoryZ': 0.1}),
    ('YESTERDAY', {'isTwoHands': False, 'avgOpenRatio': 0.7, 'faceDistance': 0.35, 'trajectoryX': -0.12}),
    ('YOU', {'isTwoHands': False, 'isPointing': True, 'pointingVecZ': -0.05}),
    ('YOU', {'isTwoHands': False, 'isPointing': True, 'pointingVecZ': -0.08}),
    ('I_ME', {'isTwoHands': False, 'isPointing': True, 'pointingVecZ': 0.05}),
    ('I_ME', {'isTwoHands': False, 'isPointing': True, 'pointingVecZ': 0.08}),
    ('HOSPITAL', {'isTwoHands': False, 'avgOpenRatio': 0.5, 'chestDistance': 0.3, 'reversalsX': 2, 'reversalsY': 2, 'rangeX': 0.25, 'rangeY': 0.25}),
    ('HOSPITAL', {'isTwoHands': False, 'isPointing': True, 'chestDistance': 0.4, 'reversalsX': 2, 'reversalsY': 1, 'rangeX': 0.22, 'rangeY': 0.22}),
    ('WATER', {'isTwoHands': False, 'thumbExt': 1.8, 'faceDistance': 0.2, 'reversalsX': 2}),
    ('WATER', {'isTwoHands': False, 'indexExt': 1.6, 'faceDistance': 0.25, 'reversalsY': 2}),
    ('FOOD', {'isTwoHands': False, 'compactness': 0.8, 'faceDistance': 0.2, 'trajectoryY': -0.1}),
    ('FOOD', {'isTwoHands': False, 'compactness': 0.9, 'faceDistance': 0.22, 'trajectoryY': -0.08}),
    ('EAT', {'isTwoHands': False, 'compactness': 0.8, 'faceDistance': 0.2, 'reversalsY': 2}),
    ('EAT', {'isTwoHands': False, 'compactness': 0.9, 'faceDistance': 0.25, 'reversalsY': 3}),
    ('DRINK', {'isTwoHands': False, 'isFist': True, 'faceDistance': 0.2, 'angle': 0.8}),
    ('DRINK', {'isTwoHands': False, 'thumbExt': 1.6, 'faceDistance': 0.25, 'trajectoryY': -0.05}),
    ('COME', {'isTwoHands': False, 'avgOpenRatio': 0.8, 'trajectoryZ': 0.1}),
    ('COME', {'isTwoHands': False, 'avgOpenRatio': 0.7, 'trajectoryZ': 0.08, 'faceDistance': 0.5}),
    ('GO', {'isTwoHands': False, 'avgOpenRatio': 0.8, 'trajectoryZ': -0.1}),
    ('GO', {'isTwoHands': False, 'avgOpenRatio': 0.7, 'trajectoryZ': -0.08, 'faceDistance': 0.5}),
    ('NEED', {'isTwoHands': False, 'indexExt': 1.5, 'trajectoryY': 0.15, 'faceDistance': 0.6}),
    ('NEED', {'isTwoHands': False, 'indexExt': 1.8, 'trajectoryY': 0.12, 'faceDistance': 0.5}),
    ('YES', {'isTwoHands': False, 'isFist': True, 'reversalsY': 2, 'trajectoryY': 0.1, 'chestDistance': 0.4}),
    ('YES', {'isTwoHands': False, 'poseFrames': 10, 'headYRange': 0.12, 'headReversalsY': 2, 'headXRange': 0.02, 'validFrames': 5}),
    ('NO', {'isTwoHands': False, 'isPointing': True, 'reversalsX': 2, 'rangeX': 0.3, 'rangeY': 0.05, 'faceDistance': 0.2}),
    ('NO', {'isTwoHands': False, 'poseFrames': 10, 'headXRange': 0.12, 'headReversalsX': 2, 'headYRange': 0.02, 'validFrames': 5})
]

tp = {c: 0 for c in classes}
fp = {c: 0 for c in classes}
fn = {c: 0 for c in classes}

for true_label, sample in dataset:
    pred = evaluate_heuristic_sim(sample)
    if pred == true_label:
        tp[true_label] += 1
    else:
        fn[true_label] += 1
        if pred in fp:
            fp[pred] += 1
        print(f"MISMATCH: True={true_label:10s} Pred={pred:10s} Sample={sample}")

print(f"{'CLASS':12s} | {'TP':2s} | {'FP':2s} | {'FN':2s} | {'PRECISION':9s} | {'RECALL':9s} | {'F1-SCORE':9s}")
print("-" * 65)

precisions = []
recalls = []
f1s = []

for c in classes:
    p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 1.0
    r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    precisions.append(p)
    recalls.append(r)
    f1s.append(f1)
    print(f"{c:12s} | {tp[c]:2d} | {fp[c]:2d} | {fn[c]:2d} | {p:9.2%} | {r:9.2%} | {f1:9.2%}")

macro_p = sum(precisions) / len(precisions)
macro_r = sum(recalls) / len(recalls)
macro_f1 = sum(f1s) / len(f1s)

print("-" * 65)
print(f"{'MACRO AVG':12s} | -- | -- | -- | {macro_p:9.2%} | {macro_r:9.2%} | {macro_f1:9.2%}")

total_tp = sum(tp.values())
total_fp = sum(fp.values())
total_fn = sum(fn.values())
micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

print(f"{'MICRO AVG':12s} | -- | -- | -- | {micro_p:9.2%} | {micro_r:9.2%} | {micro_f1:9.2%}")
