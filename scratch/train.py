!pip install -q mediapipe opencv-python-headless tensorflow tensorflowjs tqdm

import os, json, zipfile, shutil, glob, random, hashlib, csv
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import tensorflow as tf
from tqdm.notebook import tqdm

print('TF version:', tf.__version__)
print('MediaPipe version:', mp.__version__)
print('GPU available:', tf.config.list_physical_devices('GPU'))

WORKDIR = '/content/isl'
RAW_VIDEO_DIR = f'{WORKDIR}/videos'      # local Colab disk only -- ephemeral, gets cleared per category
MODEL_DIR = f'{WORKDIR}/model'
for d in [WORKDIR, RAW_VIDEO_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# === CELL ===

model = tf.keras.models.load_model('/content/isl/model/best_model.keras')

# === CELL ===

from google.colab import drive
drive.mount('/content/drive')

LANDMARK_DIR = '/content/drive/MyDrive/isl_dataset/landmarks'
MANIFEST_PATH = '/content/drive/MyDrive/isl_dataset/manifest.csv'
os.makedirs(LANDMARK_DIR, exist_ok=True)

if not os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['video_key', 'category', 'label', 'npy_filename'])
    print('created new manifest')
else:
    print('found existing manifest -- will resume from it')

# Load already-processed video keys so we never redo finished work
processed_keys = set()
with open(MANIFEST_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        processed_keys.add(row['video_key'])
print(f'{len(processed_keys)} videos already processed previously')

# === CELL ===

!wget -q -O /content/isl/model/pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
!wget -q -O /content/isl/model/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
print('Model bundles downloaded:')
!ls -la /content/isl/model/*.task

# === CELL ===

import requests

resp = requests.get('https://zenodo.org/api/records/4010759')
zip_files = [f for f in resp.json()['files'] if f['key'].endswith('.zip')]
print(f'{len(zip_files)} category zip files found:')
for f in zip_files:
    size_mb = f.get('size', 0) / (1024 * 1024)
    print(f"  - {f['key']}  (~{size_mb:.0f} MB)")

# 'all' processes every category. To do it in smaller batches instead, set e.g.:
# CATEGORIES = ['Greetings.zip', 'People.zip']
CATEGORIES = 'all'

def matches_selection(filename, categories):
    if categories == 'all':
        return True
    return filename in categories

selected_zips = [f for f in zip_files if matches_selection(f['key'], CATEGORIES)]
print(f'\n{len(selected_zips)} categories queued')

# === CELL ===

FRAME_STRIDE = 2
NUM_WORKERS = max(1, os.cpu_count() - 1)
print('Parallel workers:', NUM_WORKERS)

POSE_MODEL_PATH = f'{MODEL_DIR}/pose_landmarker.task'
HAND_MODEL_PATH = f'{MODEL_DIR}/hand_landmarker.task'

_worker_pose = None
_worker_hand = None

def _init_worker():
    global _worker_pose, _worker_hand
    BaseOptions = mp_python.BaseOptions
    VisionRunningMode = mp_vision.RunningMode

    # IMAGE mode (not VIDEO): each frame is detected independently, no timestamp
    # bookkeeping needed. VIDEO mode requires strictly-increasing timestamps on the
    # SAME landmarker instance forever -- since one worker processes many videos
    # sequentially, that breaks the moment a second video starts back at t=0.
    # We don't need VIDEO mode's temporal tracking here anyway; the LSTM is what
    # learns the temporal pattern across frames, not MediaPipe.
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
    )
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
    )
    _worker_pose = mp_vision.PoseLandmarker.create_from_options(pose_options)
    _worker_hand = mp_vision.HandLandmarker.create_from_options(hand_options)

def _landmarks_to_array(landmark_list, n_points, n_dims):
    if not landmark_list:
        return np.zeros(n_points * n_dims)
    vals = []
    for lm in landmark_list:
        if n_dims == 4:
            vals.extend([lm.x, lm.y, lm.z, getattr(lm, 'visibility', 0.0)])
        else:
            vals.extend([lm.x, lm.y, lm.z])
    return np.array(vals)

def _extract_one(args):
    video_path, out_path = args
    global _worker_pose, _worker_hand
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if i % FRAME_STRIDE == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            pose_result = _worker_pose.detect(mp_image)
            hand_result = _worker_hand.detect(mp_image)

            pose = np.zeros(33 * 4)
            if pose_result.pose_landmarks:
                pose = _landmarks_to_array(pose_result.pose_landmarks[0], 33, 4)

            lh = np.zeros(21 * 3)
            rh = np.zeros(21 * 3)
            if hand_result.hand_landmarks:
                for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                    label = handedness[0].category_name
                    arr = _landmarks_to_array(hand_lms, 21, 3)
                    if label == 'Left':
                        lh = arr
                    else:
                        rh = arr

            frames.append(np.concatenate([pose, lh, rh]))
        i += 1
    cap.release()
    if len(frames) > 0:
        np.save(out_path, np.array(frames))
        return out_path
    return None

# === CELL ===

VIDEO_EXTENSIONS = ('.mov', '.mp4', '.avi', '.mkv')

def find_videos(root_dir):
    """Walk a directory tree; return (video_path, label) pairs where label = parent folder name."""
    results = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(VIDEO_EXTENSIONS):
                label = os.path.basename(dirpath)
                results.append((os.path.join(dirpath, fn), label))
    return results

for zf in selected_zips:
    category_name = zf['key'].replace('.zip', '')
    local_zip = os.path.join(RAW_VIDEO_DIR, zf['key'])
    extract_dir = os.path.join(RAW_VIDEO_DIR, category_name)

    print(f'\n=== {category_name} ===')
    print('Downloading...')
    r = requests.get(zf['links']['self'], stream=True)
    with open(local_zip, 'wb') as out:
        for chunk in r.iter_content(chunk_size=1 << 20):
            out.write(chunk)

    print('Extracting zip...')
    with zipfile.ZipFile(local_zip) as z:
        z.extractall(extract_dir)

    videos = find_videos(extract_dir)
    print(f'Found {len(videos)} video files. Distinct labels: {len(set(l for _, l in videos))}')

    tasks = []
    manifest_rows = []
    for video_path, label in videos:
        rel_key = os.path.relpath(video_path, extract_dir)
        video_key = hashlib.md5(f'{category_name}/{rel_key}'.encode()).hexdigest()
        if video_key in processed_keys:
            continue
        npy_filename = f'{video_key}.npy'
        out_path = os.path.join(LANDMARK_DIR, npy_filename)
        tasks.append((video_path, out_path))
        manifest_rows.append((video_key, category_name, label, npy_filename))

    print(f'{len(tasks)} videos need extraction (rest already done previously)')

    if tasks:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=_init_worker) as ex:
            outcomes = list(tqdm(ex.map(_extract_one, tasks), total=len(tasks)))

        with open(MANIFEST_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            for (video_key, cat, label, npy_filename), outcome in zip(manifest_rows, outcomes):
                if outcome is not None:
                    writer.writerow([video_key, cat, label, npy_filename])
                    processed_keys.add(video_key)

    # free local disk before the next category
    os.remove(local_zip)
    shutil.rmtree(extract_dir, ignore_errors=True)
    print(f'{category_name} done, local disk freed')

print('\nAll categories processed.')

# === CELL ===

SEQ_LEN = 40

def resample_sequence(seq, target_len=SEQ_LEN):
    n = len(seq)
    if n == target_len:
        return seq
    idx = np.linspace(0, n - 1, target_len)
    return seq[np.round(idx).astype(int)]

def normalize_sequence(seq):
    seq = seq.copy()
    pose = seq[:, :132].reshape(-1, 33, 4)
    lh = seq[:, 132:195].reshape(-1, 21, 3)
    rh = seq[:, 195:258].reshape(-1, 21, 3)

    left_sh, right_sh = pose[:, 11, :2], pose[:, 12, :2]
    center = (left_sh + right_sh) / 2.0
    scale = np.linalg.norm(left_sh - right_sh, axis=1, keepdims=True)
    scale = np.where(scale < 1e-4, 1.0, scale)

    pose[:, :, :2] -= center[:, None, :]
    pose[:, :, :2] /= scale[:, None, :]
    lh[:, :, :2] -= center[:, None, :]
    lh[:, :, :2] /= scale[:, None, :]
    rh[:, :, :2] -= center[:, None, :]
    rh[:, :, :2] /= scale[:, None, :]

    return np.concatenate([
        pose.reshape(-1, 132), lh.reshape(-1, 63), rh.reshape(-1, 63)
    ], axis=1)

# === CELL ===

X, y_labels = [], []

with open(MANIFEST_PATH) as f:
    reader = csv.DictReader(f)
    manifest_entries = list(reader)

print(f'{len(manifest_entries)} entries in manifest')

for row in tqdm(manifest_entries):
    npy_path = os.path.join(LANDMARK_DIR, row['npy_filename'])
    if not os.path.exists(npy_path):
        continue
    seq = np.load(npy_path)
    if len(seq) < 3:
        continue
    seq = normalize_sequence(seq)
    seq = resample_sequence(seq)
    X.append(seq)
    y_labels.append(row['label'])

# Drop classes that don't have enough samples to actually be learned. A class with only
# 1-2 examples can't generalize -- the model just memorizes that one clip's exact signer,
# lighting, and camera angle, and will misclassify real-world attempts at that sign later.
# Better to exclude it from this training run than ship a class that silently fails.
MIN_SAMPLES_PER_CLASS = 5

from collections import Counter
label_counts = Counter(y_labels)
kept_labels = {label for label, count in label_counts.items() if count >= MIN_SAMPLES_PER_CLASS}
dropped_labels = sorted(set(y_labels) - kept_labels)

if dropped_labels:
    print(f'\nDropping {len(dropped_labels)} class(es) with fewer than {MIN_SAMPLES_PER_CLASS} samples:')
    print(', '.join(f'{l} ({label_counts[l]})' for l in dropped_labels))
    print('(these can be added back in a later training run once more videos for them are extracted)')

filtered = [(x, l) for x, l in zip(X, y_labels) if l in kept_labels]
if filtered:
    X, y_labels = zip(*filtered)
    X = list(X)
    y_labels = list(y_labels)
else:
    X, y_labels = [], []

X = np.array(X, dtype=np.float32)
classes = sorted(set(y_labels))
label_to_idx = {label: i for i, label in enumerate(classes)}
y = np.array([label_to_idx[l] for l in y_labels])

print('\nX shape:', X.shape)
print('num classes (after filtering):', len(classes))

if len(X) == 0:
    print('\nNo samples found -- check that section 6 actually ran and populated the manifest,')
    print('and that MANIFEST_PATH / LANDMARK_DIR point at the same Drive folder used during extraction.')
    print('If classes were dropped above, you may also need MIN_SAMPLES_PER_CLASS samples for at least one class.')
else:
    with open(f'{MODEL_DIR}/labels.json', 'w') as f:
        json.dump(classes, f)

# === CELL ===

from collections import Counter
from sklearn.model_selection import train_test_split

class_counts = Counter(y)
min_count = min(class_counts.values()) if class_counts else 0
print(f'Total samples: {len(y)}  |  Distinct classes: {len(class_counts)}  |  Min samples in any class: {min_count}')

if len(y) < 20 or len(class_counts) < 2:
    print('\nNOTE: this is too little data to train a real classifier -- fine for a pipeline sanity check,')
    print('but you will want to run extraction on a much larger portion of the dataset before training for real.')

# Stratified splitting needs at least 2 samples per class in each split. With a small/early
# subset that is often not true yet, so fall back to a plain (non-stratified) split rather
# than crashing -- this lets you sanity-check the rest of the pipeline on tiny test runs.
can_stratify = min_count >= 2
stratify_arg = y if can_stratify else None
if not can_stratify:
    print('Skipping stratification: at least one class has only 1 sample.')

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=stratify_arg)

temp_counts = Counter(y_temp)
temp_min = min(temp_counts.values()) if temp_counts else 0
stratify_arg2 = y_temp if temp_min >= 2 else None

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=stratify_arg2)

print('train:', X_train.shape, ' val:', X_val.shape, ' test:', X_test.shape)

# === CELL ===

def augment(seq):
    seq = seq.copy()
    if random.random() < 0.6:
        seq += np.random.normal(0, 0.015, seq.shape).astype(np.float32)
    if random.random() < 0.5:
        pose = seq[:, :132].reshape(-1, 33, 4)
        lh = seq[:, 132:195].reshape(-1, 21, 3)
        rh = seq[:, 195:258].reshape(-1, 21, 3)
        pose[:, :, 0] *= -1
        lh[:, :, 0] *= -1
        rh[:, :, 0] *= -1
        seq = np.concatenate([
            pose.reshape(-1, 132), rh.reshape(-1, 63), lh.reshape(-1, 63)
        ], axis=1)
    return seq

def make_augmented_dataset(X, y, multiplier=5):
    X_aug, y_aug = [X], [y]
    for _ in range(multiplier - 1):
        X_aug.append(np.array([augment(s) for s in X]))
        y_aug.append(y)
    return np.concatenate(X_aug), np.concatenate(y_aug)

X_train_aug, y_train_aug = make_augmented_dataset(X_train, y_train, multiplier=5)
print('augmented train shape:', X_train_aug.shape)

# === CELL ===

num_classes = len(classes)

# Stronger regularization than the first pass: smaller LSTM layers (less capacity to
# memorize), L2 weight decay, and higher dropout -- aimed at closing the train/val gap
# seen in the first run (train ~95% / val ~65% is classic overfitting).
l2 = tf.keras.regularizers.l2(1e-4)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQ_LEN, 258)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        64, return_sequences=True, kernel_regularizer=l2, recurrent_dropout=0.1)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        32, kernel_regularizer=l2, recurrent_dropout=0.1)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()

# === CELL ===

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    tf.keras.callbacks.ModelCheckpoint(f'{MODEL_DIR}/best_model.keras', monitor='val_accuracy', save_best_only=True),
]

history = model.fit(
    X_train_aug, y_train_aug,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

# === CELL ===

# Safety net: back up the trained model to Drive right away, so a runtime restart
# (which we have hit a few times now due to library bugs) can never cost you the training run.
#
# We save TWO things:
#  - the full model (for reloading/evaluating in THIS environment)
#  - weights ONLY, in H5 format (for the conversion step below, which runs in a
#    different, older-pinned environment -- weights-only files are more portable
#    across TF versions than a full model file)
DRIVE_DATASET_DIR = '/content/drive/MyDrive/isl_dataset'
os.makedirs(DRIVE_DATASET_DIR, exist_ok=True)

shutil.copy(f'{MODEL_DIR}/best_model.keras', f'{DRIVE_DATASET_DIR}/best_model.keras')
model.save_weights(f'{DRIVE_DATASET_DIR}/model.weights.h5')
shutil.copy(f'{MODEL_DIR}/labels.json', f'{DRIVE_DATASET_DIR}/labels.json')

print('Backed up to Drive:')
print(' -', f'{DRIVE_DATASET_DIR}/best_model.keras')
print(' -', f'{DRIVE_DATASET_DIR}/model.weights.h5')
print(' -', f'{DRIVE_DATASET_DIR}/labels.json')