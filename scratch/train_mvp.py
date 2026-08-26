import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from collections import Counter
import tensorflowjs as tfjs

SEQ_LEN = 40
FEATURE_DIM = 258
MODEL_DIR = 'public/models/isl-mvp-60'

def train_mvp():
    data_dir = 'data/mvp_features'
    x_path = os.path.join(data_dir, 'X.npy')
    y_path = os.path.join(data_dir, 'y.json')
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print("Data not found. Run extract_mvp.py first with a labeled manifest.")
        return
        
    X = np.load(x_path)
    with open(y_path, 'r') as f:
        y_labels = json.load(f)
        
    classes = sorted(list(set(y_labels)))
    label_to_idx = {label: i for i, label in enumerate(classes)}
    y = np.array([label_to_idx[l] for l in y_labels])
    
    print(f"Loaded {len(X)} sequences.")
    print(f"Classes ({len(classes)}): {classes}")
    
    counts = Counter(y_labels)
    for c in classes:
        print(f"  {c}: {counts[c]} samples")
        
    # We may have very few samples for the MVP hackathon (e.g., 1-2 per class if just extracted from single demo).
    # If so, we just train on them to show it can fit, but warn.
    # In a real scenario, we'd augment heavily.
    
    # We will stratify if possible, else just use a simple split or train on all if too small.
    if min(counts.values()) >= 2:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    else:
        print("Warning: Some classes have only 1 sample. Using the entire dataset for training/validation (no strict holdout).")
        X_train, X_val, y_train, y_val = X, X, y, y
        
    # Model Architecture (Lightweight adapter)
    num_classes = len(classes)
    l2 = tf.keras.regularizers.l2(1e-4)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQ_LEN, FEATURE_DIM)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, kernel_regularizer=l2, recurrent_dropout=0.1)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, kernel_regularizer=l2, recurrent_dropout=0.1)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    ]
    
    print("Training model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=8,
        callbacks=callbacks
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_acc:.4f}")
    
    # Export to TFJS
    os.makedirs(MODEL_DIR, exist_ok=True)
    tfjs.converters.save_keras_model(model, MODEL_DIR)
    
    with open(os.path.join(MODEL_DIR, 'mvp-labels.json'), 'w') as f:
        json.dump(classes, f)
        
    print(f"Model exported to {MODEL_DIR}")
    
if __name__ == '__main__':
    train_mvp()
