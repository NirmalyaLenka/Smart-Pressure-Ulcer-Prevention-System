"""
Smart Pressure Ulcer Prevention System
Temporal CNN Training Script

Trains a 1D convolutional neural network on multi-modal sensor time-series
to predict ischemia risk per body zone.

Input features per timestep (sampled at 10 Hz, 50-sample window = 5 minutes):
    - pressure_kpa      : capacitive sensor reading (0 - 40 kPa)
    - duration_mins     : cumulative contact time (0 - 240 minutes)
    - skin_temp_delta_c : deviation from baseline skin temperature (-5 to +5 C)
    - spo2_percent      : reflectance PPG tissue oxygenation (60 - 100 %)

Output:
    - risk_score        : float in [0, 1], risk of ischemia within 20-40 minutes
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt

WINDOW_LEN  = 50
FEATURES    = 4
BATCH_SIZE  = 64
SEED        = 42

FEATURE_COLS = [
    "pressure_kpa",
    "duration_mins",
    "skin_temp_delta_c",
    "spo2_percent",
]
LABEL_COL = "risk_label"


# ---------------------------------------------------------------------------
# Data loading and windowing
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load sensor CSV. Expected columns:
      patient_id, zone_id, timestamp_ms, pressure_kpa, duration_mins,
      skin_temp_delta_c, spo2_percent, risk_label
    """
    df = pd.read_csv(csv_path)
    required = ["patient_id", "zone_id", "timestamp_ms"] + FEATURE_COLS + [LABEL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df


def create_windows(df: pd.DataFrame):
    """
    Create sliding windows of length WINDOW_LEN per (patient, zone).
    Label = risk_label at the END of the window (predicting future state).
    """
    X_list, y_list = [], []

    for (patient, zone), group in df.groupby(["patient_id", "zone_id"]):
        group = group.sort_values("timestamp_ms")
        features = group[FEATURE_COLS].values.astype(np.float32)
        labels   = group[LABEL_COL].values.astype(np.float32)

        for i in range(len(features) - WINDOW_LEN):
            X_list.append(features[i : i + WINDOW_LEN])
            y_list.append(labels[i + WINDOW_LEN - 1])

    X = np.stack(X_list)   # (N, WINDOW_LEN, FEATURES)
    y = np.array(y_list)   # (N,)
    return X, y


def normalize_features(X_train, X_val, X_test):
    """Per-feature z-score normalization using training statistics."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, mean, std


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_model(window_len: int = WINDOW_LEN, n_features: int = FEATURES) -> keras.Model:
    """
    Temporal CNN:
        Input -> Conv1D(32, 5) -> ReLU -> Conv1D(64, 3) -> ReLU
              -> GlobalAvgPool -> Dense(1, sigmoid)

    Kept deliberately shallow for FPGA quantization compatibility.
    """
    inputs = keras.Input(shape=(window_len, n_features), name="sensor_window")

    x = keras.layers.Conv1D(
        filters=32, kernel_size=5, padding="causal",
        activation="relu", name="conv1"
    )(inputs)

    x = keras.layers.Conv1D(
        filters=64, kernel_size=3, padding="causal",
        activation="relu", name="conv2"
    )(x)

    x = keras.layers.GlobalAveragePooling1D(name="gap")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="risk_score")(x)

    model = keras.Model(inputs, outputs, name="ischemia_predictor")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print(f"Loading dataset from {args.dataset}")
    df = load_dataset(args.dataset)
    print(f"  Rows: {len(df)}, Positive labels: {df[LABEL_COL].sum():.0f} "
          f"({100*df[LABEL_COL].mean():.1f}%)")

    X, y = create_windows(df)
    print(f"  Windows: {len(X)}, shape: {X.shape}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

    X_train, X_val, X_test, norm_mean, norm_std = normalize_features(
        X_train, X_val, X_test)

    model = build_model()
    model.summary()

    # Class weights to handle imbalanced labels (high-risk events are rare)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    class_weight = {0: 1.0, 1: float(pos_weight)}
    print(f"  Class weight for positive class: {pos_weight:.2f}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5, patience=4),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.output, "best_model.h5"),
            monitor="val_auc", mode="max", save_best_only=True),
    ]

    os.makedirs(args.output, exist_ok=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    y_pred_prob = model.predict(X_test).flatten()
    y_pred      = (y_pred_prob >= 0.6).astype(int)

    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"\nTest AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=["safe", "at_risk"]))

    # Save normalization parameters
    np.save(os.path.join(args.output, "norm_mean.npy"), norm_mean)
    np.save(os.path.join(args.output, "norm_std.npy"),  norm_std)

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"],     label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.history["auc"],     label="train AUC")
    axes[1].plot(history.history["val_auc"], label="val AUC")
    axes[1].set_title("AUC")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "training_curves.png"))
    print(f"Training curves saved to {args.output}/training_curves.png")

    print(f"Model saved to {args.output}/best_model.h5")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ischemia risk predictor")
    parser.add_argument("--dataset", type=str,
                        default="dataset/sample_data.csv",
                        help="Path to sensor CSV dataset")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Directory to save model and artifacts")
    args = parser.parse_args()
    train(args)
