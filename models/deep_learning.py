"""
NFM — Deep Learning Model (CNN)
==================================
1-D Convolutional Neural Network that classifies raw epoch
tensors (channels × time) into *Functional Response* vs.
*Non-Response*.

Architecture
------------
Input : (n_channels, n_times, 1)   — treated as a 1-channel "image"
Conv1D × 2 → BatchNorm → ReLU → MaxPool
Flatten → Dense(64) → Dropout → Sigmoid(1)

Training
--------
Binary cross-entropy loss, Adam optimiser, early stopping on
validation loss with patience = 7.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CNN_EPOCHS, CNN_BATCH_SIZE, CNN_LEARNING_RATE,
    TEST_SIZE, CV_FOLDS, RANDOM_STATE, MODEL_DIR,
)

logger = logging.getLogger(__name__)


def _build_cnn(n_channels: int, n_times: int):
    """
    Construct a compact 1-D CNN in Keras.
    Input shape: (n_channels, n_times)  → reshaped inside the model
    to (n_times, n_channels) so Conv1D operates along time.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(n_channels, n_times), name="eeg_input")

    # Transpose to (n_times, n_channels) for Conv1D
    x = layers.Permute((2, 1))(inp)

    # Block 1
    x = layers.Conv1D(32, kernel_size=7, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Block 2
    x = layers.Conv1D(64, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Classifier head
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid", name="response_prob")(x)

    model = keras.Model(inputs=inp, outputs=out, name="NFM_CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CNN_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _prepare_data(
    epochs_data: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data and return (X_train, X_test, y_train, y_test)."""
    from sklearn.model_selection import train_test_split
    return train_test_split(
        epochs_data, labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )


def train_cnn(
    epochs_data: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    """
    Train the CNN classifier on raw epoch arrays.

    Parameters
    ----------
    epochs_data : ndarray (n_epochs, n_channels, n_times)
    labels : ndarray (n_epochs,) binary

    Returns
    -------
    dict with model, history, metrics, y_test, y_prob, fpr, tpr, roc_auc
    """
    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        roc_auc_score, roc_curve,
    )

    X_train, X_test, y_train, y_test = _prepare_data(epochs_data, labels)
    n_channels, n_times = X_train.shape[1], X_train.shape[2]

    model = _build_cnn(n_channels, n_times)
    model.summary(print_fn=logger.info)

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=7, restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0,
    )

    # Evaluate
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }
    logger.info("CNN hold-out metrics: %s", metrics)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return {
        "model": model,
        "history": history.history,
        "metrics": metrics,
        "y_test": y_test,
        "y_prob": y_prob,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": metrics["roc_auc"],
    }


def predict_proba_cnn(model, epochs_data: np.ndarray) -> np.ndarray:
    """Return P(Functional Response) from the CNN."""
    return model.predict(epochs_data, verbose=0).ravel()


def cross_validate_cnn(
    epochs_data: np.ndarray,
    labels: np.ndarray,
    n_folds: int = CV_FOLDS,
) -> Dict:
    """
    Stratified K-fold cross-validation for the CNN.
    Returns per-fold accuracy and ROC-AUC.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, roc_auc_score

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    accs, aucs = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(epochs_data, labels), 1):
        X_tr, X_val = epochs_data[train_idx], epochs_data[val_idx]
        y_tr, y_val = labels[train_idx], labels[val_idx]

        n_ch, n_t = X_tr.shape[1], X_tr.shape[2]
        model = _build_cnn(n_ch, n_t)

        import tensorflow as tf
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True)

        model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                  epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE,
                  callbacks=[es], verbose=0)

        y_prob = model.predict(X_val, verbose=0).ravel()
        accs.append(accuracy_score(y_val, (y_prob >= 0.5).astype(int)))
        aucs.append(roc_auc_score(y_val, y_prob))
        logger.info("  Fold %d — acc %.3f, auc %.3f", fold, accs[-1], aucs[-1])

    return {
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_roc_auc_mean":  float(np.mean(aucs)),
        "cv_accuracy_std":  float(np.std(accs)),
        "cv_roc_auc_std":   float(np.std(aucs)),
    }


def save_cnn(model, tag: str = "cnn") -> Path:
    path = MODEL_DIR / f"{tag}_model.keras"
    model.save(str(path))
    logger.info("Saved CNN → %s", path)
    return path
