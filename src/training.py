"""
Model Training Management Module
Handles asynchronous background training of Simple RNN and LSTM models,
real-time epoch metrics tracking, early stopping, checkpointing, and history persistence.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from config import (
    VOCAB_SIZE,
    MAX_SEQUENCE_LENGTH,
    EMBEDDING_DIM,
    RNN_UNITS,
    LSTM_UNITS,
    DENSE_UNITS,
    DROPOUT_RATE,
    LEARNING_RATE,
    BATCH_SIZE,
    DEFAULT_EPOCHS,
    RANDOM_SEED,
    RNN_MODEL_PATH,
    LSTM_MODEL_PATH,
    RNN_HISTORY_PATH,
    LSTM_HISTORY_PATH,
    TOKENIZER_PATH,
)
from src.dataset import IMDBDatasetManager
from src.preprocessing import clean_text, pad_sequences_batch
from src.vocabulary import TextTokenizer
from src.rnn_model import build_rnn_model
from src.lstm_model import build_lstm_model


class ProgressTrackingCallback(Callback):
    """
    Custom Keras Callback to broadcast training progress in real-time
    for asynchronous web dashboard updates.
    """

    def __init__(self, model_type: str, total_epochs: int, tracker: "TrainingTracker"):
        super().__init__()
        self.model_type = model_type
        self.total_epochs = total_epochs
        self.tracker = tracker
        self.epoch_start_time = 0.0

    def on_train_begin(self, logs=None):
        self.tracker.update(
            status="training",
            message=f"Starting training for {self.model_type.upper()}...",
            current_epoch=0,
            total_epochs=self.total_epochs,
            progress_pct=0.0
        )

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.tracker.update(
            current_epoch=epoch + 1,
            message=f"Epoch {epoch + 1}/{self.total_epochs} in progress..."
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        duration = round(time.time() - self.epoch_start_time, 2)
        progress = round(((epoch + 1) / self.total_epochs) * 100, 1)

        loss = float(logs.get("loss", 0.0))
        acc = float(logs.get("accuracy", 0.0))
        val_loss = float(logs.get("val_loss", 0.0))
        val_acc = float(logs.get("val_accuracy", 0.0))
        val_prec = float(logs.get("val_precision", 0.0))
        val_rec = float(logs.get("val_recall", 0.0))

        epoch_record = {
            "epoch": epoch + 1,
            "loss": round(loss, 4),
            "accuracy": round(acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_precision": round(val_prec, 4),
            "val_recall": round(val_rec, 4),
            "duration_sec": duration
        }

        self.tracker.append_epoch_history(epoch_record)
        self.tracker.update(
            current_epoch=epoch + 1,
            progress_pct=progress,
            latest_loss=round(loss, 4),
            latest_acc=round(acc, 4),
            latest_val_loss=round(val_loss, 4),
            latest_val_acc=round(val_acc, 4),
            message=f"Completed epoch {epoch + 1}/{self.total_epochs} - Val Acc: {round(val_acc * 100, 2)}%"
        )


class TrainingTracker:
    """
    Thread-safe progress tracker for training sessions.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.state: Dict[str, Any] = {
            "status": "idle",  # idle, preparing, training, completed, error
            "model_type": None,
            "current_epoch": 0,
            "total_epochs": 0,
            "progress_pct": 0.0,
            "latest_loss": None,
            "latest_acc": None,
            "latest_val_loss": None,
            "latest_val_acc": None,
            "message": "Ready to train.",
            "elapsed_seconds": 0,
            "start_time": None,
            "epochs_history": [],
            "error_message": None,
        }

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                self.state[k] = v

    def append_epoch_history(self, record: dict):
        with self._lock:
            self.state["epochs_history"].append(record)

    def get_status(self) -> dict:
        with self._lock:
            data = dict(self.state)
            if data["status"] == "training" and data["start_time"]:
                data["elapsed_seconds"] = round(time.time() - data["start_time"], 1)
            return data


# Global singleton tracker and execution lock
training_tracker = TrainingTracker()
training_lock = threading.Lock()


def run_training_pipeline(
    model_type: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    sequence_length: int = MAX_SEQUENCE_LENGTH,
    embedding_dim: int = EMBEDDING_DIM,
    hidden_units: int = RNN_UNITS,
    dense_units: int = DENSE_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    vocab_size: int = VOCAB_SIZE,
    dataset_source: str = "keras"
) -> Dict[str, Any]:
    """
    Executes the end-to-end model training workflow synchronously.
    Intended to be called in a background thread.
    """
    model_type = model_type.lower()
    if model_type not in ("rnn", "lstm"):
        raise ValueError("model_type must be either 'rnn' or 'lstm'")

    start_timestamp = time.time()
    training_tracker.update(
        status="preparing",
        model_type=model_type,
        current_epoch=0,
        total_epochs=epochs,
        progress_pct=0.0,
        start_time=start_timestamp,
        epochs_history=[],
        message="Loading dataset and preparing text preprocessing..."
    )

    try:
        # Step 1: Load Dataset
        dataset_manager = IMDBDatasetManager(random_seed=RANDOM_SEED)
        if dataset_source.lower() == "csv":
            train_df, val_df, _ = dataset_manager.load_from_csv()
        else:
            train_df, val_df, _ = dataset_manager.load_from_keras()

        # Step 2: Vocabulary creation / Tokenizer
        # Check if saved tokenizer exists; if not or different, fit on training data
        tokenizer: Optional[TextTokenizer] = None
        if TOKENIZER_PATH.exists():
            try:
                tokenizer = TextTokenizer.load(TOKENIZER_PATH)
                if tokenizer.num_words != vocab_size:
                    tokenizer = None
            except Exception:
                tokenizer = None

        if tokenizer is None:
            training_tracker.update(message="Fitting vocabulary on training data...")
            tokenizer = TextTokenizer(num_words=vocab_size)
            tokenizer.fit_on_texts(train_df["text"].tolist())
            tokenizer.save()

        # Step 3: Text to Numerical Sequences and Padding
        training_tracker.update(message="Tokenizing and padding training and validation sequences...")
        train_seqs = tokenizer.texts_to_sequences(train_df["text"].tolist())
        val_seqs = tokenizer.texts_to_sequences(val_df["text"].tolist())

        x_train = pad_sequences_batch(train_seqs, max_length=sequence_length)
        y_train = train_df["label"].values.astype("float32")

        x_val = pad_sequences_batch(val_seqs, max_length=sequence_length)
        y_val = val_df["label"].values.astype("float32")

        # Step 4: Model Building
        training_tracker.update(message=f"Compiling {model_type.upper()} model architecture...")
        if model_type == "rnn":
            model = build_rnn_model(
                vocab_size=vocab_size,
                sequence_length=sequence_length,
                embedding_dim=embedding_dim,
                rnn_units=hidden_units,
                dense_units=dense_units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )
            save_path = RNN_MODEL_PATH
            history_path = RNN_HISTORY_PATH
        else:
            model = build_lstm_model(
                vocab_size=vocab_size,
                sequence_length=sequence_length,
                embedding_dim=embedding_dim,
                lstm_units=hidden_units,
                dense_units=dense_units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )
            save_path = LSTM_MODEL_PATH
            history_path = LSTM_HISTORY_PATH

        save_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 5: Callbacks setup
        progress_cb = ProgressTrackingCallback(model_type, epochs, training_tracker)
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
        checkpoint = ModelCheckpoint(
            filepath=str(save_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )

        callbacks = [progress_cb, early_stop, checkpoint]

        # Step 6: Model Training
        training_tracker.update(status="training", message="Model training initiated...")
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Ensure model is saved
        model.save(save_path)
        total_time = round(time.time() - start_timestamp, 2)

        # Format and save history
        formatted_history = {
            "model_type": model_type,
            "training_time_seconds": total_time,
            "epochs_run": len(history.history["loss"]),
            "epochs_configured": epochs,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "embedding_dim": embedding_dim,
            "hidden_units": hidden_units,
            "dense_units": dense_units,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "loss": [round(float(v), 5) for v in history.history["loss"]],
            "val_loss": [round(float(v), 5) for v in history.history["val_loss"]],
            "accuracy": [round(float(v), 5) for v in history.history["accuracy"]],
            "val_accuracy": [round(float(v), 5) for v in history.history["val_accuracy"]],
            "precision": [round(float(v), 5) for v in history.history.get("precision", [])],
            "val_precision": [round(float(v), 5) for v in history.history.get("val_precision", [])],
            "recall": [round(float(v), 5) for v in history.history.get("recall", [])],
            "val_recall": [round(float(v), 5) for v in history.history.get("val_recall", [])],
            "final_train_accuracy": round(float(history.history["accuracy"][-1]), 4),
            "final_val_accuracy": round(float(history.history["val_accuracy"][-1]), 4),
            "final_train_loss": round(float(history.history["loss"][-1]), 4),
            "final_val_loss": round(float(history.history["val_loss"][-1]), 4),
        }

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(formatted_history, f, indent=2)

        training_tracker.update(
            status="completed",
            progress_pct=100.0,
            message=f"{model_type.upper()} training completed in {total_time}s! Model saved.",
            elapsed_seconds=total_time
        )

        return formatted_history

    except Exception as e:
        error_msg = f"Training error in {model_type.upper()}: {str(e)}"
        print(error_msg)
        training_tracker.update(
            status="error",
            error_message=str(e),
            message=error_msg
        )
        raise e


def start_background_training(config_params: dict) -> bool:
    """
    Safely initiates background training if no other training job is currently running.
    """
    if training_tracker.get_status()["status"] in ("preparing", "training"):
        return False

    def worker():
        with training_lock:
            try:
                run_training_pipeline(**config_params)
            except Exception as e:
                print(f"Background training failed: {e}")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return True
