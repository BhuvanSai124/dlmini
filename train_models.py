"""
Pre-training and Evaluation Bootstrap Script
Trains Simple RNN and LSTM models on IMDB dataset, evaluates authentic metrics,
exports raw CSV, and generates baseline results for instant faculty demonstration.
"""

import json
import os
import sys
import time
from pathlib import Path

# Set project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from config import (
    VOCAB_SIZE,
    MAX_SEQUENCE_LENGTH,
    EMBEDDING_DIM,
    RNN_UNITS,
    LSTM_UNITS,
    DENSE_UNITS,
    DROPOUT_RATE,
    LEARNING_RATE,
    IMDB_CSV_PATH,
    RNN_MODEL_PATH,
    LSTM_MODEL_PATH,
    RNN_HISTORY_PATH,
    LSTM_HISTORY_PATH,
    TOKENIZER_PATH,
)
from src.dataset import IMDBDatasetManager
from src.preprocessing import pad_sequences_batch
from src.vocabulary import TextTokenizer
from src.rnn_model import build_rnn_model
from src.lstm_model import build_lstm_model
from src.evaluation import evaluate_model
from src.comparison import generate_comparative_analysis
from src.long_dependency import run_controlled_dependency_suite


def main():
    print("=" * 70, flush=True)
    print("SentimentAI: Initial Model Training & Evaluation Pipeline", flush=True)
    print("=" * 70, flush=True)

    # 1. Dataset Loading
    dm = IMDBDatasetManager()
    print("\n[Step 1/6] Loading IMDB Dataset...", flush=True)
    
    train_size = 5000
    val_size = 1500
    test_eval_size = 2500

    train_df, val_df, test_df = dm.load_from_keras(train_size=train_size, val_size=val_size)
    print(f"Loaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples.", flush=True)

    # Export a clean sample CSV for Option 2 (Local CSV)
    if not IMDB_CSV_PATH.exists():
        print(f"Exporting sample IMDB dataset to {IMDB_CSV_PATH}...", flush=True)
        csv_df = train_df[["text", "sentiment"]].rename(columns={"text": "review"})
        csv_df.to_csv(IMDB_CSV_PATH, index=False)
        print("CSV export complete.", flush=True)

    # 2. Tokenizer & Vocabulary
    print("\n[Step 2/6] Building Training Vocabulary (Vocab size = 10,000)...", flush=True)
    tokenizer = TextTokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(train_df["text"].tolist())
    tokenizer.save()
    print(f"Tokenizer saved to {TOKENIZER_PATH}.", flush=True)

    print("Encoding and padding text sequences (Max Length = 150)...", flush=True)
    seq_len = 150
    x_train = pad_sequences_batch(tokenizer.texts_to_sequences(train_df["text"].tolist()), max_length=seq_len)
    y_train = train_df["label"].values.astype("float32")

    x_val = pad_sequences_batch(tokenizer.texts_to_sequences(val_df["text"].tolist()), max_length=seq_len)
    y_val = val_df["label"].values.astype("float32")

    epochs = 3
    batch_size = 128

    # 3. Train Simple RNN
    print("\n[Step 3/6] Training Simple RNN Model (3 Epochs, Batch Size = 128)...", flush=True)
    rnn_model = build_rnn_model(
        vocab_size=VOCAB_SIZE,
        sequence_length=seq_len,
        embedding_dim=EMBEDDING_DIM,
        rnn_units=RNN_UNITS,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE
    )

    t0_rnn = time.time()
    history_rnn = rnn_model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    rnn_time = round(time.time() - t0_rnn, 2)
    rnn_model.save(RNN_MODEL_PATH)

    rnn_hist_data = {
        "model_type": "rnn",
        "training_time_seconds": rnn_time,
        "epochs_run": len(history_rnn.history["loss"]),
        "epochs_configured": epochs,
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "loss": [round(float(v), 5) for v in history_rnn.history["loss"]],
        "val_loss": [round(float(v), 5) for v in history_rnn.history["val_loss"]],
        "accuracy": [round(float(v), 5) for v in history_rnn.history["accuracy"]],
        "val_accuracy": [round(float(v), 5) for v in history_rnn.history["val_accuracy"]],
    }
    with open(RNN_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(rnn_hist_data, f, indent=2)
    print(f"Simple RNN Training Complete in {rnn_time}s! Val Acc: {rnn_hist_data['val_accuracy'][-1]*100:.2f}%", flush=True)

    # 4. Train LSTM
    print("\n[Step 4/6] Training LSTM Model (3 Epochs, Batch Size = 128)...", flush=True)
    lstm_model = build_lstm_model(
        vocab_size=VOCAB_SIZE,
        sequence_length=seq_len,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE
    )

    t0_lstm = time.time()
    history_lstm = lstm_model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    lstm_time = round(time.time() - t0_lstm, 2)
    lstm_model.save(LSTM_MODEL_PATH)

    lstm_hist_data = {
        "model_type": "lstm",
        "training_time_seconds": lstm_time,
        "epochs_run": len(history_lstm.history["loss"]),
        "epochs_configured": epochs,
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "loss": [round(float(v), 5) for v in history_lstm.history["loss"]],
        "val_loss": [round(float(v), 5) for v in history_lstm.history["val_loss"]],
        "accuracy": [round(float(v), 5) for v in history_lstm.history["accuracy"]],
        "val_accuracy": [round(float(v), 5) for v in history_lstm.history["val_accuracy"]],
    }
    with open(LSTM_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(lstm_hist_data, f, indent=2)
    print(f"LSTM Training Complete in {lstm_time}s! Val Acc: {lstm_hist_data['val_accuracy'][-1]*100:.2f}%", flush=True)

    # 5. Evaluate on Unseen Test Partition
    print("\n[Step 5/6] Evaluating Models on Held-out Test Set...", flush=True)
    rnn_eval = evaluate_model("rnn", test_limit=test_eval_size)
    print(f"RNN Test Results: Accuracy={rnn_eval['accuracy']*100:.2f}%, F1={rnn_eval['f1_score']:.4f}, AUC={rnn_eval['roc_auc']:.4f}", flush=True)

    lstm_eval = evaluate_model("lstm", test_limit=test_eval_size)
    print(f"LSTM Test Results: Accuracy={lstm_eval['accuracy']*100:.2f}%, F1={lstm_eval['f1_score']:.4f}, AUC={lstm_eval['roc_auc']:.4f}", flush=True)

    # 6. Generate Comparative and Long-Term Dependency Results
    print("\n[Step 6/6] Generating Comparative Analysis and Long-Term Dependency Suites...", flush=True)
    generate_comparative_analysis()
    dep_results = run_controlled_dependency_suite(max_length=seq_len)
    print("Controlled long-term dependency experiments completed successfully.", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("All Models, Tokenizer, Datasets, and Metrics Successfully Generated!", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
