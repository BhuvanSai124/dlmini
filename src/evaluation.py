"""
Model Evaluation Module
Computes authentic evaluation metrics on the held-out IMDB test set (25,000 samples):
Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report, ROC-AUC,
Inference Latency, and saves persistent JSON results.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)

from config import (
    MAX_SEQUENCE_LENGTH,
    RNN_MODEL_PATH,
    LSTM_MODEL_PATH,
    RNN_EVAL_PATH,
    LSTM_EVAL_PATH,
    RNN_HISTORY_PATH,
    LSTM_HISTORY_PATH,
    TOKENIZER_PATH,
)
from src.dataset import IMDBDatasetManager
from src.preprocessing import pad_sequences_batch
from src.vocabulary import TextTokenizer
from src.rnn_model import load_trained_rnn
from src.lstm_model import load_trained_lstm


def evaluate_model(
    model_type: str,
    test_limit: Optional[int] = None,
    dataset_source: str = "keras"
) -> Dict[str, Any]:
    """
    Evaluates either 'rnn' or 'lstm' against the IMDB test set.
    """
    model_type = model_type.lower()
    if model_type not in ("rnn", "lstm"):
        raise ValueError("model_type must be either 'rnn' or 'lstm'")

    # Load model
    if model_type == "rnn":
        model = load_trained_rnn()
        eval_path = RNN_EVAL_PATH
        history_path = RNN_HISTORY_PATH
    else:
        model = load_trained_lstm()
        eval_path = LSTM_EVAL_PATH
        history_path = LSTM_HISTORY_PATH

    if model is None:
        raise FileNotFoundError(f"Trained {model_type.upper()} model file not found. Please train the model first.")

    # Load tokenizer
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError("Tokenizer file not found. Please train a model or fit the tokenizer first.")
    tokenizer = TextTokenizer.load(TOKENIZER_PATH)

    # Determine sequence length expected by the model
    sequence_length = model.input_shape[1] if model.input_shape and model.input_shape[1] else MAX_SEQUENCE_LENGTH

    # Load test set
    dm = IMDBDatasetManager()
    if dataset_source == "csv":
        _, _, test_df = dm.load_from_csv()
    else:
        _, _, test_df = dm.load_from_keras()

    if test_limit and test_limit < len(test_df):
        test_df = test_df.iloc[:test_limit].reset_index(drop=True)

    test_seqs = tokenizer.texts_to_sequences(test_df["text"].tolist())
    x_test = pad_sequences_batch(test_seqs, max_length=sequence_length)
    y_true = test_df["label"].values.astype(int)

    # Measure inference time
    start_time = time.time()
    y_probs = model.predict(x_test, batch_size=128, verbose=0).flatten()
    total_inference_time = round(time.time() - start_time, 3)
    avg_inference_latency_ms = round((total_inference_time / len(y_true)) * 1000, 3)

    # Binary decisions at threshold 0.5
    y_pred = (y_probs >= 0.5).astype(int)

    # Compute exact metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_probs))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    # Classification report
    clf_report = classification_report(y_true, y_pred, target_names=["Negative", "Positive"], output_dict=True)

    # ROC curve - downsample points to prevent large JSON payload
    fpr_arr, tpr_arr, thresh_arr = roc_curve(y_true, y_probs)
    sample_indices = np.linspace(0, len(fpr_arr) - 1, min(50, len(fpr_arr)), dtype=int)
    roc_points = [
        {"fpr": round(float(fpr_arr[i]), 4), "tpr": round(float(tpr_arr[i]), 4)}
        for i in sample_indices
    ]

    # Training time from history if available
    training_time = 0.0
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                hist = json.load(f)
                training_time = hist.get("training_time_seconds", 0.0)
        except Exception:
            pass

    evaluation_results = {
        "model_type": model_type,
        "test_samples": len(y_true),
        "sequence_length": sequence_length,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]]
        },
        "classification_report": clf_report,
        "roc_curve": roc_points,
        "training_time_seconds": training_time,
        "inference_time_seconds": total_inference_time,
        "avg_latency_ms": avg_inference_latency_ms,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)

    return evaluation_results


def get_cached_evaluation(model_type: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves cached evaluation metrics from disk.
    """
    model_type = model_type.lower()
    eval_path = RNN_EVAL_PATH if model_type == "rnn" else LSTM_EVAL_PATH
    if eval_path.exists():
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None
