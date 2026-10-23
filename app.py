"""
Flask Web Application for Deep Learning Mini Project:
Sentiment Analysis Using Simple RNN and LSTM - A Comparative Study of Long-Term Dependencies
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, render_template, request, jsonify

from config import (
    DEFAULT_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    EMBEDDING_DIM,
    RNN_UNITS,
    LSTM_UNITS,
    DENSE_UNITS,
    DROPOUT_RATE,
    VOCAB_SIZE,
    RNN_MODEL_PATH,
    LSTM_MODEL_PATH,
    RNN_HISTORY_PATH,
    LSTM_HISTORY_PATH,
    TOKENIZER_PATH,
)
from src.dataset import IMDBDatasetManager
from src.preprocessing import clean_text, tokenize_text, pad_sequences_batch
from src.vocabulary import TextTokenizer
from src.rnn_model import get_rnn_architecture_details
from src.lstm_model import get_lstm_architecture_details
from src.training import training_tracker, start_background_training
from src.evaluation import evaluate_model, get_cached_evaluation
from src.prediction import sentiment_predictor
from src.long_dependency import run_controlled_dependency_suite, get_cached_dependency_results
from src.comparison import generate_comparative_analysis

app = Flask(__name__)
dataset_manager = IMDBDatasetManager()


# ==========================================
# Frontend HTML Page Routes
# ==========================================

@app.route("/")
@app.route("/dashboard")
def page_dashboard():
    return render_template("index.html", active_page="dashboard")


@app.route("/dataset")
def page_dataset():
    return render_template("dataset.html", active_page="dataset")


@app.route("/preprocessing")
def page_preprocessing():
    return render_template("preprocessing.html", active_page="preprocessing")


@app.route("/rnn")
def page_rnn():
    return render_template("rnn.html", active_page="rnn")


@app.route("/lstm")
def page_lstm():
    return render_template("lstm.html", active_page="lstm")


@app.route("/analyzer")
def page_analyzer():
    return render_template("analyzer.html", active_page="analyzer")


@app.route("/comparison")
def page_comparison():
    return render_template("comparison.html", active_page="comparison")


@app.route("/dependency")
def page_dependency():
    return render_template("dependency.html", active_page="dependency")


@app.route("/evaluation")
def page_evaluation():
    return render_template("evaluation.html", active_page="evaluation")


@app.route("/documentation")
def page_documentation():
    return render_template("documentation.html", active_page="documentation")


# ==========================================
# REST API Endpoints
# ==========================================

@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint."""
    return jsonify({
        "status": "online",
        "service": "SentimentAI - RNN vs LSTM Analyzer",
        "python_version": sys.version.split()[0],
        "framework": "Flask & TensorFlow/Keras"
    }), 200


@app.route("/api/dataset/info", methods=["GET"])
def api_dataset_info():
    """Returns dataset metadata and statistics."""
    try:
        source = request.args.get("source", "keras")
        if source == "csv":
            train_df, val_df, test_df = dataset_manager.load_from_csv()
        meta = dataset_manager.get_metadata()
        return jsonify({"status": "success", "data": meta}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/preprocess", methods=["POST"])
def api_preprocess():
    """
    Preprocesses input text, returning cleaned text, tokens, and integer sequence.
    """
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        max_length = int(data.get("max_length", MAX_SEQUENCE_LENGTH))

        if not text.strip():
            return jsonify({"status": "error", "message": "Text cannot be empty."}), 400

        cleaned = clean_text(text)
        tokens = tokenize_text(text)

        # Attempt to use trained tokenizer if available, else local demo vocabulary
        seq = []
        pad_seq = []
        if TOKENIZER_PATH.exists():
            tok = TextTokenizer.load(TOKENIZER_PATH)
            seqs = tok.texts_to_sequences([cleaned])
            seq = seqs[0]
            pad_seq = pad_sequences_batch([seq], max_length=max_length)[0].tolist()
        else:
            # Build minimal ad-hoc sequence for visual preview before training
            demo_vocab = {"<PAD>": 0, "<OOV>": 1}
            for t in tokens:
                if t not in demo_vocab:
                    demo_vocab[t] = len(demo_vocab)
            seq = [demo_vocab.get(t, 1) for t in tokens]
            pad_seq = pad_sequences_batch([seq], max_length=max_length)[0].tolist()

        return jsonify({
            "status": "success",
            "original_text": text,
            "cleaned_text": cleaned,
            "tokens": tokens,
            "token_count": len(tokens),
            "sequence": seq,
            "padded_sequence": pad_seq,
            "sequence_length": len(pad_seq)
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Generates sentiment prediction using both trained models."""
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        max_length = int(data.get("max_length", MAX_SEQUENCE_LENGTH))

        result = sentiment_predictor.predict_text(text, max_length=max_length)
        status_code = 200 if result.get("status") == "success" else 400
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/models/status", methods=["GET"])
def api_models_status():
    """Returns filesystem existence and training status of both models."""
    rnn_exists = RNN_MODEL_PATH.exists()
    lstm_exists = LSTM_MODEL_PATH.exists()
    tok_exists = TOKENIZER_PATH.exists()

    rnn_meta = get_rnn_architecture_details() if rnn_exists else None
    lstm_meta = get_lstm_architecture_details() if lstm_exists else None

    return jsonify({
        "status": "success",
        "rnn": {
            "trained": rnn_exists,
            "path": str(RNN_MODEL_PATH),
            "architecture": rnn_meta
        },
        "lstm": {
            "trained": lstm_exists,
            "path": str(LSTM_MODEL_PATH),
            "architecture": lstm_meta
        },
        "tokenizer_ready": tok_exists,
        "active_training": training_tracker.get_status()
    }), 200


@app.route("/api/train/status", methods=["GET"])
def api_train_status():
    """Real-time training status for live dashboard progress bar."""
    return jsonify({"status": "success", "data": training_tracker.get_status()}), 200


@app.route("/api/train/rnn", methods=["POST"])
def api_train_rnn():
    """Triggers training of the Simple RNN model in background thread."""
    try:
        data = request.get_json(silent=True) or {}
        params = {
            "model_type": "rnn",
            "epochs": int(data.get("epochs", DEFAULT_EPOCHS)),
            "batch_size": int(data.get("batch_size", BATCH_SIZE)),
            "learning_rate": float(data.get("learning_rate", LEARNING_RATE)),
            "sequence_length": int(data.get("sequence_length", MAX_SEQUENCE_LENGTH)),
            "embedding_dim": int(data.get("embedding_dim", EMBEDDING_DIM)),
            "hidden_units": int(data.get("hidden_units", RNN_UNITS)),
            "dense_units": int(data.get("dense_units", DENSE_UNITS)),
            "dropout_rate": float(data.get("dropout_rate", DROPOUT_RATE)),
            "vocab_size": int(data.get("vocab_size", VOCAB_SIZE)),
            "dataset_source": data.get("dataset_source", "keras")
        }

        started = start_background_training(params)
        if not started:
            return jsonify({
                "status": "conflict",
                "message": "Another training job is currently running. Please wait for it to complete."
            }), 409

        return jsonify({
            "status": "accepted",
            "message": "Simple RNN training initiated in the background.",
            "params": params
        }), 202
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/train/lstm", methods=["POST"])
def api_train_lstm():
    """Triggers training of the LSTM model in background thread."""
    try:
        data = request.get_json(silent=True) or {}
        params = {
            "model_type": "lstm",
            "epochs": int(data.get("epochs", DEFAULT_EPOCHS)),
            "batch_size": int(data.get("batch_size", BATCH_SIZE)),
            "learning_rate": float(data.get("learning_rate", LEARNING_RATE)),
            "sequence_length": int(data.get("sequence_length", MAX_SEQUENCE_LENGTH)),
            "embedding_dim": int(data.get("embedding_dim", EMBEDDING_DIM)),
            "hidden_units": int(data.get("hidden_units", LSTM_UNITS)),
            "dense_units": int(data.get("dense_units", DENSE_UNITS)),
            "dropout_rate": float(data.get("dropout_rate", DROPOUT_RATE)),
            "vocab_size": int(data.get("vocab_size", VOCAB_SIZE)),
            "dataset_source": data.get("dataset_source", "keras")
        }

        started = start_background_training(params)
        if not started:
            return jsonify({
                "status": "conflict",
                "message": "Another training job is currently running. Please wait for it to complete."
            }), 409

        return jsonify({
            "status": "accepted",
            "message": "LSTM training initiated in the background.",
            "params": params
        }), 202
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/evaluate/<model_type>", methods=["GET"])
def api_evaluate(model_type):
    """
    Returns evaluation metrics for specified model ('rnn' or 'lstm').
    Recalculates if requested via ?recompute=true.
    """
    model_type = model_type.lower()
    if model_type not in ("rnn", "lstm"):
        return jsonify({"status": "error", "message": "Invalid model_type. Use 'rnn' or 'lstm'."}), 400

    recompute = request.args.get("recompute", "false").lower() == "true"
    cached = get_cached_evaluation(model_type)

    if cached and not recompute:
        return jsonify({"status": "success", "source": "cached", "data": cached}), 200

    try:
        limit = int(request.args.get("limit", 0)) or None
        results = evaluate_model(model_type, test_limit=limit)
        sentiment_predictor.reload_resources()
        return jsonify({"status": "success", "source": "computed", "data": results}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/compare", methods=["GET"])
def api_compare():
    """Returns side-by-side comparative analysis of RNN vs LSTM."""
    try:
        data = generate_comparative_analysis()
        return jsonify({"status": "success", "data": data}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/training/history", methods=["GET"])
def api_training_history():
    """Returns historical training loss and accuracy metrics."""
    result = {"rnn": None, "lstm": None}
    if RNN_HISTORY_PATH.exists():
        try:
            with open(RNN_HISTORY_PATH, "r", encoding="utf-8") as f:
                result["rnn"] = json.load(f)
        except Exception:
            pass

    if LSTM_HISTORY_PATH.exists():
        try:
            with open(LSTM_HISTORY_PATH, "r", encoding="utf-8") as f:
                result["lstm"] = json.load(f)
        except Exception:
            pass

    return jsonify({"status": "success", "data": result}), 200


@app.route("/api/long-dependency/results", methods=["GET"])
def api_long_dependency():
    """
    Returns controlled academic long-distance dependency experiment results.
    Accepts ?recompute=true to re-run on live model weights.
    """
    recompute = request.args.get("recompute", "false").lower() == "true"
    cached = get_cached_dependency_results()

    if cached and not recompute:
        return jsonify({"status": "success", "source": "cached", "data": cached}), 200

    try:
        data = run_controlled_dependency_suite()
        return jsonify({"status": "success", "source": "computed", "data": data}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
