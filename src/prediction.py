"""
Interactive Prediction Module
Accepts raw arbitrary user text, performs validated preprocessing and padding,
runs parallel inference against trained Simple RNN and LSTM models,
and returns probability, confidence, sentiment label, and tokenization inspection.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from config import (
    MAX_SEQUENCE_LENGTH,
    TOKENIZER_PATH,
    RNN_MODEL_PATH,
    LSTM_MODEL_PATH,
)
from src.preprocessing import clean_text, tokenize_text, pad_sequences_batch
from src.vocabulary import TextTokenizer
from src.rnn_model import load_trained_rnn
from src.lstm_model import load_trained_lstm


class SentimentPredictor:
    """
    Manages loading of models and tokenizer for instant live text inference.
    """

    def __init__(self):
        self.tokenizer: Optional[TextTokenizer] = None
        self.rnn_model = None
        self.lstm_model = None
        self._last_loaded_time = 0.0

    def reload_resources(self):
        """
        Reloads models and tokenizer from disk if they were newly trained.
        """
        if TOKENIZER_PATH.exists():
            try:
                self.tokenizer = TextTokenizer.load(TOKENIZER_PATH)
            except Exception as e:
                print(f"Failed to reload tokenizer: {e}")

        if RNN_MODEL_PATH.exists():
            self.rnn_model = load_trained_rnn()
        else:
            self.rnn_model = None

        if LSTM_MODEL_PATH.exists():
            self.lstm_model = load_trained_lstm()
        else:
            self.lstm_model = None

        self._last_loaded_time = time.time()

    def predict_text(self, text: str, max_length: int = MAX_SEQUENCE_LENGTH) -> Dict[str, Any]:
        """
        Validates user input, preprocesses, tokenizes, and computes real model predictions.
        """
        # Input Validation
        if text is None:
            return {"status": "error", "message": "Input text cannot be null."}

        raw_trimmed = str(text).strip()
        if not raw_trimmed:
            return {"status": "error", "message": "Please enter a non-empty text review."}

        # Check resources
        if self.tokenizer is None or (RNN_MODEL_PATH.exists() and self.rnn_model is None) or (LSTM_MODEL_PATH.exists() and self.lstm_model is None):
            self.reload_resources()

        if self.tokenizer is None:
            return {
                "status": "error",
                "message": "Tokenizer not initialized. Please train the models or build the dataset first."
            }

        if self.rnn_model is None and self.lstm_model is None:
            return {
                "status": "error",
                "message": "Neither RNN nor LSTM model is trained yet. Please train at least one model on the Training page."
            }

        # Preprocessing & Tokenization
        # Dynamically determine sequence length compatible with loaded model
        target_seq_len = max_length
        active_model = self.rnn_model or self.lstm_model
        if active_model is not None and hasattr(active_model, "input_shape") and active_model.input_shape:
            expected_dim = active_model.input_shape[1]
            if expected_dim is not None:
                target_seq_len = expected_dim

        cleaned = clean_text(raw_trimmed)
        tokens = tokenize_text(raw_trimmed)
        seqs = self.tokenizer.texts_to_sequences([cleaned])
        raw_sequence = seqs[0]
        padded_array = pad_sequences_batch(seqs, max_length=target_seq_len)

        # Word-by-word token inspection mapping
        oov_idx = self.tokenizer.word_index[self.tokenizer.oov_token]
        token_details = []
        for tok in tokens:
            idx = self.tokenizer.word_index.get(tok, oov_idx)
            token_details.append({
                "word": tok,
                "index": idx,
                "is_oov": idx == oov_idx
            })

        results = {
            "status": "success",
            "original_text": raw_trimmed,
            "cleaned_text": cleaned,
            "token_count": len(tokens),
            "tokens": token_details[:50],  # sample preview
            "padded_sequence_preview": padded_array[0][:25].tolist(),
            "max_sequence_length": max_length,
            "predictions": {},
            "disclaimer": "Predicted probability denotes model sigmoid activation confidence (0.0 to 1.0) and is not a 100% factual certainty guarantee."
        }

        # RNN Prediction
        if self.rnn_model is not None:
            t0 = time.time()
            rnn_prob = float(self.rnn_model.predict(padded_array, verbose=0)[0][0])
            rnn_lat = round((time.time() - t0) * 1000, 2)
            rnn_label = "Positive" if rnn_prob >= 0.5 else "Negative"
            rnn_conf = round((rnn_prob if rnn_prob >= 0.5 else 1.0 - rnn_prob) * 100, 2)

            results["predictions"]["rnn"] = {
                "model": "Simple RNN",
                "available": True,
                "probability": round(rnn_prob, 4),
                "sentiment": rnn_label,
                "confidence_percent": rnn_conf,
                "latency_ms": rnn_lat
            }
        else:
            results["predictions"]["rnn"] = {
                "model": "Simple RNN",
                "available": False,
                "sentiment": "Not trained",
                "message": "RNN model weights not found."
            }

        # LSTM Prediction
        if self.lstm_model is not None:
            t0 = time.time()
            lstm_prob = float(self.lstm_model.predict(padded_array, verbose=0)[0][0])
            lstm_lat = round((time.time() - t0) * 1000, 2)
            lstm_label = "Positive" if lstm_prob >= 0.5 else "Negative"
            lstm_conf = round((lstm_prob if lstm_prob >= 0.5 else 1.0 - lstm_prob) * 100, 2)

            results["predictions"]["lstm"] = {
                "model": "LSTM",
                "available": True,
                "probability": round(lstm_prob, 4),
                "sentiment": lstm_label,
                "confidence_percent": lstm_conf,
                "latency_ms": lstm_lat
            }
        else:
            results["predictions"]["lstm"] = {
                "model": "LSTM",
                "available": False,
                "sentiment": "Not trained",
                "message": "LSTM model weights not found."
            }

        # Comparison summary if both models are present
        if self.rnn_model is not None and self.lstm_model is not None:
            agree = results["predictions"]["rnn"]["sentiment"] == results["predictions"]["lstm"]["sentiment"]
            diff = round(abs(results["predictions"]["rnn"]["probability"] - results["predictions"]["lstm"]["probability"]), 4)
            results["comparison"] = {
                "models_agree": agree,
                "probability_difference": diff,
                "summary": "Both models agree on the sentiment polarity." if agree else "Disagreement detected: Models reached divergent sentiment classifications."
            }

        return results


# Global singleton predictor instance
sentiment_predictor = SentimentPredictor()
