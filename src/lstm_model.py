"""
Long Short-Term Memory (LSTM) Model Implementation
Academic Architecture:
Input -> Embedding(vocab_size, embedding_dim) -> LSTM(units) -> Dropout(rate) -> Dense(32, relu) -> Dense(1, sigmoid)
Features input, forget, output gates and constant error carousel (CEC) cell state.
"""

import os
from pathlib import Path
from typing import Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics

from config import (
    VOCAB_SIZE,
    MAX_SEQUENCE_LENGTH,
    EMBEDDING_DIM,
    LSTM_UNITS,
    DENSE_UNITS,
    DROPOUT_RATE,
    LEARNING_RATE,
    LSTM_MODEL_PATH
)


def build_lstm_model(
    vocab_size: int = VOCAB_SIZE,
    sequence_length: int = MAX_SEQUENCE_LENGTH,
    embedding_dim: int = EMBEDDING_DIM,
    lstm_units: int = LSTM_UNITS,
    dense_units: int = DENSE_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE
) -> tf.keras.Model:
    """
    Builds and compiles the LSTM architecture for binary sentiment classification.
    """
    inputs = layers.Input(shape=(sequence_length,), name="input_sequence", dtype="int32")

    # 1. Word Embedding Layer
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="word_embedding"
    )(inputs)

    # 2. LSTM Layer (Gated recurrence with cell state conveyor)
    x = layers.LSTM(
        units=lstm_units,
        return_sequences=False,
        name="lstm_layer"
    )(x)

    # 3. Regularization Dropout
    x = layers.Dropout(rate=dropout_rate, name="lstm_dropout")(x)

    # 4. Dense Layer with ReLU activation
    x = layers.Dense(units=dense_units, activation="relu", name="dense_features")(x)

    # 5. Output Layer with Sigmoid activation for binary classification
    outputs = layers.Dense(units=1, activation="sigmoid", name="sentiment_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="LSTM_Sentiment_Classifier")

    # Compile with identical loss, optimizer, and metrics as RNN for fair comparison
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall")
        ]
    )

    return model


def get_lstm_architecture_details(
    vocab_size: int = VOCAB_SIZE,
    sequence_length: int = MAX_SEQUENCE_LENGTH,
    embedding_dim: int = EMBEDDING_DIM,
    lstm_units: int = LSTM_UNITS,
    dense_units: int = DENSE_UNITS,
    dropout_rate: float = DROPOUT_RATE
) -> dict:
    """
    Returns structured architectural metadata for frontend inspection and academic reporting.
    """
    model = build_lstm_model(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout_rate=dropout_rate
    )

    layer_info = []
    for layer in model.layers:
        out_shape = getattr(layer, "output_shape", getattr(layer, "shape", getattr(layer, "batch_shape", "Unknown")))
        layer_info.append({
            "name": layer.name,
            "class": layer.__class__.__name__,
            "output_shape": str(out_shape),
            "params": int(layer.count_params())
        })

    return {
        "model_type": "LSTM",
        "description": "Long Short-Term Memory Network with forget, input, and output gating mechanisms",
        "total_params": int(model.count_params()),
        "trainable_params": int(sum([tf.size(w).numpy() for w in model.trainable_weights])),
        "layers": layer_info,
        "default_config": {
            "vocab_size": vocab_size,
            "sequence_length": sequence_length,
            "embedding_dim": embedding_dim,
            "lstm_units": lstm_units,
            "dense_units": dense_units,
            "dropout_rate": dropout_rate,
            "loss": "binary_crossentropy",
            "optimizer": "Adam"
        }
    }


def load_trained_lstm(model_path: Optional[Path] = None) -> Optional[tf.keras.Model]:
    """
    Loads trained LSTM model from disk if available.
    """
    model_path = model_path or LSTM_MODEL_PATH
    model_path = Path(model_path)
    if not model_path.exists():
        return None
    try:
        return models.load_model(model_path)
    except Exception as e:
        print(f"Error loading LSTM model from {model_path}: {e}")
        return None
