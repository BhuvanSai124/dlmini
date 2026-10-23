"""
Configuration Module for Sentiment Analysis Project
Sentiment Analysis Using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM)
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
MODELS_RNN_DIR = MODELS_DIR / "rnn"
MODELS_LSTM_DIR = MODELS_DIR / "lstm"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"
RESULTS_PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Artifact and Asset Paths
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
VOCAB_CONFIG_PATH = MODELS_DIR / "vocab_config.json"
DATASET_METADATA_PATH = DATA_PROCESSED_DIR / "dataset_metadata.json"

RNN_MODEL_PATH = MODELS_RNN_DIR / "rnn_model.keras"
LSTM_MODEL_PATH = MODELS_LSTM_DIR / "lstm_model.keras"

RNN_HISTORY_PATH = RESULTS_METRICS_DIR / "rnn_history.json"
LSTM_HISTORY_PATH = RESULTS_METRICS_DIR / "lstm_history.json"

RNN_EVAL_PATH = RESULTS_METRICS_DIR / "rnn_eval.json"
LSTM_EVAL_PATH = RESULTS_METRICS_DIR / "lstm_eval.json"

COMPARISON_METRICS_PATH = RESULTS_METRICS_DIR / "comparison_metrics.json"
DEPENDENCY_RESULTS_PATH = RESULTS_METRICS_DIR / "dependency_results.json"

# Default Raw IMDB CSV Path
IMDB_CSV_PATH = DATA_RAW_DIR / "imdb_dataset.csv"

# Dataset Parameters
TOTAL_SAMPLES = 50000
TRAIN_SAMPLES = 20000
VAL_SAMPLES = 5000
TEST_SAMPLES = 25000

# Text Preprocessing & Vocabulary Parameters
VOCAB_SIZE = 10000  # Top 10,000 words
MAX_SEQUENCE_LENGTH = 250  # Default sequence length
ALLOWED_SEQUENCE_LENGTHS = [50, 100, 250, 500]
OOV_TOKEN = "<OOV>"
PAD_TOKEN = "<PAD>"
PADDING_TYPE = "post"
TRUNCATING_TYPE = "post"

# Model Hyperparameters
EMBEDDING_DIM = 64
RNN_UNITS = 64
LSTM_UNITS = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 64
DEFAULT_EPOCHS = 5
RANDOM_SEED = 42

# Ensure all essential directories exist
for folder in [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    MODELS_RNN_DIR,
    MODELS_LSTM_DIR,
    RESULTS_METRICS_DIR,
    RESULTS_PLOTS_DIR,
    RESULTS_PREDICTIONS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)
