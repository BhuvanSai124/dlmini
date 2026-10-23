"""
Dataset Loading, Splitting, and Statistics Module
Supports both Option 1 (TensorFlow/Keras IMDB) and Option 2 (Local IMDB CSV).
Calculates authentic dataset statistics without fabrication.
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd

from config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATASET_METADATA_PATH,
    IMDB_CSV_PATH,
    RANDOM_SEED,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
    TEST_SAMPLES,
)


class IMDBDatasetManager:
    """
    Manages loading, splitting, and summarizing the IMDB Sentiment Dataset.
    Supports Keras native loading and CSV loading.
    """

    def __init__(self, random_seed: int = RANDOM_SEED):
        self.random_seed = random_seed
        self.active_source: str = "Not loaded"
        self.metadata: Dict[str, Any] = {}

    def decode_keras_sequence(self, sequence: list[int], inv_word_index: dict[int, str]) -> str:
        """
        Decodes a Keras integer sequence back to original text words.
        Indices are offset by 3: 0=PAD, 1=START, 2=UNK, 3=UNUSED.
        """
        return " ".join([inv_word_index.get(idx, "<UNK>") for idx in sequence if idx > 1])

    def load_from_keras(
        self,
        num_words: Optional[int] = None,
        train_size: int = TRAIN_SAMPLES,
        val_size: int = VAL_SAMPLES
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Option 1: Load IMDB dataset directly through TensorFlow/Keras.
        Returns (train_df, val_df, test_df) each with ['text', 'label', 'sentiment'].
        """
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        from tensorflow.keras.datasets import imdb

        print("Loading IMDB dataset via TensorFlow/Keras...")
        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = imdb.load_data(num_words=num_words)
        word_index = imdb.get_word_index()
        inv_word_index = {i + 3: w for w, i in word_index.items()}
        inv_word_index[0] = "<PAD>"
        inv_word_index[1] = "<START>"
        inv_word_index[2] = "<UNK>"
        inv_word_index[3] = "<UNUSED>"

        # Stratified / reproducible split for train and validation
        rng = np.random.RandomState(self.random_seed)
        indices = np.arange(len(x_train_raw))
        rng.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]

        # Decode only the needed subsets for optimal performance
        train_texts = [self.decode_keras_sequence(x_train_raw[i], inv_word_index) for i in train_indices]
        val_texts = [self.decode_keras_sequence(x_train_raw[i], inv_word_index) for i in val_indices]
        
        # For testing, decode test_size (default all or up to 5000 for rapid evaluation)
        test_texts = [self.decode_keras_sequence(x_test_raw[i], inv_word_index) for i in range(len(x_test_raw))]

        train_df = pd.DataFrame({
            "text": train_texts,
            "label": y_train_raw[train_indices].astype(int),
            "sentiment": ["positive" if y == 1 else "negative" for y in y_train_raw[train_indices]]
        })

        val_df = pd.DataFrame({
            "text": val_texts,
            "label": y_train_raw[val_indices].astype(int),
            "sentiment": ["positive" if y == 1 else "negative" for y in y_train_raw[val_indices]]
        })

        test_df = pd.DataFrame({
            "text": test_texts,
            "label": y_test_raw.astype(int),
            "sentiment": ["positive" if y == 1 else "negative" for y in y_test_raw]
        })

        self.active_source = "TensorFlow / Keras Built-in IMDB Dataset"
        self._calculate_metadata(train_df, val_df, test_df)

        return train_df, val_df, test_df

    def load_from_csv(
        self,
        csv_path: Optional[Path] = None,
        train_size: int = TRAIN_SAMPLES,
        val_size: int = VAL_SAMPLES
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Option 2: Load IMDB dataset from CSV file.
        CSV must have review/text column and sentiment/label column.
        """
        csv_path = csv_path or IMDB_CSV_PATH
        csv_path = Path(csv_path)

        if not csv_path.exists():
            # If CSV does not exist yet, build it automatically from Keras IMDB!
            print(f"CSV not found at {csv_path}. Generating from Keras IMDB...")
            self.export_keras_to_csv(csv_path)

        print(f"Loading IMDB dataset from CSV: {csv_path}...")
        df = pd.read_csv(csv_path)

        # Standardize column names
        text_col = "review" if "review" in df.columns else ("text" if "text" in df.columns else df.columns[0])
        label_col = "sentiment" if "sentiment" in df.columns else ("label" if "label" in df.columns else df.columns[1])

        df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "sentiment"})
        df["text"] = df["text"].astype(str)

        # Normalize sentiment / labels
        if df["sentiment"].dtype == object:
            df["sentiment"] = df["sentiment"].str.strip().str.lower()
            df["label"] = (df["sentiment"] == "positive").astype(int)
        else:
            df["label"] = df["sentiment"].astype(int)
            df["sentiment"] = ["positive" if y == 1 else "negative" for y in df["label"]]

        # Handle split column if present, else perform reproducible split
        if "split" in df.columns:
            train_full = df[df["split"] == "train"].reset_index(drop=True)
            test_df = df[df["split"] == "test"].reset_index(drop=True)
        else:
            rng = np.random.RandomState(self.random_seed)
            shuffled = df.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)
            train_full = shuffled.iloc[:25000].reset_index(drop=True)
            test_df = shuffled.iloc[25000:50000].reset_index(drop=True)

        rng = np.random.RandomState(self.random_seed)
        shuffled_train_idx = np.arange(len(train_full))
        rng.shuffle(shuffled_train_idx)

        train_indices = shuffled_train_idx[:train_size]
        val_indices = shuffled_train_idx[train_size:train_size + val_size]

        train_df = train_full.iloc[train_indices].reset_index(drop=True)
        val_df = train_full.iloc[val_indices].reset_index(drop=True)

        self.active_source = f"Local CSV File ({csv_path.name})"
        self._calculate_metadata(train_df, val_df, test_df)

        return train_df, val_df, test_df

    def export_keras_to_csv(self, output_path: Optional[Path] = None):
        """
        Exports the 50,000 IMDB reviews to a local CSV so Option 2 works without external downloads.
        """
        output_path = output_path or IMDB_CSV_PATH
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        from tensorflow.keras.datasets import imdb

        print("Exporting Keras IMDB dataset to CSV...")
        (x_train, y_train), (x_test, y_test) = imdb.load_data()
        word_index = imdb.get_word_index()
        inv_word_index = {i + 3: w for w, i in word_index.items()}
        inv_word_index[0] = "<PAD>"
        inv_word_index[1] = "<START>"
        inv_word_index[2] = "<UNK>"
        inv_word_index[3] = "<UNUSED>"

        train_texts = [self.decode_keras_sequence(seq, inv_word_index) for seq in x_train]
        test_texts = [self.decode_keras_sequence(seq, inv_word_index) for seq in x_test]

        df_train = pd.DataFrame({
            "review": train_texts,
            "sentiment": ["positive" if y == 1 else "negative" for y in y_train],
            "split": "train"
        })
        df_test = pd.DataFrame({
            "review": test_texts,
            "sentiment": ["positive" if y == 1 else "negative" for y in y_test],
            "split": "test"
        })

        full_df = pd.concat([df_train, df_test], ignore_index=True)
        full_df.to_csv(output_path, index=False)
        print(f"IMDB CSV successfully saved to {output_path} ({len(full_df)} records).")

    def _calculate_metadata(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Computes authentic dataset statistics from actual loaded dataframes.
        """
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        all_df["word_count"] = all_df["text"].apply(lambda t: len(t.split()))

        # Vocab count
        unique_words = set()
        for text in train_df["text"]:
            unique_words.update(text.lower().split())

        pos_count = int((all_df["label"] == 1).sum())
        neg_count = int((all_df["label"] == 0).sum())

        pos_samples = train_df[train_df["label"] == 1]["text"].head(3).tolist()
        neg_samples = train_df[train_df["label"] == 0]["text"].head(3).tolist()

        self.metadata = {
            "dataset_name": "IMDB Large Movie Review Dataset",
            "active_source": self.active_source,
            "total_reviews": len(all_df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "positive_reviews": pos_count,
            "negative_reviews": neg_count,
            "positive_percentage": round((pos_count / len(all_df)) * 100, 2),
            "negative_percentage": round((neg_count / len(all_df)) * 100, 2),
            "vocabulary_size_train": len(unique_words),
            "avg_review_length": round(float(all_df["word_count"].mean()), 1),
            "min_review_length": int(all_df["word_count"].min()),
            "max_review_length": int(all_df["word_count"].max()),
            "median_review_length": int(all_df["word_count"].median()),
            "sample_positive_reviews": pos_samples,
            "sample_negative_reviews": neg_samples,
        }

        # Save metadata
        DATASET_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DATASET_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns cached or loaded metadata.
        """
        if self.metadata:
            return self.metadata
        if DATASET_METADATA_PATH.exists():
            with open(DATASET_METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                return self.metadata
        # If not present, load via keras to compute metadata
        self.load_from_keras()
        return self.metadata
