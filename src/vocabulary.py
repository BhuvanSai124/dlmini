"""
Vocabulary and Tokenizer Module
Handles vocabulary creation exclusively from training data, word-to-index mapping,
OOV handling, and conversion between text tokens and integer sequences.
"""

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Optional

from config import (
    VOCAB_SIZE,
    OOV_TOKEN,
    PAD_TOKEN,
    TOKENIZER_PATH,
    VOCAB_CONFIG_PATH
)
from src.preprocessing import clean_text, tokenize_text


class TextTokenizer:
    """
    Custom lightweight, deterministic text tokenizer compatible with standard Keras embeddings.
    Index 0 is reserved for PAD.
    Index 1 is reserved for OOV.
    Indices 2 to vocab_size-1 are assigned to top frequent words.
    """

    def __init__(self, num_words: int = VOCAB_SIZE, oov_token: str = OOV_TOKEN):
        self.num_words = num_words
        self.oov_token = oov_token
        self.pad_token = PAD_TOKEN
        self.word_index: dict[str, int] = {self.pad_token: 0, self.oov_token: 1}
        self.index_word: dict[int, str] = {0: self.pad_token, 1: self.oov_token}
        self.word_counts: Counter = Counter()
        self.is_fitted: bool = False

    def fit_on_texts(self, texts: list[str]) -> "TextTokenizer":
        """
        Builds vocabulary based on text frequency, strictly on the training set.
        """
        self.word_counts = Counter()
        for text in texts:
            tokens = tokenize_text(text)
            self.word_counts.update(tokens)

        # Most frequent words
        most_common = self.word_counts.most_common(self.num_words - 2)

        self.word_index = {self.pad_token: 0, self.oov_token: 1}
        self.index_word = {0: self.pad_token, 1: self.oov_token}

        for idx, (word, _) in enumerate(most_common, start=2):
            self.word_index[word] = idx
            self.index_word[idx] = word

        self.is_fitted = True
        return self

    def texts_to_sequences(self, texts: list[str]) -> list[list[int]]:
        """
        Converts a list of texts into sequences of integers.
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted or loaded before converting texts to sequences.")

        sequences = []
        oov_idx = self.word_index[self.oov_token]

        for text in texts:
            tokens = tokenize_text(text)
            seq = [self.word_index.get(tok, oov_idx) for tok in tokens]
            sequences.append(seq)
        return sequences

    def sequence_to_text(self, sequence: list[int], skip_special: bool = False) -> str:
        """
        Converts an integer sequence back to reconstructed text.
        """
        words = []
        for idx in sequence:
            if skip_special and idx in (0, 1):
                continue
            words.append(self.index_word.get(idx, self.oov_token))
        return " ".join(words)

    def save(self, pkl_path: Optional[Path] = None, json_path: Optional[Path] = None):
        """
        Saves tokenizer state to disk.
        """
        pkl_path = pkl_path or TOKENIZER_PATH
        json_path = json_path or VOCAB_CONFIG_PATH

        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)

        meta = {
            "num_words": self.num_words,
            "oov_token": self.oov_token,
            "vocab_actual_size": len(self.word_index),
            "sample_words": list(self.word_index.keys())[:30],
            "total_unique_words_seen": len(self.word_counts),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, pkl_path: Optional[Path] = None) -> "TextTokenizer":
        """
        Loads tokenizer from disk.
        """
        pkl_path = pkl_path or TOKENIZER_PATH
        if not pkl_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found at {pkl_path}")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
