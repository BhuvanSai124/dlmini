"""
Text Preprocessing Module for Sentiment Analysis
Preserves crucial sentiment signals (especially negations), cleans HTML/whitespace,
handles tokenization and sequence padding.
"""

import re
import numpy as np

# Core negation words that must never be stripped during text cleaning
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none",
    "cannot", "can't", "cant", "won't", "wont", "don't", "dont",
    "didn't", "didnt", "isn't", "isnt", "aren't", "arent",
    "wasn't", "wasnt", "weren't", "werent", "hasn't", "hasnt",
    "haven't", "havent", "hadn't", "hadnt", "couldn't", "couldnt",
    "shouldn't", "shouldnt", "wouldn't", "wouldnt", "hardly", "scarcely", "barely"
}

# Standard English stopwords (excluding negations and sentiment modifiers)
FILTER_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more",
    "most", "my", "myself", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with",
    "would", "you", "your", "yours", "yourself", "yourselves"
} - NEGATION_WORDS  # Explicit guarantee: negations are NEVER in the filter list


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Cleans raw review text:
    1. Handles None / non-string inputs safely.
    2. Converts to lowercase.
    3. Strips HTML markup (<br />, etc.).
    4. Expands standard contractions to retain negation cues (e.g. n't -> not).
    5. Normalizes punctuation while separating word tokens.
    6. Preserves negation tokens.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Handle contractions to preserve negations
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)

    # 4. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 5. Keep letters and single spaces; replace other symbols with space
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 6. Normalize whitespace
    tokens = text.strip().split()

    if remove_stopwords:
        tokens = [t for t in tokens if t in NEGATION_WORDS or t not in FILTER_STOPWORDS]

    return " ".join(tokens)


def tokenize_text(text: str) -> list[str]:
    """
    Tokenizes cleaned text into a list of word tokens.
    """
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []


def pad_sequence(sequence: list[int], max_length: int, padding: str = "post", truncating: str = "post") -> list[int]:
    """
    Pads or truncates a 1D sequence of integers to max_length.
    Uses 0 for <PAD>.
    """
    seq = list(sequence)
    if len(seq) > max_length:
        if truncating == "post":
            seq = seq[:max_length]
        else:
            seq = seq[-max_length:]
    elif len(seq) < max_length:
        pad_len = max_length - len(seq)
        padding_zeros = [0] * pad_len
        if padding == "post":
            seq = seq + padding_zeros
        else:
            seq = padding_zeros + seq
    return seq


def pad_sequences_batch(sequences: list[list[int]], max_length: int, padding: str = "post", truncating: str = "post") -> np.ndarray:
    """
    Pads a list of integer sequences into a 2D numpy array of shape (num_samples, max_length).
    """
    padded = [pad_sequence(s, max_length, padding, truncating) for s in sequences]
    return np.array(padded, dtype=np.int32)
