"""
Unit Tests for NLP Text Preprocessing Pipeline
"""

import unittest
from src.preprocessing import clean_text, tokenize_text, pad_sequence, pad_sequences_batch, NEGATION_WORDS


class TestPreprocessing(unittest.TestCase):

    def test_lowercase_and_html_stripping(self):
        raw = "The MOVIE was <br/><br/> FANTASTIC! <p>Truly great</p>"
        cleaned = clean_text(raw)
        self.assertNotIn("<br/>", cleaned)
        self.assertNotIn("<p>", cleaned)
        self.assertEqual(cleaned, "the movie was fantastic truly great")

    def test_negation_preservation(self):
        """Crucial test: Ensure negation words are never stripped!"""
        negation_sentence = "This was not a good movie and I could never recommend it."
        cleaned = clean_text(negation_sentence)
        tokens = tokenize_text(negation_sentence)
        self.assertIn("not", tokens)
        self.assertIn("never", tokens)
        self.assertTrue(set(tokens).intersection(NEGATION_WORDS))

    def test_contractions_expansion(self):
        text = "I can't believe it wasn't good"
        cleaned = clean_text(text)
        self.assertIn("not", cleaned)

    def test_empty_and_none_handling(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text("   "), "")
        self.assertEqual(tokenize_text(""), [])

    def test_padding_and_truncation(self):
        seq = [10, 20, 30]
        # Pad to 5
        padded = pad_sequence(seq, max_length=5, padding="post")
        self.assertEqual(padded, [10, 20, 30, 0, 0])

        # Truncate to 2
        truncated = pad_sequence(seq, max_length=2, truncating="post")
        self.assertEqual(truncated, [10, 20])

    def test_batch_padding(self):
        batch = [[1, 2], [3, 4, 5, 6]]
        arr = pad_sequences_batch(batch, max_length=4)
        self.assertEqual(arr.shape, (2, 4))
        self.assertEqual(arr[0].tolist(), [1, 2, 0, 0])
        self.assertEqual(arr[1].tolist(), [3, 4, 5, 6])


if __name__ == "__main__":
    unittest.main()
