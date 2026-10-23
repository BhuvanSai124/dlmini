"""
Unit Tests for Sentiment Prediction Module and Input Validation
"""

import unittest
from src.prediction import SentimentPredictor


class TestPrediction(unittest.TestCase):

    def setUp(self):
        self.predictor = SentimentPredictor()

    def test_empty_input_validation(self):
        res1 = self.predictor.predict_text("")
        self.assertEqual(res1["status"], "error")
        self.assertIn("non-empty", res1["message"])

        res2 = self.predictor.predict_text("   \n\t  ")
        self.assertEqual(res2["status"], "error")

        res3 = self.predictor.predict_text(None)
        self.assertEqual(res3["status"], "error")


if __name__ == "__main__":
    unittest.main()
