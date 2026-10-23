"""
Unit Tests for Deep Learning Architectures (Simple RNN and LSTM)
"""

import unittest
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.rnn_model import build_rnn_model
from src.lstm_model import build_lstm_model


class TestModelArchitectures(unittest.TestCase):

    def test_rnn_architecture(self):
        model = build_rnn_model(
            vocab_size=1000,
            sequence_length=50,
            embedding_dim=32,
            rnn_units=16,
            dense_units=8,
            dropout_rate=0.2
        )
        self.assertIsNotNone(model)
        # Check output shape: (batch_size, 1)
        self.assertEqual(model.output_shape, (None, 1))
        # Check layer types
        layer_classes = [l.__class__.__name__ for l in model.layers]
        self.assertIn("Embedding", layer_classes)
        self.assertIn("SimpleRNN", layer_classes)
        self.assertIn("Dropout", layer_classes)
        self.assertIn("Dense", layer_classes)

    def test_lstm_architecture(self):
        model = build_lstm_model(
            vocab_size=1000,
            sequence_length=50,
            embedding_dim=32,
            lstm_units=16,
            dense_units=8,
            dropout_rate=0.2
        )
        self.assertIsNotNone(model)
        self.assertEqual(model.output_shape, (None, 1))
        layer_classes = [l.__class__.__name__ for l in model.layers]
        self.assertIn("Embedding", layer_classes)
        self.assertIn("LSTM", layer_classes)
        self.assertIn("Dropout", layer_classes)
        self.assertIn("Dense", layer_classes)


if __name__ == "__main__":
    unittest.main()
