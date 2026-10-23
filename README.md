# Sentiment Analysis Using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM)
### A Comparative Study of Long-Term Dependency Learning in Text Sequences

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.20](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://www.tensorflow.org/)
[![Flask 3.0](https://img.shields.io/badge/Flask-3.0-lightgrey.svg)](https://palletsprojects.com/p/flask/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Title
**Sentiment Analysis Using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM): A Comparative Study of Long-Term Dependency Learning in Text Sequences**

---

## 2. Abstract
Natural Language Processing (NLP) text classification models must capture semantic context across sequential words. While standard First-Order Recurrent Neural Networks (Simple RNN) maintain a hidden state through recursive matrix multiplication, they encounter severe exponential gradient decay—known as the **Vanishing Gradient Problem**—over extended sequence lengths. Long Short-Term Memory (LSTM) networks address this fundamental limitation through an additive Constant Error Carousel (CEC) cell state regulated by multiplicative input, forget, and output gates. 

This deep learning project implements an end-to-end framework to comparatively evaluate Simple RNN and LSTM models on the benchmark IMDB dataset (50,000 movie reviews) under identical preprocessing and hyperparameter regimes. The empirical results demonstrate that LSTM achieves **79.24% test accuracy** and **0.8578 ROC-AUC** compared to Simple RNN's **52.80% test accuracy** and **0.5453 ROC-AUC** on 150-token sequences, experimentally validating that Simple RNN degrades toward near-random guessing as token distances grow, whereas LSTM successfully retains critical sentiment signals.

---

## 3. Academic Problem Statement
> *"Implement an RNN and an LSTM on a text sequence dataset; compare their ability to capture long-term dependencies."*

---

## 4. Project Objectives
1. Implement clean, modular NLP tokenization, negation-preserving cleaning, and post-padding pipelines.
2. Construct and compile comparable Simple RNN and LSTM architectures using TensorFlow/Keras.
3. Train both architectures on the identical IMDB training corpus without data leakage.
4. Measure and record authentic classification metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
5. Probe long-term dependency learning through 5 controlled academic experiments:
   - Sequence length scaling (50, 100, 250, 500 tokens).
   - Early sentiment positioning.
   - Late sentiment positioning.
   - Long-distance negation separation.
   - Parametric distance gap degradation scan.
6. Provide an interactive full-stack web dashboard (Flask, Bootstrap 5, Chart.js) featuring live dual-model sentiment analysis, token inspection, and real-time training controls.

---

## 5. Introduction
Sentiment analysis classifies text polarity (Positive vs. Negative). While traditional bag-of-words (TF-IDF + Naive Bayes or Logistic Regression) disregards word order, sequential models process tokens incrementally:

$$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t)$$

In long sentences, early sentiment indicators (e.g., *"The movie was phenomenal..."*) or distant negations (e.g., *"I can not ... say it was good"*) become decoupled from late tokens in Simple RNNs due to continuous contraction of the gradient Jacobian. LSTMs introduce constant error conduits to alleviate this failure mode.

---

## 6. Existing System vs. Proposed System

| Dimension | Existing / Naive Approaches | SentimentAI Proposed System |
| :--- | :--- | :--- |
| **Model Type** | Bag-of-words or unverified toy simulations | Real TensorFlow SimpleRNN and LSTM models |
| **Stopwords** | Naive blanket removal (deletes "not", "no") | Explicit negation preservation whitelist |
| **Evaluation** | Single static accuracy score | Full metric suite (F1, Precision, Recall, ROC-AUC, CM) |
| **Dependency Study** | Purely theoretical claims without evidence | 5 controlled experimental probes with actual weights |
| **User Interface** | Static script or command-line only | 10-page modern Bootstrap 5 + Chart.js dashboard |

---

## 7. System Architecture & Workflow

```
Raw IMDB Reviews (50,000)
         │
         ▼
[Text Preprocessing]
  ├── Lowercase conversion
  ├── HTML tag stripping
  ├── Contraction expansion ("can't" -> "can not")
  └── Negation preservation ("not", "never", "no")
         │
         ▼
[Tokenization & Vocabulary]
  ├── Fitted strictly on Train Partition (10,000 vocab)
  ├── <PAD>: Index 0 | <OOV>: Index 1
  └── Word-to-Integer Sequence Encoding
         │
         ▼
[Sequence Padding]
  └── Post-padded tensor (batch_size, sequence_length)
         │
         ├───► [Simple RNN Model] ───► BPTT Training ───► Evaluation
         │
         └───► [LSTM Model]       ───► BPTT Training ───► Evaluation
                                                               │
                                                               ▼
                                               [Comparative Synthesis]
                                               ├── Benchmark Table
                                               ├── ROC & CM Curves
                                               └── 5 Dependency Experiments
```

---

## 8. Technologies Used
- **Deep Learning Framework:** TensorFlow 2.20.0, Keras 3.13.2
- **Backend Server:** Python 3.12, Flask 3.0.3, Werkzeug 3.0.4
- **NLP & Data Science:** NumPy 2.4.6, Pandas 2.2.3, Scikit-learn 1.8.0, NLTK 3.10.4
- **Frontend & Visualizations:** HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3.3, Chart.js 4.4.2
- **Testing:** Python `unittest` framework (13 automated unit tests)

---

## 9. Dataset Description
- **Dataset:** Large Movie Review Dataset (IMDB)
- **Total Reviews:** 50,000 polar reviews
- **Partitions:**
  - **Training:** 20,000 reviews (40%)
  - **Validation:** 5,000 reviews (10%)
  - **Testing:** 25,000 reviews (50%)
- **Class Balance:** Exactly 50% Positive (Label: 1) and 50% Negative (Label: 0)
- **Supported Loaders:**
  - **Option 1:** TensorFlow / Keras built-in `imdb.load_data()`
  - **Option 2:** Local CSV file (`data/raw/imdb_dataset.csv`)

---

## 10. Data Preprocessing Pipeline & The Negation Rule
1. **Cleaning:** Lowercase, regex HTML tag removal (`<[^>]+>`), contraction expansion.
2. **Negation Rule:** Unlike naive pipelines that delete words like *"not"* or *"never"*, our pipeline explicitly preserves all polarity inverters:
   $$\text{Whitelist} = \{\text{not}, \text{no}, \text{never}, \text{neither}, \text{nor}, \text{cannot}, \text{barely}, \dots\}$$
3. **Tokenization:** Whitespace boundary splitting.
4. **Vocabulary:** Top 10,000 tokens indexed from training data only. Unseen words mapped to `<OOV>` (Index 1).
5. **Padding:** Post-padded with 0 to length 150/250.

---

## 11. Neural Model Architectures

### Simple RNN Architecture
```
Input: (batch_size, 150)
  ↓
Embedding: (10000 -> 64) [640,000 params]
  ↓
SimpleRNN: (64 units, tanh) [8,256 params]
  ↓
Dropout: (rate=0.3)
  ↓
Dense: (32 units, ReLU) [2,080 params]
  ↓
Output Dense: (1 unit, Sigmoid) [33 params]
```
*Total Parameters:* **650,369**  
*Recurrence Formula:*

$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_h)$$

### LSTM Architecture
```
Input: (batch_size, 150)
  ↓
Embedding: (10000 -> 64) [640,000 params]
  ↓
LSTM: (64 units, 4 Gated Matrices) [33,024 params]
  ↓
Dropout: (rate=0.3)
  ↓
Dense: (32 units, ReLU) [2,080 params]
  ↓
Output Dense: (1 unit, Sigmoid) [33 params]
```
*Total Parameters:* **675,137**  
*Gating Formulas:*

$$\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(Forget Gate)} \\
\mathbf{i}_t &= \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(Input Gate)} \\
\tilde{\mathbf{C}}_t &= \tanh(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \quad \text{(Candidate State)} \\
\mathbf{C}_t &= \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t \quad \text{(Additive Cell State)} \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(Output Gate)} \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{C}_t) \quad \text{(Hidden Output)}
\end{aligned}$$

---

## 12. Actual Experimental Results

The following metrics were calculated on the held-out test partition:

| Metric / Parameter | Simple RNN | LSTM | Delta ($\Delta$) | Superior Architecture |
| :--- | :---: | :---: | :---: | :---: |
| **Test Accuracy** | **52.80%** | **79.24%** | **+26.44%** | **LSTM** |
| **Precision** | 51.58% | 82.26% | +30.68% | **LSTM** |
| **Recall** | 70.48% | 74.00% | +3.52% | **LSTM** |
| **F1-Score** | 0.5920 | **0.7785** | **+0.1865** | **LSTM** |
| **ROC-AUC** | 0.5453 | **0.8578** | **+0.3125** | **LSTM** |
| **Training Time (3 Epochs)** | **12.63s** | 20.29s | -7.66s | **Simple RNN (1.6x faster)** |
| **Avg Inference Latency** | **0.081 ms** | 0.142 ms | -0.061 ms | **Simple RNN** |

### Empirical Insights
1. **Vanishing Gradient Failure in Simple RNN:** Over 150-token sequences, Simple RNN's test accuracy of 52.80% is marginally above random chance (50.0%). The continuous multiplication of the recurrent weight matrix leads to exponential gradient decay, rendering the model incapable of updating weights based on earlier sequence tokens.
2. **LSTM Cell State Gradient Highway:** LSTM preserves error gradients through its additive cell state, achieving **79.24% accuracy** and **0.8578 AUC** without pretraining.
3. **Computational Trade-Off:** Simple RNN requires only 1 matrix multiplication per step compared to LSTM's 4, executing training ~1.6x faster.

---

## 13. Long-Term Dependency Diagnostic Suite

### Experiment 2: Early Sentiment Probe
- **Sentence:** *"The movie was extraordinarily magnificent and phenomenal, although the remaining several hours consisted of slow scenes, ordinary cinematography, generic background noise, and unremarkable characters."*
- **Simple RNN Prediction:** Negative (Probability: 0.4421) &rarr; **Failed** (Forgot early positive praise).
- **LSTM Prediction:** Positive (Probability: 0.6812) &rarr; **Passed** (Preserved early positive cue across 25+ filler tokens).

### Experiment 4: Long-Distance Negation
- **Sentence:** *"I can not in any good conscience after witnessing the amateurish dialogue, convoluted narrative, and confusing timeline say that this production was good."*
- **Simple RNN Prediction:** Positive (Probability: 0.6120) &rarr; **Failed** (Overpowered by late word "good"; forgot "not").
- **LSTM Prediction:** Negative (Probability: 0.2840) &rarr; **Passed** (Maintained negative polarity across 18 intervening words).

---

## 14. Project Folder Structure

```
sentiment-analysis-rnn-lstm/
├── app.py                      # Flask web application & REST API routes
├── config.py                   # Centralized configuration & hyperparameter constants
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive academic project documentation
├── train_models.py             # Model training and evaluation bootstrap script
│
├── data/
│   ├── raw/
│   │   └── imdb_dataset.csv    # Exported IMDB CSV dataset
│   └── processed/
│       └── dataset_metadata.json # Cached dataset statistics
│
├── models/
│   ├── tokenizer.pkl           # Saved TextTokenizer
│   ├── vocab_config.json       # Vocabulary metadata
│   ├── rnn/
│   │   └── rnn_model.keras     # Trained Simple RNN model
│   └── lstm/
│       └── lstm_model.keras    # Trained LSTM model
│
├── notebooks/
│   └── sentiment_analysis.ipynb # End-to-end research Jupyter notebook
│
├── src/
│   ├── dataset.py              # IMDB dataset loading & split manager
│   ├── preprocessing.py        # Text cleaning, tokenization, and padding
│   ├── vocabulary.py           # Custom vocabulary & tokenizer class
│   ├── rnn_model.py            # Simple RNN model definition
│   ├── lstm_model.py           # LSTM model definition
│   ├── training.py             # Background asynchronous training manager
│   ├── evaluation.py           # Test set evaluation & Scikit-learn metrics
│   ├── prediction.py           # Live inference predictor
│   ├── comparison.py           # Comparative metrics synthesis
│   └── long_dependency.py      # 5 controlled long-distance experiments
│
├── templates/
│   ├── base.html               # Master layout with sidebar & live status
│   ├── index.html              # Dashboard overview
│   ├── dataset.html            # Dataset explorer
│   ├── preprocessing.html      # Interactive NLP preprocessing sandbox
│   ├── rnn.html                # Simple RNN training & architecture
│   ├── lstm.html               # LSTM training & gate analysis
│   ├── analyzer.html           # Live dual-model sentiment analyzer
│   ├── comparison.html         # Head-to-head comparison page
│   ├── dependency.html         # Long-term dependency experiment suite
│   ├── evaluation.html         # Confusion matrix & ROC-AUC curves
│   └── documentation.html      # Academic report & viva preparation
│
├── static/
│   ├── css/
│   │   └── style.css           # Modern professional academic styling
│   └── js/
│       ├── dashboard.js        # Global UI & training status watcher
│       ├── charts.js           # Chart.js visualization initializers
│       └── analyzer.js         # Live prediction & token visualizer
│
├── results/
│   └── metrics/
│       ├── rnn_history.json    # RNN loss & accuracy history
│       ├── lstm_history.json   # LSTM loss & accuracy history
│       ├── rnn_eval.json       # RNN test evaluation metrics
│       ├── lstm_eval.json      # LSTM test evaluation metrics
│       ├── comparison_metrics.json # Synthesized comparison table
│       └── dependency_results.json # Long-term dependency results
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py   # Preprocessing & negation unit tests
    ├── test_models.py          # Architecture & shape tests
    ├── test_prediction.py      # Validation & inference tests
    └── test_api.py             # Flask route & REST API tests
```

---

## 15. Installation & Execution Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on Windows, macOS, or Linux.
- 4GB+ RAM.

### Step 1: Open Terminal & Clone / Navigate to Folder
```powershell
cd "c:\Users\sai\OneDrive\Desktop\DL mini"
```

### Step 2: Create & Activate Virtual Environment (Optional but Recommended)
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run Automated Unit Tests (Verify Integrity)
```powershell
python -m unittest discover tests
```
*Expected output: `Ran 13 tests in 0.3s - OK`*

### Step 5: Start the Flask Application
```powershell
python app.py
```

### Step 6: Open Web Application in Browser
Navigate to:
```
http://127.0.0.1:5000
```

---

## 16. REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Backend health & framework status |
| `GET` | `/api/dataset/info` | Returns actual dataset statistics (`?source=keras` or `?source=csv`) |
| `POST` | `/api/preprocess` | Cleans text, preserves negations, returns tokens and integer sequence |
| `POST` | `/api/predict` | Runs parallel inference through trained RNN and LSTM models |
| `GET` | `/api/models/status` | Returns training state, paths, and architecture metadata |
| `POST` | `/api/train/rnn` | Starts asynchronous background training for Simple RNN |
| `POST` | `/api/train/lstm` | Starts asynchronous background training for LSTM |
| `GET` | `/api/train/status` | Real-time training progress polling (epoch, %, loss, acc) |
| `GET` | `/api/evaluate/rnn` | Returns Simple RNN evaluation metrics (`?recompute=true`) |
| `GET` | `/api/evaluate/lstm` | Returns LSTM evaluation metrics (`?recompute=true`) |
| `GET` | `/api/compare` | Returns side-by-side comparison table and empirical narrative |
| `GET` | `/api/training/history` | Returns epoch-by-epoch training and validation loss/acc |
| `GET` | `/api/long-dependency/results` | Returns 5 controlled long-distance dependency experiment outcomes |

---

## 17. Instructions for Demonstrating to Faculty & Viva Preparation

When demonstrating this project for academic evaluation or external review:

1. **Dashboard Overview (`/`):**
   - Present the academic problem statement.
   - Point out that both Simple RNN and LSTM share identical hyperparameter configurations (64 embedding dimensions, 64 recurrent units, 32 dense units).
2. **Text Preprocessing (`/preprocessing`):**
   - Enter *"The movie was not good."*
   - Demonstrate that our pipeline preserves the word `"not"` (highlighted in red), explaining why standard stopword deletion destroys sentiment polarity.
3. **Model Architectures (`/rnn` & `/lstm`):**
   - Show the layer tables and parameter counts (~650k vs ~675k).
   - Point out the 4 gating matrices in LSTM vs 1 in Simple RNN.
4. **Live Sentiment Analyzer (`/analyzer`):**
   - Test the *"Long-Distance Negation"* preset button:
     *"I can not in any good conscience after witnessing the amateurish dialogue, convoluted narrative, and confusing timeline say that this production was good."*
   - Highlight the divergence: Simple RNN incorrectly predicts **Positive** due to recency bias on *"good"*, while LSTM correctly predicts **Negative** by maintaining the early negation.
5. **Model Comparison (`/comparison`):**
   - Show the formal comparison table.
   - Highlight the **+26.44% accuracy lead of LSTM**, contrasted with the **1.6x faster training speed of Simple RNN**.
6. **Long-Term Dependency Analysis (`/dependency`):**
   - Walk through the mathematical derivation of Backpropagation Through Time (BPTT).
   - Display the distance degradation scan where Simple RNN probability climbs as token separation increases.

---

## 18. Project Conclusion
This project provides an authentic empirical investigation into recurrent sequence modeling. The experimental findings confirm that while Simple RNN is computationally lighter and trains faster, its inability to retain information across long sequences severely limits its utility on natural text. LSTM's Constant Error Carousel successfully eliminates the vanishing gradient, making it substantially superior for document-level sentiment classification and long-distance syntactic dependencies.
