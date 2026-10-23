/**
 * Live Sentiment Analyzer UI Script
 * SentimentAI - RNN vs LSTM Analyzer
 */

const PRESETS = {
  pos: "The movie was extraordinarily magnificent and phenomenal! The acting was superb and the screenplay was brilliant.",
  neg: "A completely dreadful and boring film with terrible dialogue, wooden acting, and no redeeming qualities whatsoever.",
  early_pos: "The movie was extraordinarily magnificent and phenomenal, although the remaining several hours consisted of slow scenes, ordinary cinematography, generic background noise, and unremarkable characters.",
  late_pos: "The storyline spent hours introducing countless secondary characters, wandering through endless mundane conversations and average settings, but in the final moments delivered an astonishingly brilliant climax.",
  distant_neg: "I can not in any good conscience after witnessing the amateurish dialogue, convoluted narrative, and confusing timeline say that this production was good."
};

const NEGATIONS = new Set(["not", "no", "never", "neither", "nor", "none", "cannot", "cant", "wont", "dont", "didnt", "isnt", "arent", "wasnt", "werent", "hasnt", "havent", "hadnt", "couldnt", "shouldnt", "wouldnt", "hardly", "scarcely", "barely"]);

document.addEventListener('DOMContentLoaded', function () {
  const textInput = document.getElementById('analyzerTextInput');
  const analyzeBtn = document.getElementById('btnAnalyzeText');

  if (textInput) {
    textInput.addEventListener('input', updateCharCount);
    updateCharCount();
  }

  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', runLiveAnalysis);
  }
});

function updateCharCount() {
  const text = document.getElementById('analyzerTextInput').value;
  const counter = document.getElementById('charTokenCounter');
  if (counter) {
    const chars = text.length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    counter.textContent = `${chars} characters &bull; ~${words} words`;
  }
}

function setAnalyzerPreset(key) {
  const txt = PRESETS[key] || '';
  const input = document.getElementById('analyzerTextInput');
  if (input) {
    input.value = txt;
    updateCharCount();
    runLiveAnalysis();
  }
}

function runLiveAnalysis() {
  const text = document.getElementById('analyzerTextInput').value;
  const maxLen = document.getElementById('analyzerMaxLen').value;
  const analyzeBtn = document.getElementById('btnAnalyzeText');

  if (!text.trim()) {
    alert('Please enter text to analyze.');
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> Analyzing...';

  fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: text,
      max_length: parseInt(maxLen)
    })
  })
    .then((res) => res.json())
    .then((data) => {
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = '<i class="bi bi-cpu-fill me-1"></i> Analyze Sentiment';

      if (data.status === 'success') {
        populatePredictionUI(data);
      } else {
        alert(data.message || 'Analysis failed.');
      }
    })
    .catch((err) => {
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = '<i class="bi bi-cpu-fill me-1"></i> Analyze Sentiment';
      alert('Error during inference: ' + err);
    });
}

function populatePredictionUI(d) {
  document.getElementById('evalTimestamp').textContent = new Date().toLocaleTimeString();

  const rnn = d.predictions.rnn;
  const lstm = d.predictions.lstm;

  // Simple RNN UI
  if (rnn.available) {
    const isPos = rnn.sentiment === 'Positive';
    const badge = document.getElementById('badgeRnnSentiment');
    badge.className = `badge ${isPos ? 'bg-success' : 'bg-danger'}`;
    badge.textContent = rnn.sentiment;

    document.getElementById('valRnnProb').textContent = rnn.probability.toFixed(4);
    document.getElementById('barRnnConfidence').style.width = rnn.confidence_percent + '%';
    document.getElementById('barRnnConfidence').className = isPos ? 'bg-success h-100' : 'bg-danger h-100';
    document.getElementById('lblRnnConfidence').textContent = rnn.confidence_percent + '%';
    document.getElementById('lblRnnLatency').textContent = rnn.latency_ms + ' ms';
  } else {
    document.getElementById('badgeRnnSentiment').textContent = 'Untrained';
    document.getElementById('valRnnProb').textContent = '--';
  }

  // LSTM UI
  if (lstm.available) {
    const isPos = lstm.sentiment === 'Positive';
    const badge = document.getElementById('badgeLstmSentiment');
    badge.className = `badge ${isPos ? 'bg-success' : 'bg-danger'}`;
    badge.textContent = lstm.sentiment;

    document.getElementById('valLstmProb').textContent = lstm.probability.toFixed(4);
    document.getElementById('barLstmConfidence').style.width = lstm.confidence_percent + '%';
    document.getElementById('barLstmConfidence').className = isPos ? 'bg-success h-100' : 'bg-danger h-100';
    document.getElementById('lblLstmConfidence').textContent = lstm.confidence_percent + '%';
    document.getElementById('lblLstmLatency').textContent = lstm.latency_ms + ' ms';
  } else {
    document.getElementById('badgeLstmSentiment').textContent = 'Untrained';
    document.getElementById('valLstmProb').textContent = '--';
  }

  // Comparison Box
  const notice = document.getElementById('modelsAgreementNotice');
  if (d.comparison) {
    if (d.comparison.models_agree) {
      notice.className = 'p-2 rounded bg-success-subtle text-success border border-success text-center small fw-semibold';
      notice.innerHTML = `<i class="bi bi-check-circle-fill me-1"></i> Both Models Agree: ${rnn.sentiment} (Delta: ${d.comparison.probability_difference})`;
    } else {
      notice.className = 'p-2 rounded bg-danger-subtle text-danger border border-danger text-center small fw-semibold';
      notice.innerHTML = `<i class="bi bi-exclamation-triangle-fill me-1"></i> Divergence Detected! Simple RNN: ${rnn.sentiment} (${rnn.probability}) vs LSTM: ${lstm.sentiment} (${lstm.probability})`;
    }
  }

  // Render Tokens
  const container = document.getElementById('analyzerTokensContainer');
  container.innerHTML = '';
  (d.tokens || []).forEach((item) => {
    const isNeg = NEGATIONS.has(item.word.toLowerCase());
    let cls = 'token-badge';
    if (isNeg) cls += ' negation';
    if (item.is_oov) cls += ' oov';

    container.innerHTML += `<span class="${cls}" title="Vocab ID: ${item.index}">${item.word} [${item.index}]</span>`;
  });
}
