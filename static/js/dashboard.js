/**
 * Dashboard & Global UI Logic
 * SentimentAI - RNN vs LSTM Analyzer
 */

let globalTrainingInterval = null;

document.addEventListener('DOMContentLoaded', function () {
  initSidebarToggle();
  startGlobalTrainingWatcher();

  const refreshBtn = document.getElementById('btnRefreshStatus');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', loadDashboardData);
  }

  const quickPredBtn = document.getElementById('btnDashboardPredict');
  if (quickPredBtn) {
    quickPredBtn.addEventListener('click', runDashboardQuickPrediction);
  }
});

function initSidebarToggle() {
  const toggleBtn = document.getElementById('sidebarToggle');
  const sidebar = document.querySelector('.app-sidebar');
  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('show');
    });
  }
}

function startGlobalTrainingWatcher() {
  checkGlobalTrainingStatus();
  globalTrainingInterval = setInterval(checkGlobalTrainingStatus, 2500);
}

function checkGlobalTrainingStatus() {
  fetch('/api/train/status')
    .then((res) => res.json())
    .then((resp) => {
      if (resp.status === 'success') {
        const d = resp.data;
        const dot = document.getElementById('trainingStatusDot');
        const txt = document.getElementById('trainingStatusText');

        if (!dot || !txt) return;

        if (d.status === 'training' || d.status === 'preparing') {
          dot.className = 'status-dot warning';
          txt.textContent = `Training ${d.model_type ? d.model_type.toUpperCase() : ''} (${d.progress_pct || 0}%)`;
        } else if (d.status === 'completed') {
          dot.className = 'status-dot active';
          txt.textContent = 'Training Completed';
        } else if (d.status === 'error') {
          dot.className = 'status-dot danger';
          txt.textContent = 'Training Error';
        } else {
          dot.className = 'status-dot idle';
          txt.textContent = 'Models Idle';
        }
      }
    })
    .catch(() => {});
}

function loadDashboardData() {
  // Check model statuses
  fetch('/api/models/status')
    .then((res) => res.json())
    .then((data) => {
      const rnnStatus = document.getElementById('statRnnStatus');
      const lstmStatus = document.getElementById('statLstmStatus');
      const badgeRnn = document.getElementById('badgeRnnTrained');
      const badgeLstm = document.getElementById('badgeLstmTrained');

      if (rnnStatus) {
        rnnStatus.textContent = data.rnn.trained ? 'Trained' : 'Untrained';
        rnnStatus.className = data.rnn.trained ? 'stat-value text-success' : 'stat-value text-warning';
      }
      if (badgeRnn) {
        badgeRnn.textContent = data.rnn.trained ? 'Weights Ready' : 'Untrained';
        badgeRnn.className = data.rnn.trained ? 'badge bg-success-subtle text-success' : 'badge bg-warning-subtle text-warning';
      }

      if (lstmStatus) {
        lstmStatus.textContent = data.lstm.trained ? 'Trained' : 'Untrained';
        lstmStatus.className = data.lstm.trained ? 'stat-value text-success' : 'stat-value text-warning';
      }
      if (badgeLstm) {
        badgeLstm.textContent = data.lstm.trained ? 'Weights Ready' : 'Untrained';
        badgeLstm.className = data.lstm.trained ? 'badge bg-success-subtle text-success' : 'badge bg-warning-subtle text-warning';
      }
    });

  // Load evaluation summaries
  fetch('/api/compare')
    .then((res) => res.json())
    .then((resp) => {
      if (resp.status === 'success') {
        const d = resp.data;
        const rnnEl = document.getElementById('statRnnAcc');
        const lstmEl = document.getElementById('statLstmAcc');

        if (rnnEl && d.rnn_evaluation) {
          rnnEl.textContent = `Test Acc: ${(d.rnn_evaluation.accuracy * 100).toFixed(2)}%`;
        }
        if (lstmEl && d.lstm_evaluation) {
          lstmEl.textContent = `Test Acc: ${(d.lstm_evaluation.accuracy * 100).toFixed(2)}%`;
        }

        if (typeof renderDashboardHistoryCharts === 'function') {
          renderDashboardHistoryCharts(d.rnn_history, d.lstm_history);
        }
      }
    });
}

function runDashboardQuickPrediction() {
  const text = document.getElementById('dashboardQuickText').value;
  const resultBox = document.getElementById('quickPredictionResult');

  if (!text.trim()) {
    alert('Please enter text to classify.');
    return;
  }

  resultBox.innerHTML = '<span class="spinner-border spinner-border-sm text-primary"></span> Classifying...';

  fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: text })
  })
    .then((res) => res.json())
    .then((data) => {
      if (data.status === 'success') {
        const rnn = data.predictions.rnn;
        const lstm = data.predictions.lstm;

        let rnnHtml = rnn.available
          ? `<span class="badge ${rnn.sentiment === 'Positive' ? 'bg-success' : 'bg-danger'}">${rnn.sentiment} (${rnn.confidence_percent}%)</span>`
          : '<span class="badge bg-secondary">Untrained</span>';

        let lstmHtml = lstm.available
          ? `<span class="badge ${lstm.sentiment === 'Positive' ? 'bg-success' : 'bg-danger'}">${lstm.sentiment} (${lstm.confidence_percent}%)</span>`
          : '<span class="badge bg-secondary">Untrained</span>';

        resultBox.innerHTML = `
        <div class="d-flex justify-content-around align-items-center">
          <div>
            <div class="small text-muted">Simple RNN</div>
            ${rnnHtml}
          </div>
          <div class="border-start ps-3">
            <div class="small text-muted">LSTM</div>
            ${lstmHtml}
          </div>
        </div>
      `;
      } else {
        resultBox.innerHTML = `<span class="text-danger small">${data.message || 'Prediction failed.'}</span>`;
      }
    })
    .catch((err) => {
      resultBox.innerHTML = `<span class="text-danger small">Error: ${err}</span>`;
    });
}
