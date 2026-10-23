/**
 * Chart.js Visualizations Module
 * SentimentAI - RNN vs LSTM Analyzer
 */

const chartInstances = {};

function destroyChart(id) {
  if (chartInstances[id]) {
    chartInstances[id].destroy();
    delete chartInstances[id];
  }
}

// 1. Dashboard History Charts
function renderDashboardHistoryCharts(rnnHist, lstmHist) {
  const accCanvas = document.getElementById('chartDashboardAccuracy');
  const lossCanvas = document.getElementById('chartDashboardLoss');

  if (!accCanvas || !lossCanvas) return;

  const epochs = Math.max(
    rnnHist ? rnnHist.epochs_run : 0,
    lstmHist ? lstmHist.epochs_run : 0,
    5
  );
  const labels = Array.from({ length: epochs }, (_, i) => `Epoch ${i + 1}`);

  // Accuracy Chart
  destroyChart('dashAcc');
  chartInstances['dashAcc'] = new Chart(accCanvas, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'RNN Val Accuracy',
          data: rnnHist ? rnnHist.val_accuracy : [],
          borderColor: '#0284c7',
          backgroundColor: 'rgba(2, 132, 199, 0.1)',
          tension: 0.3,
          fill: true
        },
        {
          label: 'LSTM Val Accuracy',
          data: lstmHist ? lstmHist.val_accuracy : [],
          borderColor: '#7c3aed',
          backgroundColor: 'rgba(124, 58, 237, 0.1)',
          tension: 0.3,
          fill: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0.5, max: 1.0, title: { display: true, text: 'Accuracy' } }
      }
    }
  });

  // Loss Chart
  destroyChart('dashLoss');
  chartInstances['dashLoss'] = new Chart(lossCanvas, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'RNN Val Loss',
          data: rnnHist ? rnnHist.val_loss : [],
          borderColor: '#0284c7',
          borderDash: [5, 5],
          tension: 0.3
        },
        {
          label: 'LSTM Val Loss',
          data: lstmHist ? lstmHist.val_loss : [],
          borderColor: '#7c3aed',
          borderDash: [5, 5],
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { title: { display: true, text: 'Binary Crossentropy Loss' } }
      }
    }
  });
}

// 2. Dataset Distribution Chart
function renderDatasetDistributionChart(train, val, test) {
  const canvas = document.getElementById('chartDatasetDist');
  if (!canvas) return;

  destroyChart('dsDist');
  chartInstances['dsDist'] = new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels: ['Train (20k)', 'Validation (5k)', 'Test (25k)'],
      datasets: [
        {
          data: [train, val, test],
          backgroundColor: ['#2563eb', '#f59e0b', '#7c3aed'],
          borderWidth: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom' }
      }
    }
  });
}

// 3. Single Model (RNN or LSTM) History Charts
function renderModelHistoryCharts(prefix, history) {
  const accCanvas = document.getElementById(`chart${prefix}Accuracy`);
  const lossCanvas = document.getElementById(`chart${prefix}Loss`);

  if (!accCanvas || !lossCanvas || !history) return;

  const labels = Array.from({ length: history.epochs_run }, (_, i) => `Epoch ${i + 1}`);
  const color = prefix === 'Rnn' ? '#0284c7' : '#7c3aed';

  destroyChart(`${prefix}Acc`);
  chartInstances[`${prefix}Acc`] = new Chart(accCanvas, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Training Accuracy',
          data: history.accuracy,
          borderColor: color,
          backgroundColor: 'transparent',
          borderDash: [3, 3],
          tension: 0.2
        },
        {
          label: 'Validation Accuracy',
          data: history.val_accuracy,
          borderColor: color,
          backgroundColor: `${color}15`,
          fill: true,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0.5, max: 1.0, title: { display: true, text: 'Accuracy' } }
      }
    }
  });

  destroyChart(`${prefix}Loss`);
  chartInstances[`${prefix}Loss`] = new Chart(lossCanvas, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Training Loss',
          data: history.loss,
          borderColor: '#ef4444',
          borderDash: [3, 3],
          tension: 0.2
        },
        {
          label: 'Validation Loss',
          data: history.val_loss,
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          fill: true,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { title: { display: true, text: 'Loss' } }
      }
    }
  });
}

// 4. Comparison Bar Chart
function renderComparisonBarChart(rnnEval, lstmEval) {
  const canvas = document.getElementById('chartMetricsComparison');
  if (!canvas) return;

  const metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'];
  const rnnData = rnnEval
    ? [rnnEval.accuracy, rnnEval.precision, rnnEval.recall, rnnEval.f1_score, rnnEval.roc_auc]
    : [0, 0, 0, 0, 0];
  const lstmData = lstmEval
    ? [lstmEval.accuracy, lstmEval.precision, lstmEval.recall, lstmEval.f1_score, lstmEval.roc_auc]
    : [0, 0, 0, 0, 0];

  destroyChart('cmpBar');
  chartInstances['cmpBar'] = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: metrics,
      datasets: [
        {
          label: 'Simple RNN',
          data: rnnData,
          backgroundColor: '#0284c7'
        },
        {
          label: 'LSTM',
          data: lstmData,
          backgroundColor: '#7c3aed'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0.5, max: 1.0, title: { display: true, text: 'Score' } }
      }
    }
  });
}

// 5. Sequence Length Scaling Line Chart
function renderSequenceScalingChart(scalingList) {
  const canvas = document.getElementById('chartSeqLengthPerformance');
  if (!canvas || !scalingList) return;

  const labels = scalingList.map((s) => `${s.seq_len} Tokens`);
  const rnnAcc = scalingList.map((s) => s.rnn_acc);
  const lstmAcc = scalingList.map((s) => s.lstm_acc);

  destroyChart('seqScale');
  chartInstances['seqScale'] = new Chart(canvas, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Simple RNN Accuracy',
          data: rnnAcc,
          borderColor: '#0284c7',
          backgroundColor: 'rgba(2, 132, 199, 0.1)',
          tension: 0.3,
          fill: true
        },
        {
          label: 'LSTM Accuracy',
          data: lstmAcc,
          borderColor: '#7c3aed',
          backgroundColor: 'rgba(124, 58, 237, 0.1)',
          tension: 0.3,
          fill: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0.7, max: 0.95, title: { display: true, text: 'Accuracy' } }
      }
    }
  });
}

// 6. Combined ROC Curves Chart
function renderCombinedRocChart(rnnData, lstmData) {
  const canvas = document.getElementById('chartCombinedRoc');
  if (!canvas) return;

  const datasets = [
    {
      label: 'Random Baseline (AUC = 0.50)',
      data: [
        { x: 0, y: 0 },
        { x: 1, y: 1 }
      ],
      borderColor: '#94a3b8',
      borderDash: [5, 5],
      pointRadius: 0
    }
  ];

  if (rnnData && rnnData.roc_curve) {
    datasets.push({
      label: `Simple RNN (AUC = ${rnnData.roc_auc})`,
      data: rnnData.roc_curve.map((p) => ({ x: p.fpr, y: p.tpr })),
      borderColor: '#0284c7',
      tension: 0.2,
      pointRadius: 2
    });
  }

  if (lstmData && lstmData.roc_curve) {
    datasets.push({
      label: `LSTM (AUC = ${lstmData.roc_auc})`,
      data: lstmData.roc_curve.map((p) => ({ x: p.fpr, y: p.tpr })),
      borderColor: '#7c3aed',
      tension: 0.2,
      pointRadius: 2
    });
  }

  destroyChart('combRoc');
  chartInstances['combRoc'] = new Chart(canvas, {
    type: 'line',
    data: { datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'False Positive Rate (FPR)' } },
        y: { min: 0, max: 1, title: { display: true, text: 'True Positive Rate (TPR)' } }
      }
    }
  });
}

// 7. Distance Scan Degradation Chart
function renderDistanceScanChart(scanList) {
  const canvas = document.getElementById('chartDistanceScan');
  if (!canvas || !scanList) return;

  const labels = scanList.map((s) => `${s.token_distance} words`);
  const rnnProbs = scanList.map((s) => s.rnn_prob);
  const lstmProbs = scanList.map((s) => s.lstm_prob);

  destroyChart('distScan');
  chartInstances['distScan'] = new Chart(canvas, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Simple RNN Prob (Climbing = Vanishing Negation)',
          data: rnnProbs,
          borderColor: '#ef4444',
          tension: 0.3,
          pointRadius: 4
        },
        {
          label: 'LSTM Prob (Low = Preserved Negation)',
          data: lstmProbs,
          borderColor: '#10b981',
          tension: 0.3,
          pointRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0.0, max: 1.0, title: { display: true, text: 'Predicted Sigmoid Probability' } }
      }
    }
  });
}
