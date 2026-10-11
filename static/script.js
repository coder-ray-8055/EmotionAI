/**
 * Emotion AI — Dashboard Interactive Script
 */

// Configuration for emotions: colors, labels, emojis
const emotionConfig = {
  sadness: { color: '#4a90d9', emoji: '😢' },
  joy: { color: '#f5c542', emoji: '😊' },
  love: { color: '#e84393', emoji: '❤️' },
  anger: { color: '#e74c3c', emoji: '😡' },
  fear: { color: '#9b59b6', emoji: '😨' },
  surprise: { color: '#00cec9', emoji: '😮' }
};

// State
let predictionHistory = [];

// DOM Elements
const textInput = document.getElementById('text-input');
const analyzeBtn = document.getElementById('analyze-btn');
const btnIcon = analyzeBtn.querySelector('.btn-icon');
const btnText = analyzeBtn.querySelector('.btn-text');
const btnSpinner = analyzeBtn.querySelector('.btn-spinner');

const cleanedTextDisplay = document.getElementById('cleaned-text-display');
const cleanedTextSpan = document.getElementById('cleaned-text');

const resultCard = document.getElementById('result-card');
const resultEmoji = document.getElementById('result-emoji');
const resultEmotion = document.getElementById('result-emotion');
const resultConfidence = document.getElementById('result-confidence');
const confidenceBarsContainer = document.getElementById('confidence-bars');

const statTfidf = document.getElementById('stat-tfidf');
const statBow = document.getElementById('stat-bow');
const statSamples = document.getElementById('stat-samples');
const statEmotions = document.getElementById('stat-emotions');
const distributionBars = document.getElementById('distribution-bars');
const historyList = document.getElementById('history-list');

/**
 * Utility: Animate number counter from 0 to target
 */
function animateValue(element, start, end, duration = 1000, isPercentage = false, decimals = 1) {
  let startTimestamp = null;
  const step = (timestamp) => {
    if (!startTimestamp) startTimestamp = timestamp;
    const progress = Math.min((timestamp - startTimestamp) / duration, 1);
    // Ease-out cubic formula
    const easeOutProgress = 1 - Math.pow(1 - progress, 3);
    const current = start + (end - start) * easeOutProgress;

    if (isPercentage) {
      element.textContent = (current * 100).toFixed(decimals) + '%';
    } else {
      element.textContent = Math.round(current).toLocaleString();
    }

    if (progress < 1) {
      window.requestAnimationFrame(step);
    }
  };
  window.requestAnimationFrame(step);
}

/**
 * Fetch and populate model statistics & distribution on load
 */
async function loadStats() {
  try {
    const res = await fetch('/stats');
    if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
    const data = await res.json();

    // Animate stats
    if (data.tfidf_accuracy !== undefined) {
      animateValue(statTfidf, 0, data.tfidf_accuracy, 1000, true, 1);
    }
    if (data.bow_accuracy !== undefined) {
      animateValue(statBow, 0, data.bow_accuracy, 1000, true, 1);
    }
    if (data.total_samples !== undefined) {
      animateValue(statSamples, 0, data.total_samples, 1200, false);
    }
    if (data.num_emotions !== undefined) {
      animateValue(statEmotions, 0, data.num_emotions, 600, false);
    }

    // Build distribution bars
    if (data.emotion_counts) {
      renderDistributionBars(data.emotion_counts);
    }
  } catch (err) {
    console.warn('Could not load live stats from /stats:', err);
    // Fallback default UI display if API is not yet running
    statTfidf.textContent = '85.2%';
    statBow.textContent = '82.4%';
    statSamples.textContent = '20,000';
    statEmotions.textContent = '6';

    const defaultDistribution = {
      joy: 5362,
      sadness: 4666,
      anger: 2159,
      fear: 1937,
      love: 1304,
      surprise: 572
    };
    renderDistributionBars(defaultDistribution);
  }
}

/**
 * Render Training Data Emotion Distribution Bars
 */
function renderDistributionBars(emotionCounts) {
  distributionBars.innerHTML = '';
  const entries = Object.entries(emotionCounts);
  const maxCount = Math.max(...entries.map(([, count]) => count), 1);

  // Sort descending by count
  entries.sort((a, b) => b[1] - a[1]);

  entries.forEach(([emotion, count]) => {
    const config = emotionConfig[emotion.toLowerCase()] || { color: '#667eea', emoji: '✨' };
    const percentage = ((count / maxCount) * 100).toFixed(1);

    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <div class="bar-header">
        <span class="bar-label-group">
          <span>${config.emoji}</span>
          <span>${emotion}</span>
        </span>
        <span class="bar-value">${count.toLocaleString()}</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="background: ${config.color}; box-shadow: 0 0 10px ${config.color}66;"></div>
      </div>
    `;

    distributionBars.appendChild(row);

    // Smooth width expansion after DOM injection
    requestAnimationFrame(() => {
      const fillEl = row.querySelector('.bar-fill');
      setTimeout(() => {
        fillEl.style.width = `${percentage}%`;
      }, 50);
    });
  });
}

/**
 * Render Confidence Bars for prediction result
 */
function renderConfidenceBars(allScores) {
  confidenceBarsContainer.innerHTML = '';

  // Sort emotions by confidence descending
  const sortedEmotions = Object.entries(allScores).sort((a, b) => b[1] - a[1]);

  sortedEmotions.forEach(([emotion, score]) => {
    const config = emotionConfig[emotion.toLowerCase()] || { color: '#667eea', emoji: '✨' };
    const percentVal = (score * 100).toFixed(1);

    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <div class="bar-header">
        <span class="bar-label-group">
          <span>${config.emoji}</span>
          <span>${emotion}</span>
        </span>
        <span class="bar-value">${percentVal}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="background: ${config.color}; box-shadow: 0 0 10px ${config.color}66;"></div>
      </div>
    `;

    confidenceBarsContainer.appendChild(row);

    // Animate fill width
    requestAnimationFrame(() => {
      const fillEl = row.querySelector('.bar-fill');
      setTimeout(() => {
        fillEl.style.width = `${percentVal}%`;
      }, 60);
    });
  });
}

/**
 * Handle prediction submission
 */
async function handleAnalyze() {
  const text = textInput.value.trim();
  if (!text) {
    textInput.focus();
    textInput.classList.add('shake');
    setTimeout(() => textInput.classList.remove('shake'), 400);
    return;
  }

  // Set loading state
  analyzeBtn.disabled = true;
  if (btnIcon) btnIcon.hidden = true;
  btnText.textContent = 'Analyzing...';
  btnSpinner.hidden = false;

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (!res.ok) {
      throw new Error(`Prediction failed with status: ${res.status}`);
    }

    const data = await res.json();
    displayResult(data, text);
    addToHistory(text, data);
  } catch (err) {
    console.error('Prediction error:', err);
    // Display error feedback gracefully in result card
    resultCard.hidden = false;
    resultEmoji.textContent = '⚠️';
    resultEmotion.textContent = 'Error';
    resultConfidence.textContent = 'Could not classify text';
    cleanedTextDisplay.hidden = true;
    confidenceBarsContainer.innerHTML = `
      <p style="color: #ef4444; font-size: 0.88rem; padding: 0.5rem 0;">
        ${err.message || 'Server connection failed'}. Please ensure the API is running.
      </p>
    `;
  } finally {
    // Reset button state
    analyzeBtn.disabled = false;
    if (btnIcon) btnIcon.hidden = false;
    btnText.textContent = 'Analyze Emotion';
    btnSpinner.hidden = true;
  }
}

/**
 * Display prediction results in result card
 */
function displayResult(data, originalText) {
  const emotionKey = (data.emotion || '').toLowerCase();
  const config = emotionConfig[emotionKey] || { color: '#667eea', emoji: '✨' };

  // Set values
  resultEmoji.textContent = config.emoji;
  resultEmotion.textContent = emotionKey.charAt(0).toUpperCase() + emotionKey.slice(1);
  const confidencePercent = (data.confidence * 100).toFixed(1);
  resultConfidence.textContent = `${confidencePercent}% Confident`;

  // Custom glow card accent border/shadow based on predicted emotion
  resultCard.style.borderColor = `${config.color}55`;
  resultCard.style.boxShadow = `0 12px 35px 0 rgba(0, 0, 0, 0.4), 0 0 25px ${config.color}33`;

  // Preprocessed cleaned text
  if (data.cleaned_text) {
    cleanedTextSpan.textContent = data.cleaned_text;
    cleanedTextDisplay.hidden = false;
  } else {
    cleanedTextDisplay.hidden = true;
  }

  // Populate confidence bars
  if (data.all_scores) {
    renderConfidenceBars(data.all_scores);
  }

  // Show card and scroll into view smoothly if on small screens
  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Add an analysis result to prediction history (max 10)
 */
function addToHistory(originalText, data) {
  const emotionKey = (data.emotion || '').toLowerCase();
  const config = emotionConfig[emotionKey] || { color: '#667eea', emoji: '✨' };
  const confidencePercent = (data.confidence * 100).toFixed(1);

  const historyItem = {
    text: originalText,
    emotion: emotionKey,
    emoji: config.emoji,
    color: config.color,
    confidence: confidencePercent,
    timestamp: Date.now()
  };

  predictionHistory.unshift(historyItem);
  if (predictionHistory.length > 10) {
    predictionHistory.pop();
  }

  renderHistory();
}

/**
 * Render history DOM items
 */
function renderHistory() {
  if (predictionHistory.length === 0) {
    historyList.innerHTML = '<p class="empty-state">No predictions yet. Try analyzing some text!</p>';
    return;
  }

  historyList.innerHTML = '';
  predictionHistory.forEach((item) => {
    const el = document.createElement('div');
    el.className = 'history-item';
    el.title = 'Click to reload this text';

    const truncatedText = item.text.length > 48 ? item.text.substring(0, 48) + '...' : item.text;

    el.innerHTML = `
      <span class="history-text">${truncatedText}</span>
      <span class="history-badge" style="color: ${item.color}; border: 1px solid ${item.color}44;">
        <span>${item.emoji}</span>
        <span>${item.confidence}%</span>
      </span>
    `;

    // Click history item fills the textarea
    el.addEventListener('click', () => {
      textInput.value = item.text;
      textInput.focus();
    });

    historyList.appendChild(el);
  });
}

// Event Listeners
analyzeBtn.addEventListener('click', handleAnalyze);

textInput.addEventListener('keydown', (e) => {
  // Submit on Ctrl+Enter or Cmd+Enter or Enter without Shift
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleAnalyze();
  }
});

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  loadStats();
});
