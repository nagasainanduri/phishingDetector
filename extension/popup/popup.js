document.addEventListener('DOMContentLoaded', async () => {
    const urlDisplay = document.getElementById('url-display');
    const scanBtn = document.getElementById('scan-btn');
    const statusContainer = document.getElementById('status-container');
    const errorContainer = document.getElementById('error-container');
    const reasonsContainer = document.getElementById('reasons-container');
    const reasonsList = document.getElementById('reasons-list');
    const scoreCircle = document.getElementById('score-circle');
    const scoreText = document.getElementById('score-text');
    const severityBadge = document.getElementById('severity-badge');

    let currentUrl = '';

    // Get current active tab
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tab && tab.url) {
            currentUrl = tab.url;
            urlDisplay.textContent = currentUrl;
        } else {
            urlDisplay.textContent = 'Unable to read tab URL';
            scanBtn.disabled = true;
        }
    } catch (e) {
        urlDisplay.textContent = 'Error accessing tab';
        scanBtn.disabled = true;
    }

    scanBtn.addEventListener('click', async () => {
        if (!currentUrl) return;

        // Reset UI
        errorContainer.classList.add('hidden');
        statusContainer.classList.add('hidden');
        reasonsContainer.classList.add('hidden');
        scanBtn.textContent = 'Scanning...';
        scanBtn.disabled = true;

        try {
            // Attempt to get DOM signals from the active tab
            let domSignals = null;
            try {
                const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
                const response = await chrome.tabs.sendMessage(tab.id, { action: "extract_dom_signals" });
                if (response && response.dom_signals) {
                    domSignals = response.dom_signals;
                }
            } catch (err) {
                console.warn("Could not get DOM signals from tab (content script might not be loaded on this page):", err);
            }

            chrome.storage.sync.get({
                backend_url: 'http://127.0.0.1:5000',
                privacy_mode: 'local_only',
                telemetry: false
            }, async (settings) => {
                try {
                    const apiUrl = `${settings.backend_url.replace(/\/$/, '')}/api/predict`;
                    const response = await fetch(apiUrl, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ 
                            url: currentUrl, 
                            page_signals: domSignals,
                            privacy_mode: settings.privacy_mode,
                            telemetry: settings.telemetry
                        })
                    });

                    if (!response.ok) {
                        throw new Error(`Server returned ${response.status}`);
                    }

                    const data = await response.json();
                    
                    if (data.length > 0) {
                        const result = data[0];
                        if (result.error) {
                            showError(result.error);
                        } else {
                            showResult(result);
                        }
                    } else {
                        showError("No data received from API");
                    }
                } catch (err) {
                    showError(`Detection API unreachable: ${err.message}. Is the local Flask server running?`);
                } finally {
                    scanBtn.textContent = 'Re-Scan Tab';
                    scanBtn.disabled = false;
                }
            });
        } catch (err) {
            // Error caught from DOM extraction
        }
    });

    function showError(msg) {
        errorContainer.textContent = msg;
        errorContainer.classList.remove('hidden');
    }

    function showResult(result) {
        statusContainer.classList.remove('hidden');
        
        // Update Risk Score Ring
        const score = result.risk_score || 0;
        scoreText.textContent = score;
        scoreCircle.setAttribute('stroke-dasharray', `${score}, 100`);
        
        // Update Severity Badge
        const sev = result.severity || 'UNKNOWN';
        severityBadge.textContent = sev;
        severityBadge.className = 'severity-badge'; // reset
        
        if (sev === 'CRITICAL') {
            severityBadge.classList.add('sev-critical');
            scoreCircle.style.stroke = 'var(--sev-critical)';
        } else if (sev === 'HIGH') {
            severityBadge.classList.add('sev-high');
            scoreCircle.style.stroke = 'var(--sev-high)';
        } else if (sev === 'MEDIUM') {
            severityBadge.classList.add('sev-medium');
            scoreCircle.style.stroke = 'var(--sev-medium)';
        } else if (sev === 'LOW') {
            severityBadge.classList.add('sev-low');
            scoreCircle.style.stroke = 'var(--sev-low)';
        } else {
            severityBadge.classList.add('sev-unknown');
            scoreCircle.style.stroke = 'var(--text-secondary)';
        }

        // Update Reasons (System Evidence)
        if (result.reasons && result.reasons.length > 0) {
            reasonsList.innerHTML = '';
            result.reasons.forEach(reason => {
                const li = document.createElement('li');
                li.textContent = reason;
                reasonsList.appendChild(li);
            });
            reasonsContainer.classList.remove('hidden');
        } else {
            reasonsContainer.classList.add('hidden');
        }
        
        // Update Model Explanation (XAI)
        const expContainer = document.getElementById('model-explanation-container');
        const expList = document.getElementById('explanation-list');
        const expLimitation = document.getElementById('explanation-limitation');
        
        expList.innerHTML = '';
        expLimitation.classList.add('hidden');
        
        if (result.explanation_limitation) {
            expLimitation.textContent = result.explanation_limitation;
            expLimitation.classList.remove('hidden');
            expContainer.classList.remove('hidden');
        } else if (result.model_explanation && result.model_explanation.length > 0) {
            result.model_explanation.forEach(exp => {
                const li = document.createElement('li');
                li.textContent = exp;
                expList.appendChild(li);
            });
            expContainer.classList.remove('hidden');
        } else {
            expContainer.classList.add('hidden');
        }
        
        // Show feedback container
        const feedbackContainer = document.getElementById('feedback-container');
        feedbackContainer.classList.remove('hidden');
        
        // Wire up feedback buttons
        const btnCorrect = document.getElementById('btn-feedback-correct');
        const btnFP = document.getElementById('btn-feedback-fp');
        const btnFN = document.getElementById('btn-feedback-fn');
        
        const submitFeedback = (feedbackType) => {
            let shareRawUrl = false;
            if (feedbackType === 'false_positive' || feedbackType === 'false_negative') {
                shareRawUrl = confirm("To help improve PhishGuard, would you like to share the raw URL for our dataset? (If you click Cancel, we will only log an anonymized hash for statistical tracking).");
            }
            
            chrome.storage.sync.get({ backend_url: 'http://127.0.0.1:5000' }, (settings) => {
                const apiUrl = `${settings.backend_url.replace(/\/$/, '')}/api/feedback`;
                fetch(apiUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        url: currentUrl,
                        feedback_type: feedbackType,
                        share_raw_url: shareRawUrl,
                        risk_score: result.risk_score,
                        prediction: result.result
                    })
                }).then(() => {
                    const status = document.getElementById('feedback-status');
                    status.textContent = 'Feedback submitted!';
                    status.classList.remove('hidden');
                    btnCorrect.disabled = true;
                    btnFP.disabled = true;
                    btnFN.disabled = true;
                }).catch(e => console.error("Feedback error", e));
            });
        };
        
        btnCorrect.onclick = () => submitFeedback('correct');
        btnFP.onclick = () => submitFeedback('false_positive');
        btnFN.onclick = () => submitFeedback('false_negative');
    }
});
