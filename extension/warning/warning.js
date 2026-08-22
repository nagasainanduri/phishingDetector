document.addEventListener('DOMContentLoaded', () => {
    const params = new URLSearchParams(window.location.search);
    const targetUrl = params.get('url');
    
    // Safety check, if no URL just go to a blank page or google
    if (!targetUrl) {
        window.location.href = 'https://www.google.com';
        return;
    }
    
    // Extract parameters
    const score = parseInt(params.get('score')) || 0;
    const severity = params.get('sev') || 'UNKNOWN';
    const action = params.get('action') || 'BLOCK';
    const reasonsRaw = params.get('reasons');
    const modelExpRaw = params.get('model_explanation');
    const modelLimitation = params.get('explanation_limitation');
    
    // Update UI elements
    document.getElementById('url-display').textContent = targetUrl;
    document.getElementById('severity-badge').textContent = severity;
    document.getElementById('action-label').textContent = `Policy Action: ${action}`;
    
    // Update Score Ring
    const scoreCircle = document.getElementById('score-circle');
    const scoreText = document.getElementById('score-text');
    scoreText.textContent = score;
    
    // Give a slight delay for the animation to kick in
    setTimeout(() => {
        scoreCircle.setAttribute('stroke-dasharray', `${score}, 100`);
    }, 100);

    // Update specific warning titles based on Threat Intel or heuristics
    const warningTitle = document.getElementById('warning-title');
    if (action === 'BLOCK' && score >= 90) {
        warningTitle.textContent = "Confirmed Deceptive Site Ahead";
    } else if (action === 'WARN') {
        warningTitle.textContent = "Suspected Deceptive Site Ahead";
        document.body.style.backgroundColor = "#e67c73"; // slightly lighter orange/red for warning
    }
    
    // Parse and display reasons
    if (reasonsRaw) {
        try {
            const reasons = JSON.parse(decodeURIComponent(reasonsRaw));
            const list = document.getElementById('reasons-list');
            reasons.forEach(r => {
                const li = document.createElement('li');
                li.textContent = r;
                
                // Highlight Threat Intel or Brand Impersonation
                if (r.includes("MALICIOUS by Threat Intelligence") || r.includes("Brand Impersonation")) {
                    li.style.fontWeight = "bold";
                }
                
                list.appendChild(li);
            });
        } catch (e) {
            console.error("Failed to parse reasons:", e);
        }
    }
    
    // Parse and display model explanations
    const expContainer = document.getElementById('model-explanation-container');
    const expList = document.getElementById('explanation-list');
    const expLimitation = document.getElementById('explanation-limitation');
    
    if (modelLimitation && modelLimitation !== "null") {
        expLimitation.textContent = modelLimitation;
        expLimitation.classList.remove('hidden');
        expContainer.classList.remove('hidden');
    } else if (modelExpRaw) {
        try {
            const modelExp = JSON.parse(decodeURIComponent(modelExpRaw));
            if (modelExp && modelExp.length > 0) {
                modelExp.forEach(r => {
                    const li = document.createElement('li');
                    li.textContent = r;
                    expList.appendChild(li);
                });
                expContainer.classList.remove('hidden');
            }
        } catch (e) {
            console.error("Failed to parse model explanations:", e);
        }
    }

    // Button event listeners
    document.getElementById('btn-back').addEventListener('click', () => {
        // Go back in history if possible, else go to a safe page
        if (window.history.length > 1) {
            window.history.back();
        } else {
            window.location.href = 'https://www.google.com';
        }
    });

    document.getElementById('btn-continue').addEventListener('click', () => {
        // To bypass the block, we can send a message to background to allowlist this URL temporarily,
        // but for now we simply navigate to it. 
        // We append a bypass flag so background.js doesn't immediately redirect us back!
        const bypassUrl = new URL(targetUrl);
        bypassUrl.searchParams.append('phishguard_bypass', '1');
        window.location.href = bypassUrl.href;
    });

    // Feedback event listeners
    const btnCorrect = document.getElementById('btn-feedback-correct');
    const btnFP = document.getElementById('btn-feedback-fp');
    
    const submitFeedback = (feedbackType) => {
        let shareRawUrl = false;
        if (feedbackType === 'false_positive') {
            shareRawUrl = confirm("To help improve PhishGuard, would you like to share the raw URL for our dataset? (If you click Cancel, we will only log an anonymized hash for statistical tracking).");
        }
        
        chrome.storage.sync.get({ backend_url: 'http://127.0.0.1:5000' }, (settings) => {
            const apiUrl = `${settings.backend_url.replace(/\/$/, '')}/api/v1/feedback`;
            const headers = { 'Content-Type': 'application/json' };
            
            fetch(apiUrl, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({
                    url: targetUrl,
                    feedback_type: feedbackType,
                    share_raw_url: shareRawUrl,
                    risk_score: score,
                    prediction: severity
                })
            }).then(() => {
                const status = document.getElementById('feedback-status');
                status.textContent = 'Feedback submitted! Thank you.';
                status.classList.remove('hidden');
                btnCorrect.disabled = true;
                btnFP.disabled = true;
            }).catch(e => console.error("Feedback error", e));
        });
    };
    
    if (btnCorrect) btnCorrect.onclick = () => submitFeedback('correct');
    if (btnFP) btnFP.onclick = () => submitFeedback('false_positive');
});
