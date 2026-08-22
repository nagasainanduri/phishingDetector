chrome.runtime.onInstalled.addListener(() => {
    console.log("PhishGuard MVP installed.");
});

// Listen for auto-scans from content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "active_scan") {
        fetch('http://127.0.0.1:5000/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: request.url, page_signals: request.dom_signals })
        })
        .then(response => response.json())
        .then(data => {
            if (data && data.length > 0) {
                const result = data[0];
                
                // Policy Engine decision enforcement
                if (result.action === 'BLOCK' || result.action === 'WARN') {
                    // Redirect the active tab to our native warning page
                    const warningUrl = chrome.runtime.getURL('warning/warning.html');
                    const params = new URLSearchParams({
                        url: request.url,
                        score: result.risk_score || 0,
                        sev: result.severity || 'UNKNOWN',
                        action: result.action,
                        reasons: encodeURIComponent(JSON.stringify(result.reasons || []))
                    });
                    
                    chrome.tabs.update(sender.tab.id, {
                        url: `${warningUrl}?${params.toString()}`
                    });
                }
            }
        })
        .catch(err => console.error("PhishGuard Active Scan Error:", err));
    }
});
