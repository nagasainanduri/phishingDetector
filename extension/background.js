chrome.runtime.onInstalled.addListener(() => {
    console.log("PhishGuard MVP installed.");
});

// Listen for auto-scans from content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "active_scan") {
        console.log(`Performing active scan on: ${request.url}`);
        
        chrome.storage.sync.get({
            backend_url: 'http://127.0.0.1:5000',
            privacy_mode: 'local_only',
            telemetry: false
        }, (settings) => {
            const apiUrl = `${settings.backend_url.replace(/\/$/, '')}/api/v1/analyze`;
            const headers = { 'Content-Type': 'application/json' };
            
            fetch(apiUrl, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({
                    url: request.url,
                    page_signals: request.dom_signals,
                    privacy_mode: settings.privacy_mode,
                    telemetry: settings.telemetry
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data && data.length > 0) {
                    const result = data[0];
                    if (result.action === 'WARN' || result.action === 'BLOCK') {
                        // Redirect to warning page
                        const warningUrl = chrome.runtime.getURL('warning/warning.html');
                        const params = new URLSearchParams({
                            url: request.url,
                            score: result.risk_score || 0,
                            sev: result.severity || 'UNKNOWN',
                            action: result.action,
                            reasons: encodeURIComponent(JSON.stringify(result.reasons || [])),
                            model_explanation: encodeURIComponent(JSON.stringify(result.model_explanation || [])),
                            explanation_limitation: result.explanation_limitation || "null"
                        });
                        
                        chrome.tabs.update(sender.tab.id, {
                            url: `${warningUrl}?${params.toString()}`
                        });
                    }
                }
            })
            .catch(error => {
                console.error('PhishGuard Error:', error);
            });
        });
        
        return true;
    }
});
