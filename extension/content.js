// extension/content.js

// Listens for messages from the popup to extract DOM signals
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "extract_dom_signals") {
        const signals = extractStructuralSignals();
        sendResponse({ dom_signals: signals });
    }
    return true;
});

// Auto-scan on load for the Policy Engine and browser warning system
window.addEventListener('load', () => {
    // Do not auto-scan if we intentionally bypassed the warning
    if (window.location.search.includes('phishguard_bypass=1')) {
        return;
    }
    
    // Don't auto scan extension pages or localhost (for development safety)
    if (window.location.protocol === 'chrome-extension:' || window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost') {
        return;
    }

    const signals = extractStructuralSignals();
    chrome.runtime.sendMessage({
        action: "active_scan",
        url: window.location.href,
        dom_signals: signals
    });
});

function extractStructuralSignals() {
    const signals = {
        has_password_field: false,
        has_login_form: false,
        cross_origin_form_action: false,
        hidden_iframes: false,
        num_forms: 0,
        num_iframes: 0,
        title: document.title
    };

    // Check for password inputs
    const passwordInputs = document.querySelectorAll('input[type="password"]');
    if (passwordInputs.length > 0) {
        signals.has_password_field = true;
    }

    // Analyze forms
    const forms = document.querySelectorAll('form');
    signals.num_forms = forms.length;
    
    forms.forEach(form => {
        // Check for cross-origin action
        const action = form.getAttribute('action');
        if (action && action.startsWith('http')) {
            try {
                const actionUrl = new URL(action);
                if (actionUrl.hostname !== window.location.hostname) {
                    signals.cross_origin_form_action = true;
                }
            } catch (e) {
                // Ignore invalid URLs
            }
        }
        
        // Check for login indicators based on structure/classes (not values)
        const formHtml = form.innerHTML.toLowerCase();
        if (formHtml.includes('login') || formHtml.includes('sign in') || formHtml.includes('signin')) {
            signals.has_login_form = true;
        }
    });

    // Analyze iframes
    const iframes = document.querySelectorAll('iframe');
    signals.num_iframes = iframes.length;
    
    iframes.forEach(iframe => {
        const style = window.getComputedStyle(iframe);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0' || iframe.width === '0' || iframe.height === '0') {
            signals.hidden_iframes = true;
        }
    });

    return signals;
}
