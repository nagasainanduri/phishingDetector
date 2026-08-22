// Default settings
const DEFAULT_SETTINGS = {
    backend_url: CONFIG.BACKEND_URL,
    privacy_mode: 'local_only',
    active_scanning: false,
    telemetry: false
};

// Saves options to chrome.storage
function saveOptions() {
    const backendUrl = document.getElementById('backend-url').value || DEFAULT_SETTINGS.backend_url;
    const privacyMode = document.getElementById('privacy-mode').value;
    const activeScanning = document.getElementById('active-scanning').checked;
    const telemetry = document.getElementById('telemetry').checked;

    chrome.storage.sync.set({
        backend_url: backendUrl,
        privacy_mode: privacyMode,
        active_scanning: activeScanning,
        telemetry: telemetry
    }, () => {
        const toast = document.getElementById('toast');
        toast.classList.add('show');
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    });
}

// Restores select box and checkbox state using the preferences
// stored in chrome.storage.
function restoreOptions() {
    chrome.storage.sync.get(DEFAULT_SETTINGS, (items) => {
        document.getElementById('backend-url').value = items.backend_url;
        document.getElementById('privacy-mode').value = items.privacy_mode;
        document.getElementById('active-scanning').checked = items.active_scanning;
        document.getElementById('telemetry').checked = items.telemetry;
    });
}

document.addEventListener('DOMContentLoaded', restoreOptions);
document.getElementById('save-btn').addEventListener('click', saveOptions);

// Add Custom Modal Logic
const showCustomConfirm = (title, message) => {
    return new Promise((resolve) => {
        const modal = document.getElementById('custom-confirm-modal');
        const titleEl = document.getElementById('modal-title');
        const textEl = document.getElementById('modal-text');
        const btnYes = document.getElementById('modal-btn-yes');
        const btnNo = document.getElementById('modal-btn-no');
        
        titleEl.textContent = title;
        textEl.textContent = message;
        
        modal.classList.remove('hidden');
        
        const cleanup = () => {
            modal.classList.add('hidden');
            btnYes.removeEventListener('click', onYes);
            btnNo.removeEventListener('click', onNo);
        };
        
        const onYes = () => { cleanup(); resolve(true); };
        const onNo = () => { cleanup(); resolve(false); };
        
        btnYes.addEventListener('click', onYes);
        btnNo.addEventListener('click', onNo);
    });
};

document.getElementById('active-scanning').addEventListener('change', async (e) => {
    if (e.target.checked) {
        e.target.checked = false; // Revert immediately
        const confirmed = await showCustomConfirm(
            "Enable Active Scanning?",
            "This will automatically scan every page you visit. The extension will read the content of all pages you navigate to in order to detect phishing. Do you agree?"
        );
        if (confirmed) e.target.checked = true;
    }
});

document.getElementById('telemetry').addEventListener('change', async (e) => {
    if (e.target.checked) {
        e.target.checked = false; // Revert immediately
        const confirmed = await showCustomConfirm(
            "Enable Telemetry?",
            "Allow the backend to anonymously store queried URLs to improve the detection model over time. No personally identifiable information is stored. Do you agree?"
        );
        if (confirmed) e.target.checked = true;
    }
});

document.getElementById('privacy-mode').addEventListener('change', async (e) => {
    if (e.target.value === 'local_online') {
        e.target.value = 'local_only'; // Revert immediately
        const confirmed = await showCustomConfirm(
            "Enable Online Intelligence?",
            "This will allow the backend to query 3rd-party intelligence providers (like Google Safe Browsing) by sharing the URLs you visit. Do you agree?"
        );
        if (confirmed) e.target.value = 'local_online';
    }
});
