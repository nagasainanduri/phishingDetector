// Default settings
const DEFAULT_SETTINGS = {
    backend_url: 'http://127.0.0.1:5000',
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
        // Update status to let user know options were saved.
        const status = document.getElementById('status-msg');
        status.textContent = 'Options saved.';
        status.style.display = 'block';
        setTimeout(() => {
            status.textContent = '';
            status.style.display = 'none';
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
