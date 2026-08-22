# PhishGuard Privacy Policy

we believe privacy is a fundamental human right. Our phishing detection engine is designed from the ground up to protect you from malicious websites without compromising your personal data. 

This Privacy Policy explicitly details what data is processed, how it is used, and guarantees that **we collect absolutely no data unless you explicitly opt-in.**

---

## 1. Zero Data Collection by Default

When you install the PhishGuard extension, it operates in a **strictly localized, privacy-first mode**. 
- **No Telemetry:** We do not track your browsing history.
- **No Background Scanning:** We do not actively scan the websites you visit in the background.
- **No Data Harvesting:** We do not collect analytics, telemetry, or user identification metrics.

By default, the extension will only scan a URL when you *manually* click the PhishGuard extension icon and request a scan.

---

## 2. Opt-In Features and Data Usage

PhishGuard offers advanced security features that require external data processing. **These features are completely disabled by default** and require your explicit, manual opt-in via the Settings page.

If you choose to enable these features, here is exactly what data is used and how:

### A. Active Background Scanning
*Default: OFF*
If you opt-in to Active Scanning, PhishGuard will automatically scan the URLs of the websites you visit to provide real-time warnings before you interact with a malicious page. 
- **Data Used:** The URL of the current tab and structural HTML signals (e.g., the presence of a password field). 
- **Data Storage:** This data is sent to the PhishGuard backend solely for instantaneous risk analysis. **It is not saved, logged, or tracked** unless you also enable Telemetry.

### B. Online Threat Intelligence
*Default: OFF*
If you opt-in to Online Intelligence, PhishGuard will cross-reference scanned URLs against third-party global threat databases (like Google Safe Browsing or URLhaus) to improve detection accuracy.
- **Data Used:** The URL being scanned.
- **Data Storage:** The URL is securely transmitted to third-party APIs for reputation checks. We do not control the data retention policies of these third parties.

### C. Telemetry & Dataset Contribution
*Default: OFF*
If you opt-in to Telemetry, you agree to help us improve the PhishGuard Machine Learning models by contributing detection results to our training datasets.
- **Data Used:** The URL scanned, the AI model's prediction, and the calculated risk score.
- **Data Minimization:** Even when opted-in, PhishGuard utilizes data minimization techniques. If you submit feedback on a URL, we will only log an **anonymized, irreversible cryptographic hash** (SHA-256) of the URL, unless you explicitly click "Yes, Share URL" to share the raw string.

---

## 3. Local Storage

PhishGuard uses your browser's local storage (`chrome.storage.sync`) exclusively to save your configuration preferences (e.g., whether you have opted into telemetry, and your configured backend API URL). This data remains on your device and synced across your browser profiles. It is never transmitted to us.

---

## 4. Your Rights and Control

You are in total control of your data. You can revoke your consent for Active Scanning, Online Intelligence, or Telemetry at any time by simply toggling them off in the extension's Settings menu. The changes take effect immediately.

If you have any questions or concerns regarding this policy or the PhishGuard open-source codebase, please refer to our repository documentation or open an issue.
