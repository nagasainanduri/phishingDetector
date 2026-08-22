# PhishGuard

**Version:** 1.0.0-rc.1  
**Status:** Release Candidate

## 1. Project Overview

PhishGuard is a privacy-first, machine learning-backed web security and browser extension platform designed to detect and warn against potentially malicious and phishing websites.

It evaluates URLs in real time by combining machine learning predictions with URL canonicalization, handcrafted heuristics, brand impersonation detection, optional threat intelligence, and experimental page-level signals.

Rather than relying on a single detection mechanism, PhishGuard follows a **Defense-in-Depth** architecture designed to improve detection robustness while providing users with understandable explanations for security decisions.

This repository contains:

- The Flask backend API (`app.py`)
- The Chrome extension (`extension/`)
- The core detection and risk engine (`detector/`)
- Model training and evaluation pipelines (`scripts/`)
- Automated tests (`tests/`)
- CI/CD workflows (`.github/workflows/`)

---

## 2. Architecture

PhishGuard operates on a strict **Defense-in-Depth** architecture that separates URL canonicalization, detection, risk assessment, and policy decisions.

```text
                         PHISHGUARD
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
         Chrome Extension    API            CLI
              |              |              |
              +--------------+--------------+
                             |
                             v
                    Input Validation
                             |
                             v
                 URL Canonicalization
                             |
             +---------------+---------------+
             |                               |
             v                               v
      Raw Representation            Canonical Representation
             |                               |
             +---------------+---------------+
                             |
                             v
                     Feature Extraction
                             |
             +---------------+---------------+
             |               |               |
             v               v               v
            ML          Heuristics       Page Analysis
             |               |               |
             +---------------+---------------+
                             |
                             v
                    Brand Detection
                             |
                             v
                  Threat Intelligence
                             |
                             v
                    DetectionResult
                             |
                             v
                       Risk Engine
                             |
                +------------+------------+
                |                         |
                v                         v
           Risk Score                Confidence
                |
                v
                     Policy Engine
                             |
                 +-----------+-----------+
                 |           |           |
                 v           v           v
               ALLOW        WARN       BLOCK
```

The architecture deliberately separates:

1. **Detection** — what signals were observed.
2. **Risk Assessment** — how those signals combine into a risk score.
3. **Policy** — what action should be taken.

This allows individual detection components to evolve independently without coupling them directly to browser blocking decisions.

---

## 3. Features

### Implemented Functionality

- **Real-time URL Canonicalization**
  - Bounded recursive decoding
  - IDNA/Punycode normalization
  - Unicode/confusable character analysis
  - Path normalization
  - Hexadecimal IP normalization
  - Integer IP normalization
  - Preservation of raw and canonical URL representations
  - Configurable maximum decoding depth

- **Privacy Controls**
  - Local-only detection mode
  - Optional external threat intelligence
  - Telemetry disabled by default
  - User-controlled feedback submission

- **Model Explainability (XAI)**
  - SHAP-based feature attribution
  - Human-readable explanations of model contributions
  - Feature-level evidence can be surfaced to users
  - Designed to improve transparency rather than treating the ML model as an opaque authority

- **Brand Impersonation Detection**
  - Fuzzy matching
  - Levenshtein-distance-based similarity
  - Detection against commonly targeted brands

- **User Feedback Pipeline**
  - False-positive feedback
  - False-negative feedback
  - Optional URL privacy protection through hashing

- **Risk & Policy Engine**
  - Combines ML predictions and non-ML security signals
  - Produces a normalized 0–100 risk score
  - Maps risk levels to policy actions such as ALLOW, WARN, or BLOCK

- **Hardened API**
  - Versioned `/api/v1/` endpoints
  - Configurable request/resource limits
  - Request validation
  - Explicit error handling
  - Production-oriented WSGI deployment support

- **CI/CD Integration**
  - Automated unit and integration testing
  - Dependency checks
  - Chrome extension manifest validation
  - GitHub Actions integration

### Experimental Functionality

- **DOM Signal Processing**
  - The Chrome extension can extract page-level signals such as forms and iframes.
  - These signals are currently treated as experimental and are not yet part of the primary ML feature set.
  - Future versions may incorporate DOM signals into the detection and risk engines.

### Optional Functionality

- **Threat Intelligence**
  - PhishTank integration
  - VirusTotal integration
  - Requires corresponding API keys
  - Disabled when operating in Local Only mode

- **Telemetry**
  - Disabled by default
  - Can be explicitly enabled by the operator
  - Intended for improving detection and system monitoring

---

## 4. Detection Pipeline

The pipeline runs synchronously for each URL.

### Step 1 — Input Validation

Requests are validated against configurable limits before expensive processing begins.

Validation includes:

- Request body size
- URL length
- Batch size
- Number of URLs per request
- Input structure

### Step 2 — URL Canonicalization

The raw URL is normalized into a canonical representation while preserving the original representation for security analysis.

Canonicalization includes:

- Percent-decoding
- Bounded recursive decoding
- IDNA/Punycode normalization
- Hexadecimal IP normalization
- Integer IP normalization
- Path normalization
- Suspicious encoding detection

Recursive decoding is deliberately bounded to prevent uncontrolled resource consumption.

The system retains:

- Raw URL
- Canonical URL
- Normalization transformations
- Encoding depth
- Suspicious transformation indicators

Canonicalization does not make external network requests.

### Step 3 — Feature Extraction

The canonicalized URL is used to extract the 17 static URL features currently used by the ML model.

Raw URL information may also be retained where it represents useful security evidence.

### Step 4 — Heuristics & Brand Analysis

Handcrafted detection rules are applied alongside the ML model.

These include signals such as:

- Suspicious URL structure
- Domain characteristics
- Encoding indicators
- IP-based URLs
- Excessive subdomains
- Brand similarity
- Other URL-level anomalies

Generic characteristics such as uncommon or newer TLDs are treated only as **weak contextual signals**.

No TLD is considered malicious solely because of its TLD.

### Step 5 — ML Inference

The current production model is a **Random Forest Classifier** trained on the 17 extracted URL features.

The model produces an **uncalibrated model probability**.

This probability is not automatically treated as the final probability that a website is malicious.

### Step 6 — Threat Intelligence

When permitted by privacy settings and configured by the operator, PhishGuard may query external threat-intelligence providers such as:

- PhishTank
- VirusTotal

External intelligence is optional and is disabled when operating in Local Only mode.

### Step 7 — Risk Engine

The Risk Engine aggregates:

- ML output
- Heuristic findings
- Brand similarity
- Canonicalization findings
- Threat intelligence
- Other available security signals

It produces a final risk score in the range:

```text
0–100
```

The risk score is a **decision-oriented composite score**, not automatically a probability.

### Step 8 — Policy Engine

The final risk assessment is mapped to a policy action:

```text
ALLOW
WARN
BLOCK
```

Detection and policy decisions remain separate so that future policy changes do not require redesigning the underlying detection mechanisms.

---

## 5. ML Model

The current primary ML model is a:

**Random Forest Classifier**

trained on 17 extracted URL features.

The model relies on `scikit-learn` to maintain a relatively small deployment footprint and low inference latency.

The model is integrated with SHAP for feature-level attribution.

### Why Random Forest?

Multiple candidate approaches were evaluated during development.

The final model selection was based on practical deployment criteria rather than model complexity alone, including:

- Predictive performance
- Precision
- Recall
- Inference latency
- Model footprint
- Feature-level interpretability
- Deployment complexity

Random Forest currently provides the best practical balance for the current browser-extension deployment target among the models evaluated.

A CNN-based approach also demonstrated useful performance during experimentation and remains a potential future research direction.

The project does not claim that Random Forest is universally superior to other ML architectures.

---

## 6. Model Explainability & User Transparency

PhishGuard integrates SHAP-based explainability to provide feature-level insight into ML predictions.

The objective is not simply to tell the user:

```text
"This website is malicious."
```

Instead, PhishGuard aims to provide understandable evidence such as:

```text
Why was this URL considered risky?

- High domain entropy
- Suspicious URL encoding
- Unusual subdomain structure
- High brand similarity
- Suspicious IP representation
```

Where supported, SHAP feature attribution can identify which model features contributed most strongly to the prediction.

### Important distinction

Model explanations describe **feature contributions to a prediction**.

They should not be interpreted as proof of causality.

Similarly:

```text
risk_score
```

is not equivalent to:

```text
probability of phishing
```

The system currently uses uncalibrated ML probabilities and therefore does not claim that the numerical model probability represents an exact statistical probability of maliciousness.

The purpose of explainability is to improve:

- User transparency
- Analyst understanding
- Debugging
- Model evaluation
- Security decision transparency

---

## 7. Model Benchmark Results

These metrics are based on the project's local evaluation dataset using an 80/20 evaluation split.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 91.04% |
| **Precision** | 87.99% |
| **Recall** | 94.60% |
| **Inference Latency** | ~0.008 ms |
| **Model Size** | 16.2 MB |

These values represent the current evaluated Random Forest implementation and should not be interpreted as universal real-world performance.

Future evaluations should additionally measure:

- F1 score
- False-positive rate
- False-negative rate
- ROC-AUC / PR-AUC where appropriate
- Cross-validation performance
- Dataset drift
- Performance on unseen/adversarial domains
- Inference performance under realistic extension workloads

---

## 8. Adversarial Evaluation

An adversarial evaluation framework (`scripts/evaluate_adversarial.py`) was developed to test evasive and deceptive URLs.

The canonicalization engine successfully mitigates several previously identified evasion techniques.

| Threat Category | Status |
| :--- | :--- |
| Typosquatting (`g00gle.com`) | ✅ Detected |
| Punycode/Homoglyphs | ✅ Detected |
| Hex-Encoded IPs | ✅ Detected |
| Subdomain Abuse | ✅ Detected |
| Brand Impersonation | ✅ Detected |
| URL Encoding Evasion | ✅ Detected |
| Recursive Encoding Evasion | ✅ Mitigated through bounded decoding |
| Misleading Paths | ❌ Current False Negative |
| Open Redirects | ❌ Current False Negative |

The two known false-negative categories remain documented intentionally.

They represent areas for future research rather than hidden limitations.

---

## 9. Chrome Extension Installation

1. Open Chrome and navigate to:

```text
chrome://extensions/
```

2. Enable **Developer mode**.

3. Click **Load unpacked**.

4. Select the `extension/` directory from this repository.

5. Click the PhishGuard icon.

6. Open the **Options** page.

7. Configure the backend URL, for example:

```text
http://127.0.0.1:5000
```

The extension is currently intended primarily for development and personal deployment.

Future releases may introduce more robust deployment and identity mechanisms for public or multi-user installations.

---

## 10. API Usage

### Endpoint

```text
POST /api/v1/analyze
```

### Headers

```text
Content-Type: application/json
```

### Example Body

```json
{
  "url": "http://example-login-update.com",
  "privacy_mode": "local_only",
  "telemetry": false
}
```

### Privacy Mode

The `local_only` mode prevents the backend from contacting optional third-party threat-intelligence providers.

---

## 11. CLI Usage

Predictions can also be performed directly from the terminal without starting the API.

### Single URL

```bash
python scripts/cli.py --url http://example.com
```

### Batch file

```bash
python scripts/cli.py --file urls.txt
```

---

## 12. Configuration & Deployment

Copy the example environment configuration:

```bash
cp .env.example .env
```

Configure resource limits, model settings, privacy options, and optional threat-intelligence API keys as required.

For deployment, use a production WSGI server such as Gunicorn rather than Flask's development server.

A `gunicorn.conf.py` configuration is provided in the repository.

Example:

```bash
gunicorn -c gunicorn.conf.py app:app
```

### Production-Oriented Deployment

For public deployments, PhishGuard should be placed behind an appropriate reverse proxy or load balancer.

Recommended architecture:

```text
Internet
   |
   v
Reverse Proxy / Load Balancer
   |
   +-- TLS termination
   +-- Rate limiting / access controls
   |
   v
Gunicorn
   |
   v
Flask API
   |
   v
PhishGuard Detection Engine
```

The Flask development server should not be used as the public-facing production server.

---

## 13. Privacy Model

PhishGuard defaults to:

**Privacy-Preserving Local Only Mode**

### Local Only

When Local Only mode is enabled:

- Third-party threat-intelligence APIs are not queried.
- Browsing URLs are not sent to external intelligence providers.
- Detection continues using locally available detection mechanisms.

### Telemetry

Telemetry is disabled by default.

If telemetry is explicitly enabled, the operator should clearly understand what information is being recorded and configure deployment policies accordingly.

Telemetry should remain data-minimized and should not be enabled silently.

### Feedback

When submitting feedback, users can choose whether to provide the raw URL.

Where raw URLs are withheld, the system can store a SHA-256 hash instead.

---

## 14. Threat Intelligence Integrations

PhishGuard optionally integrates with:

- PhishTank
- VirusTotal

To enable the integrations, configure:

```text
PHISHTANK_API_KEY
VIRUSTOTAL_API_KEY
```

Threat intelligence is optional.

Local Only mode prevents external threat-intelligence requests.

External provider failures should not cause the core detection engine to fail.

---

## 15. API Security & Resource Protection

The current release focuses on the security of the detection pipeline and resource handling.

Configurable resource limits include:

```text
PHISHGUARD_MAX_REQUEST_BYTES
PHISHGUARD_MAX_URL_LENGTH
PHISHGUARD_MAX_BATCH_SIZE
PHISHGUARD_MAX_URLS_PER_REQUEST
PHISHGUARD_MAX_DECODE_DEPTH
```

These limits help prevent:

- Oversized request abuse
- Excessively long URL processing
- Excessive batch requests
- Unbounded decoding
- Resource-exhaustion attacks

Expected input errors are handled gracefully.

Unexpected internal exceptions are not silently converted into successful responses and should remain observable through appropriate server-side errors.

### Authentication

Dedicated API authentication and user/API-key management are considered future production-scale functionality.

The current release is primarily intended for:

- Local deployment
- Personal use
- Development
- Controlled browser-extension deployments
- Portfolio and research demonstration

Public multi-user deployments should introduce stronger authentication and access-control mechanisms before being exposed to untrusted clients.

---

## 16. Testing & CI/CD

The repository uses `pytest` for unit and integration testing.

GitHub Actions automatically performs:

- Automated testing
- Dependency checks
- Extension manifest validation

Run the test suite with:

```bash
pytest tests/ -v
```

Current test suite:

```text
51 tests
```

The CI pipeline is intended to prevent regressions across the backend, detection engine, and Chrome extension.

---

## 17. Screenshots / Demo

Screenshots and demonstration material will be added in a future documentation update.

Planned demonstration material:

- Chrome extension popup
- Risk assessment
- Explainability output
- Detection warning
- Local Only mode
- CLI output
- Backend API response
- Adversarial test examples

---

## 18. Known Limitations

### 18.1 Model Stagnation

The Random Forest model is currently trained on a static dataset.

Without continuous dataset expansion and validated retraining, detection performance may degrade as phishing techniques evolve.

### 18.2 Open Redirects

Open redirects on otherwise trusted domains can still produce false negatives because the root domain may appear trustworthy while redirect parameters point toward malicious destinations.

This is a known limitation of the current URL-focused detection architecture.

Future work should investigate redirect-aware analysis while avoiding unsafe automatic URL fetching or SSRF vulnerabilities.

### 18.3 Misleading Paths

Certain misleading path structures can still evade the current feature and heuristic combination.

This remains an adversarial evaluation target for future iterations.

### 18.4 Uncalibrated Probabilities

The Random Forest probability output is currently uncalibrated.

Therefore:

```text
model_probability != guaranteed probability of maliciousness
```

The final risk score is a separate composite decision score.

Future versions may evaluate probability calibration techniques such as:

- Platt scaling
- Isotonic regression

Calibration should only be introduced after evaluation on appropriate validation data.

### 18.5 Static URL Feature Limitations

The primary ML model currently relies on static URL features.

Some attacks cannot be reliably detected from URL structure alone.

Examples include:

- Legitimate-looking URLs hosting malicious content
- Dynamic client-side attacks
- Highly convincing cloned websites
- Malicious content delivered after navigation
- Complex redirect chains
- DOM-based deception

These limitations motivate future page-level and DOM-based detection.

---

## 19. Future Work

### 1. Continuous Feedback and Retraining

Automate the validated feedback pipeline to support periodic model retraining.

Potential workflow:

```text
User Feedback
      |
      v
Validation
      |
      v
Dataset Expansion
      |
      v
Model Retraining
      |
      v
Benchmarking
      |
      v
Model Approval
      |
      v
Deployment
```

Retraining should not automatically promote an unvalidated model to production.

### 2. Dataset Expansion

Expand the dataset with:

- Newly observed phishing domains
- Newly registered domains
- Diverse benign domains
- Adversarial examples
- Internationalized domains
- Additional URL obfuscation techniques

Dataset quality and class balance should be monitored continuously.

### 3. Additional ML Research

Continue benchmarking alternative architectures, including the previously evaluated CNN approach and other suitable models.

Evaluation should consider:

- Accuracy
- Precision
- Recall
- F1
- False-positive rate
- False-negative rate
- Inference latency
- Model size
- Explainability
- Deployment complexity

A more complex model should only replace the current Random Forest implementation if it demonstrates a meaningful practical advantage.

### 4. DOM-Based Machine Learning

Expand the experimental DOM signal pipeline into a more complete page-analysis system.

Potential future signals include:

- Forms
- Iframes
- Login elements
- External resource relationships
- Brand/logo similarity
- Page structure
- Suspicious JavaScript behaviors

Computer-vision or multimodal approaches may be investigated for website/logo impersonation detection.

### 5. Redirect-Aware Analysis

Develop safe redirect analysis capable of identifying:

- Open redirects
- Redirect parameters
- Nested destination URLs
- Suspicious destination domains

This must be implemented carefully to avoid introducing:

- SSRF
- Internal network access
- Resource exhaustion
- Unsafe automatic URL fetching

### 6. Stronger Production Authentication

For future public or multi-user deployments, introduce:

- API authentication
- API-key management
- User identity
- Per-client quotas
- Access control
- Abuse prevention
- Deployment-level authorization

Authentication is intentionally not required for the current local/personal extension deployment.

### 7. Probability Calibration

Evaluate calibrated ML probabilities so that model outputs can be interpreted more reliably.

Calibration should be measured independently from the final composite risk score.

### 8. Improved Extension UX

Improve the browser experience by exposing security reasoning in a concise and understandable format.

Example:

```text
HIGH RISK

Risk Score: 87/100

Why?

• Domain resembles a known brand
• Suspicious URL encoding detected
• Unusual subdomain structure
• High domain similarity
```

The goal is to provide users with actionable security information rather than simply displaying a binary phishing verdict.

---

## 20. Release Status

PhishGuard is currently a:

**1.0.0 Release Candidate**

The current implementation provides a complete end-to-end prototype consisting of:

- URL canonicalization
- Static feature extraction
- Random Forest ML inference
- SHAP-based explainability
- Heuristic detection
- Brand impersonation detection
- Risk scoring
- Policy decisions
- Optional threat intelligence
- Privacy controls
- User feedback
- Chrome extension integration
- CLI support
- Automated testing
- CI/CD

The project remains under active development.

Known adversarial limitations and future production-scale requirements are intentionally documented rather than hidden.

---

## Disclaimer

PhishGuard is an experimental security tool intended for research, education, development, and controlled deployment.

It provides risk assessments and suspected phishing classifications but does not guarantee detection of all malicious websites or novel attacks.

A high risk score should be interpreted as a security warning rather than absolute proof of malicious intent.

Similarly, the current ML probability output is uncalibrated and must not be interpreted as an exact statistical probability of maliciousness.

Users should continue to apply normal security practices and should not rely on PhishGuard as their sole security control.
