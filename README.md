# Phishing Web Detector

A machine learning-based web application designed to detect phishing websites by analyzing URL features. This project aims to enhance cybersecurity by identifying malicious URLs that mimic legitimate websites to steal sensitive user information.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [CLI Usage](#cli-usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

Phishing attacks are a prevalent cybersecurity threat, where attackers create fraudulent websites that appear legitimate to deceive users into providing sensitive information such as login credentials or financial details. The Phishing Web Detector leverages machine learning techniques to classify URLs as either phishing or legitimate based on extracted features, helping users avoid malicious websites.

This repository contains the source code for a web application, a CLI tool, and supporting scripts for feature extraction and model training.

---

## Features

- **URL Analysis:** Extracts 17 features from URLs, including address bar, domain, and HTML/JS-based characteristics.
- **Machine Learning Models:** Uses a Random Forest classifier for accurate phishing detection.
- **Web Interface:** Clean, responsive UI built with Flask and Tailwind CSS.
- **CLI Tool:** Command-line interface for single URL or batch file predictions.
- **Real-Time Prediction:** Immediate feedback via web or CLI.
- **Extensible Design:** Modular codebase for easy feature/model integration.

---

## Technologies Used

- **Python:** Core programming language.
- **Scikit-learn:** Machine learning model implementation.
- **Flask:** Web framework for the UI.
- **Pandas & NumPy:** Data manipulation and analysis.
- **BeautifulSoup & tldextract:** Feature extraction from URLs and HTML.
- **Tailwind CSS:** Styling for the web interface.
- **Git:** Version control.

---

## Installation

1. **Clone the Repository:**
    ```sh
    git clone https://github.com/nagasainanduri/phishingwebdetector.git
    cd phishingwebdetector
    ```

2. **Install Python Dependencies:**  
   Ensure Python 3.6+ is installed. Install packages using:
    ```sh
    pip install -r requirements.txt
    ```

3. **Download the Dataset:**  
   Download phishing URLs from PhishTank and legitimate URLs from UNB CIC. Place them in the `data/` directory as `phishing_urls.csv` and `legitimate_urls.csv`. Preprocess with:
    ```sh
    python scripts/fetch_data.py
    ```

4. **Train the Model:**  
    ```sh
    python scripts/train_model.py
    ```

5. **Run the Web Application:**  
    ```sh
    python app.py
    ```
    Visit [http://localhost:5000](http://localhost:5000).

---

## Usage

### Web Interface

1. Navigate to [http://localhost:5000](http://localhost:5000).
2. Enter a URL and click "Analyze URL".
3. View the result (Phishing/Legitimate) and confidence score.

### API Usage

Send a POST request to `/api/predict`:
```sh
curl -X POST -H "Content-Type: application/json" -d '{"url":"http://example.com"}' http://localhost:5000/api/predict
```

---

## CLI Usage

Use the CLI for single or batch URL predictions:

- **Single URL:**
    ```sh
    python cli.py --url http://example.com
    ```

- **Batch File (URLs in `urls.txt`, one per line):**
    ```sh
    python cli.py --file urls.txt
    ```

---

## Dataset

- **Phishing URLs:** From kaggele 
- **Legitimate URLs:** From UNB CIC/Kaggele or similar sources.

**Features Extracted (17 total):**
- **Address Bar:** IP presence, URL length, HTTPS, dots, slashes, @ symbol, dash, query presence.
- **Domain:** Domain length, subdomain presence, TLD length, domain age, DNS record.
- **HTML/JS:** Number of anchors, external anchors, forms, pop-up presence, meta refresh tags.

Data is preprocessed into `data/processed_data.csv`.

---

## Model Training

The project uses a Random Forest classifier for binary classification:

- **Feature Extraction:** Handled by `scripts/feature_extraction.py`.
- **Training:** 80/20 train-test split, with accuracy, precision, and recall metrics.
- **Serialization:** Model saved as `models/phishing_detector.pkl`.

To retrain, run:
```sh
python scripts/train_model.py
```

---

## Contributing

1. Fork the repository.
2. Create a branch:
    ```sh
    git checkout -b feature-branch
    ```
3. Commit changes:
    ```sh
    git commit -m "Add feature"
    ```
4. Push:
    ```sh
    git push origin feature-branch
    ```
5. Open a Pull Request.

Ensure code follows PEP 8 and includes tests.

---

## License

Apache 2.0 License. See [LICENSE](LICENSE) for details.

---

## Contact

**Author:** Nagasai Nanduri  
**GitHub Issues:** Open an issue for bug reports or feature requests.

**Disclaimer:**  
This tool is for educational and research purposes only. It is not foolproof and should not be the sole method for ensuring online safety. Verify URLs through trusted sources.

This tool is currently under development and may not work as intended.