# Continuous Integration and Continuous Deployment (CI/CD)

PhishGuard employs GitHub Actions to ensure code quality, dependency security, and core logic stability. The CI pipeline is explicitly designed to be fast and hardware-agnostic.

## Workflows Overview

1. **Main CI Pipeline (`.github/workflows/ci.yml`)**
   - **Triggers**: On push and PR to the `main` branch.
   - **Linting**: Uses `flake8` to enforce Python syntax and catch undefined names or complexity issues.
   - **Security**: Uses `pip-audit` to detect known CVE vulnerabilities in third-party dependencies (`requirements.txt`).
   - **Testing**: Runs the entire `pytest` suite against the Python backend.

2. **Extension Validation (`.github/workflows/extension.yml`)**
   - **Triggers**: On modifications to the `extension/` directory.
   - **Checks**: Verifies the syntax of `manifest.json` and ensures that critical background and popup scripts exist before allowing merges.

## What is Excluded from CI?

> [!IMPORTANT]
> **GPU and Heavy Workloads**
> To avoid expensive runner costs and long queue times, **model training scripts and SHAP benchmark generation are intentionally excluded from automated CI**.
> 
> These ML workflows (e.g. `scripts/train_model.py` and `scripts/evaluate_adversarial.py`) should be executed manually or on dedicated GPU hardware before a new model artifact is pushed to the repository.

## Running Tests Locally

Before submitting a Pull Request, you should run the CI checks locally on your machine.

**1. Install Development Dependencies:**
```bash
pip install flake8 pip-audit pytest
```

**2. Run Linters:**
```bash
flake8 app.py detector/ tests/ scripts/ --count --select=E9,F63,F7,F82 --show-source --statistics
```

**3. Run Security Audits:**
```bash
pip-audit -r requirements.txt
```

**4. Run All Unit & Integration Tests:**
```bash
pytest tests/ -v
```

If all tests pass locally, they will likely pass in the GitHub Actions runner.
