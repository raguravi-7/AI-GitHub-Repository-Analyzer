# AI GitHub Repository Analyzer

An ML-powered dashboard that analyzes any public GitHub repository and gives
insights into repo health, issue types, and bug-prone files.

## What it does

| Module | Technique | Output |
|---|---|---|
| Issue Classification | TF-IDF + Logistic Regression (trained live on the repo's own issues) | Categorizes issues as Bug / Feature / Question / Docs / Other |
| Bug-Prone File Prediction | Logistic Regression on commit-derived features (touch frequency, author diversity) | Risk score per file — which files are most likely to contain bugs |
| Commit Trend Analysis | Time-series aggregation | Commit activity over time |
| Contributor Activity | Frequency counts | Top contributors by commit volume |

## Project Structure

```
github_analyzer/
├── app.py         # Streamlit UI - run this file
├── fetcher.py      # GitHub REST API wrapper (issues, commits, files)
├── ml_models.py    # IssueClassifier + BugProneFilePredictor (the ML core)
├── utils.py        # Dataframe helpers for charts
└── requirements.txt
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional but recommended) Create a GitHub Personal Access Token:
   - GitHub → Settings → Developer settings → Personal access tokens → Generate new token
   - No special scopes needed for public repos
   - Without a token: 60 API requests/hour. With a token: 5000/hour.

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. In the sidebar, enter a repo as `owner/repo` (e.g. `psf/requests`, `pallets/flask`)
   and click **Analyze Repository**.

## How the ML actually works (for your project report / viva)

**Issue Classifier**
- Uses GitHub's own issue labels as weak/ground-truth signals when present.
- When labels are missing or too sparse, falls back to keyword-based
  pseudo-labeling to bootstrap a training set (this is a common technique
  called *weak supervision*).
- Trains a fresh TF-IDF + Logistic Regression model **per repository, on the fly**
  — meaning the model adapts to each repo's own vocabulary instead of using a
  generic pretrained classifier.
- Reports train/test accuracy in the UI.

**Bug-Prone File Predictor**
- Parses every analyzed commit's message for bug-fix keywords (fix, bug, patch,
  error, resolve, hotfix, crash) to derive a bug-fix label.
- Builds a per-file feature table: total commit touches, distinct author count.
- Trains a Logistic Regression model where the label is "was this file touched
  in a bug-fix commit at least once" — then outputs a probability score
  (0 to 1) per file as its "bug-proneness risk".

## Notes for your presentation

- This is a genuine end-to-end ML pipeline: raw data → feature engineering →
  model training → prediction → visualization, not just an API wrapper.
- Rate limits: GitHub API is public but limited. For the demo, pick a small-to-medium
  repo (not something huge like `torvalds/linux`) so the 50-100 commit analysis
  finishes quickly.
- Good demo repos: `psf/requests`, `pallets/flask`, `tiangolo/fastapi`.
