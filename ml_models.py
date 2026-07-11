"""
ml_models.py
Two ML components:
1. IssueClassifier      - classifies issue text into Bug / Feature / Question / Docs / Other
                           using TF-IDF + Logistic Regression.
2. BugProneFilePredictor - scores each file's risk of being bug-prone using a
                           Logistic Regression model trained on commit-derived features.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 1. Issue Classification
# ---------------------------------------------------------------------------

LABEL_KEYWORDS = {
    "bug": ["bug", "error", "issue", "fix", "crash", "fail", "broken", "defect"],
    "feature": ["feature", "add", "request", "enhancement", "new", "support"],
    "question": ["question", "how to", "help", "clarify", "?"],
    "docs": ["doc", "documentation", "readme", "typo"],
}


def _weak_label_from_text(title, body, github_labels):
    """
    Generate a pseudo-label using GitHub's own labels first (if informative),
    otherwise fall back to keyword matching on the issue text.
    This bootstraps training data when the repo has no consistent labeling.
    """
    labels_lower = " ".join(github_labels).lower()
    for cat, keywords in LABEL_KEYWORDS.items():
        if any(k in labels_lower for k in keywords):
            return cat

    text = f"{title} {body}".lower()
    scores = {cat: sum(text.count(k) for k in kws) for cat, kws in LABEL_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


class IssueClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=500)
        self.is_trained = False
        self.classes_ = None

    def prepare_training_data(self, issues):
        texts, labels = [], []
        for issue in issues:
            text = f"{issue['title']} {issue['body']}".strip()
            if not text:
                continue
            label = _weak_label_from_text(issue["title"], issue["body"], issue["labels"])
            texts.append(text)
            labels.append(label)
        return texts, labels

    def fit(self, issues):
        texts, labels = self.prepare_training_data(issues)
        if len(set(labels)) < 2 or len(texts) < 8:
            # Not enough variety/data to train a real model -> use rule-based fallback
            self.is_trained = False
            return

        X = self.vectorizer.fit_transform(texts)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.2, random_state=42, stratify=labels
            )
        except ValueError:
            X_train, y_train = X, labels
            X_test, y_test = X, labels

        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.classes_ = self.model.classes_
        self.train_accuracy = self.model.score(X_train, y_train)
        self.test_accuracy = self.model.score(X_test, y_test)

    def predict(self, issues):
        results = []
        for issue in issues:
            text = f"{issue['title']} {issue['body']}".strip()
            if self.is_trained and text:
                vec = self.vectorizer.transform([text])
                pred = self.model.predict(vec)[0]
                conf = float(np.max(self.model.predict_proba(vec)))
            else:
                pred = _weak_label_from_text(issue["title"], issue["body"], issue["labels"])
                conf = 0.5  # rule-based fallback, no real confidence score
            results.append({**issue, "predicted_category": pred, "confidence": round(conf, 2)})
        return results


# ---------------------------------------------------------------------------
# 2. Bug-Prone File Prediction
# ---------------------------------------------------------------------------

BUGFIX_PATTERN = re.compile(r"\b(fix|bug|patch|error|resolve|hotfix|crash)\b", re.IGNORECASE)


def build_file_feature_table(commits):
    """
    Build a per-file feature table from enriched commit data
    (each commit must have 'files', 'author', 'date', 'message').
    """
    rows = {}
    for c in commits:
        is_bugfix = bool(BUGFIX_PATTERN.search(c.get("message", "")))
        for f in c.get("files", []):
            row = rows.setdefault(
                f,
                {"file": f, "total_touches": 0, "bugfix_touches": 0, "authors": set(), "dates": []},
            )
            row["total_touches"] += 1
            if is_bugfix:
                row["bugfix_touches"] += 1
            row["authors"].add(c.get("author", "unknown"))
            if c.get("date"):
                row["dates"].append(c["date"])

    records = []
    for f, r in rows.items():
        dates = sorted(r["dates"])
        records.append(
            {
                "file": f,
                "total_touches": r["total_touches"],
                "bugfix_touches": r["bugfix_touches"],
                "distinct_authors": len(r["authors"]),
                "first_touch": dates[0] if dates else None,
                "last_touch": dates[-1] if dates else None,
            }
        )
    return pd.DataFrame(records)


class BugProneFilePredictor:
    """
    Trains a Logistic Regression model to estimate the probability that a file
    is 'bug-prone', using total_touches and distinct_authors as features
    (bugfix_touches itself is used only to derive the training label, not as
    an input feature, so the model has to generalize instead of memorizing).
    """

    def __init__(self):
        self.model = LogisticRegression(max_iter=500)
        self.is_trained = False

    def fit_predict(self, df):
        if df.empty or len(df) < 5:
            df["risk_score"] = 0.0
            return df

        df = df.copy()
        df["label"] = (df["bugfix_touches"] > 0).astype(int)

        X = df[["total_touches", "distinct_authors"]].values
        y = df["label"].values

        if len(set(y)) < 2:
            # every file identical outcome -> fall back to normalized heuristic score
            max_touch = df["total_touches"].max() or 1
            df["risk_score"] = (df["total_touches"] / max_touch).round(3)
            self.is_trained = False
            return df.sort_values("risk_score", ascending=False)

        self.model.fit(X, y)
        self.is_trained = True
        probs = self.model.predict_proba(X)[:, 1]
        df["risk_score"] = probs.round(3)
        return df.sort_values("risk_score", ascending=False)
