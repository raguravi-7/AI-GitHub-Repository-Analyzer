"""
utils.py
Small helper functions for turning raw commit/issue lists into
dataframes ready for charts in the Streamlit dashboard.
"""

import pandas as pd


def commits_to_dataframe(commits):
    df = pd.DataFrame(commits)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["day"] = df["date"].dt.date
    return df


def commit_trend(df, freq="D"):
    """Group commit counts by day/week for a trend chart."""
    if df.empty:
        return pd.DataFrame(columns=["period", "commits"])
    grouped = df.set_index("date").resample(freq).size().reset_index()
    grouped.columns = ["period", "commits"]
    return grouped


def contributor_activity(df):
    if df.empty:
        return pd.DataFrame(columns=["author", "commits"])
    counts = df["author"].value_counts().reset_index()
    counts.columns = ["author", "commits"]
    return counts


def issues_to_dataframe(issues):
    return pd.DataFrame(issues)
