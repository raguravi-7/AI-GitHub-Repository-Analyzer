

import streamlit as st
import plotly.express as px

from fetcher import GitHubFetcher
from ml_models import IssueClassifier, build_file_feature_table, BugProneFilePredictor
from utils import commits_to_dataframe, commit_trend, contributor_activity, issues_to_dataframe

st.set_page_config(page_title="AI GitHub Repository Analyzer", layout="wide")

st.title("🔍 AI GitHub Repository Analyzer")
st.caption("ML-powered insights into repo health, issue types, and bug-prone files")

# ---------------------------------------------------------------------------
# Sidebar - inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Repository")
    repo_url = st.text_input("Owner/Repo", placeholder="e.g. pallets/flask")
    token = st.text_input("GitHub Token (optional, raises rate limit)", type="password")
    max_commits = st.slider("Commits to analyze in depth", 20, 150, 60, step=10)
    run_btn = st.button("🚀 Analyze Repository", type="primary")

    st.markdown("---")
    st.markdown(
        "**Note:** Without a token, GitHub allows 60 API requests/hour. "
        "A token raises this to 5000/hour — create one at "
        "GitHub → Settings → Developer settings → Personal access tokens."
    )

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
if run_btn:
    if "/" not in repo_url:
        st.error("Enter repo as owner/repo, e.g. facebook/react")
        st.stop()

    owner, repo = repo_url.strip().split("/", 1)

    try:
        with st.spinner("Connecting to GitHub..."):
            fetcher = GitHubFetcher(owner, repo, token=token or None)
            info = fetcher.get_repo_info()

        with st.spinner("Fetching issues..."):
            issues = fetcher.get_issues(max_pages=4, per_page=50)

        with st.spinner("Fetching commits..."):
            commits = fetcher.get_commits(max_pages=3, per_page=50)

        with st.spinner(f"Analyzing changed files in {min(max_commits, len(commits))} commits..."):
            commits = fetcher.enrich_commits_with_files(commits, limit=max_commits)

    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.stop()

    # --- Repo overview ---
    st.subheader(f"📦 {info.get('full_name', repo_url)}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⭐ Stars", info.get("stargazers_count", 0))
    c2.metric("🍴 Forks", info.get("forks_count", 0))
    c3.metric("🐛 Open Issues", info.get("open_issues_count", 0))
    c4.metric("💬 Language", info.get("language", "N/A"))

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Commit Trends", "🏷️ Issue Classification (ML)", "⚠️ Bug-Prone Files (ML)", "👥 Contributors"]
    )

    # --- Tab 1: Commit trend ---
    with tab1:
        cdf = commits_to_dataframe(commits)
        if cdf.empty:
            st.info("No commit data available.")
        else:
            trend = commit_trend(cdf, freq="D")
            fig = px.line(trend, x="period", y="commits", title="Commit Activity Over Time")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(cdf[["sha", "author", "date", "message"]].head(20), use_container_width=True)

    # --- Tab 2: Issue classification ---
    with tab2:
        if not issues:
            st.info("No issues found in this repository.")
        else:
            st.markdown("**Model:** TF-IDF + Logistic Regression, trained on this repo's own issue text/labels.")
            clf = IssueClassifier()
            clf.fit(issues)
            classified = clf.predict(issues)
            idf = issues_to_dataframe(classified)

            if clf.is_trained:
                st.success(
                    f"Model trained on repo data — train accuracy: {clf.train_accuracy:.2f}, "
                    f"test accuracy: {clf.test_accuracy:.2f}"
                )
            else:
                st.warning("Not enough labeled variety to train a model — used keyword-based fallback instead.")

            fig2 = px.pie(idf, names="predicted_category", title="Issue Category Breakdown")
            st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(
                idf[["number", "title", "predicted_category", "confidence", "state"]],
                use_container_width=True,
            )

    # --- Tab 3: Bug-prone files ---
    with tab3:
        file_df = build_file_feature_table(commits)
        if file_df.empty:
            st.info("No file-change data available (try increasing commits analyzed).")
        else:
            st.markdown(
                "**Model:** Logistic Regression trained on commit frequency + author diversity "
                "per file to estimate bug-proneness risk."
            )
            predictor = BugProneFilePredictor()
            scored = predictor.fit_predict(file_df)
            if predictor.is_trained:
                st.success("Model trained successfully on this repo's commit history.")
            else:
                st.warning("Not enough variety to train — used a normalized frequency heuristic instead.")

            top_risky = scored.head(15)
            fig3 = px.bar(
                top_risky, x="risk_score", y="file", orientation="h",
                title="Top 15 Bug-Prone Files (Risk Score)",
            )
            fig3.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig3, use_container_width=True)
            st.dataframe(scored, use_container_width=True)

    # --- Tab 4: Contributors ---
    with tab4:
        cdf = commits_to_dataframe(commits)
        cont = contributor_activity(cdf)
        if cont.empty:
            st.info("No contributor data available.")
        else:
            fig4 = px.bar(cont.head(15), x="author", y="commits", title="Top Contributors (by commits analyzed)")
            st.plotly_chart(fig4, use_container_width=True)
            st.dataframe(cont, use_container_width=True)

else:
    st.info("👈 Enter a repo (e.g. `psf/requests`) in the sidebar and click **Analyze Repository** to start.")
