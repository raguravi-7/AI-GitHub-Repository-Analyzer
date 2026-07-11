"""
fetcher.py
Handles all communication with the GitHub REST API.
Pulls repository metadata, issues, and commit history (with changed files).
"""

import requests
import time
from datetime import datetime

GITHUB_API = "https://api.github.com"


class GitHubFetcher:
    def __init__(self, owner, repo, token=None):
        self.owner = owner
        self.repo = repo
        self.session = requests.Session()
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.session.headers.update(headers)
        self.base = f"{GITHUB_API}/repos/{owner}/{repo}"

    def _get(self, url, params=None):
        resp = self.session.get(url, params=params)
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            raise RuntimeError(
                "GitHub API rate limit exceeded. Add a personal access token "
                "to increase the limit (60/hr without token, 5000/hr with token)."
            )
        if resp.status_code == 404:
            raise RuntimeError("Repository not found. Check owner/repo name.")
        resp.raise_for_status()
        return resp

    def get_repo_info(self):
        r = self._get(self.base)
        return r.json()

    def get_issues(self, max_pages=5, per_page=50):
        """Fetch issues (excludes pull requests). Returns list of dicts."""
        issues = []
        for page in range(1, max_pages + 1):
            r = self._get(
                f"{self.base}/issues",
                params={"state": "all", "per_page": per_page, "page": page},
            )
            batch = r.json()
            if not batch:
                break
            for item in batch:
                if "pull_request" in item:
                    continue  # skip PRs, keep only real issues
                issues.append(
                    {
                        "number": item["number"],
                        "title": item.get("title") or "",
                        "body": item.get("body") or "",
                        "state": item.get("state"),
                        "labels": [l["name"] for l in item.get("labels", [])],
                        "created_at": item.get("created_at"),
                        "comments": item.get("comments", 0),
                    }
                )
            if len(batch) < per_page:
                break
        return issues

    def get_commits(self, max_pages=3, per_page=50):
        """Fetch recent commits (basic info only, no file list yet)."""
        commits = []
        for page in range(1, max_pages + 1):
            r = self._get(
                f"{self.base}/commits",
                params={"per_page": per_page, "page": page},
            )
            batch = r.json()
            if not batch:
                break
            for c in batch:
                commit_info = c.get("commit", {})
                author = commit_info.get("author", {}) or {}
                commits.append(
                    {
                        "sha": c["sha"],
                        "message": commit_info.get("message", ""),
                        "author": author.get("name", "unknown"),
                        "date": author.get("date"),
                    }
                )
            if len(batch) < per_page:
                break
        return commits

    def get_commit_files(self, sha):
        """Fetch list of files changed in a single commit (1 API call each)."""
        r = self._get(f"{self.base}/commits/{sha}")
        data = r.json()
        files = data.get("files", [])
        return [f["filename"] for f in files]

    def enrich_commits_with_files(self, commits, limit=60, pause=0.0):
        """
        Adds a 'files' key to each commit dict by calling the per-commit endpoint.
        Limited to `limit` commits to respect API rate limits.
        """
        enriched = []
        for c in commits[:limit]:
            try:
                c["files"] = self.get_commit_files(c["sha"])
            except Exception:
                c["files"] = []
            enriched.append(c)
            if pause:
                time.sleep(pause)
        return enriched
