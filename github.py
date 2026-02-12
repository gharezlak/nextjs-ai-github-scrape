import os
import time
import csv
import base64
import json
import math
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set, Iterator
from urllib.parse import quote

import requests

# =========================
# Config / Tunables
# =========================
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise SystemExit("Missing GITHUB_TOKEN env var")

API_BASE = "https://api.github.com"
USER_AGENT = "github-repo-scout-plus"

MAX_REPOS = 300                 # number of qualifying repos to scrape
SEARCH_PER_PAGE = 50            # max 100
MIN_STARS = 300
MIN_LAST_PUSHED_DAYS = 365      # prefer actively maintained repos
MAX_SEARCH_PAGES = 20           # GitHub Search API effectively caps at 1000 results

TOP_CONTRIBUTORS = 10
ENRICH_CONTRIBUTORS = True      # user profile enrichment hits /users/{login}
EXCLUDE_BOT_ACCOUNTS = True

# Keep only repos that match your target profile
REQUIRE_NEXTJS = True
REQUIRE_AI_PACKAGES = True

# Monorepo scanning knobs
TREE_TRUNCATION_WARNING = True
MAX_PACKAGE_JSON_PER_REPO = 30  # cap number of package.json parsed per repo
MAX_TREE_ENTRIES = 5000         # cap tree list size (GitHub API returns truncated sometimes)

# pacing to reduce secondary rate-limit risk
PACE_SECONDS = 0.20

OUT_REPOS = "repos.csv"
OUT_CONTRIBS = "repo_contributors.csv"
OUT_TOP_USERS = "top_users.csv"

# Core web stack signals
NEXT_DEP_KEYS = {"next"}
REACT_DEP_KEYS = {"react", "react-dom"}

# Modern AI package names (npm) to detect explicitly
AI_DEP_KEYS = {
    # Model provider SDKs
    "openai",
    "@anthropic-ai/sdk",
    "@google/generative-ai",
    "@google/genai",
    "groq-sdk",
    "cohere-ai",
    "replicate",
    "together-ai",
    "ollama",
    "mistralai",
    "voyageai",
    # Vercel AI SDK ecosystem
    "ai",                    # current package name
    "@vercel/ai",            # legacy package name
    "@ai-sdk/openai",
    "@ai-sdk/anthropic",
    "@ai-sdk/google",
    "@ai-sdk/groq",
    "@ai-sdk/azure",
    "@ai-sdk/amazon-bedrock",
    "@ai-sdk/react",
    # LangChain / orchestration
    "langchain",
    "@langchain/core",
    "@langchain/openai",
    "@langchain/anthropic",
    "@langchain/community",
    "langsmith",
    "llamaindex",
    # RAG / vector db clients often used in modern AI apps
    "@pinecone-database/pinecone",
    "weaviate-client",
    "@qdrant/js-client-rest",
    "chromadb",
    "@supabase/supabase-js",
}

# Strong full-stack package signals (non-AI, but high-signal for modern app stacks)
FULLSTACK_DEP_KEYS = {
    "prisma",
    "@prisma/client",
    "drizzle-orm",
    "next-auth",
    "@auth/core",
    "@auth/prisma-adapter",
    "@trpc/server",
    "@trpc/client",
    "@trpc/react-query",
    "zod",
    "zustand",
    "@tanstack/react-query",
}

# Heuristics for likely “app package.json” in monorepos
PREFERRED_PKG_PATH_HINTS = [
    "/apps/",
    "/packages/",
    "/web/",
    "/frontend/",
    "/client/",
    "/site/",
    "/dashboard/",
]


# =========================
# HTTP + Rate Limit Handling
# =========================
def gh_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": USER_AGENT,
    }


def _backoff_sleep(resp: requests.Response, attempt: int) -> None:
    """
    Handle primary + secondary-ish rate limits.
    """
    retry_after = resp.headers.get("retry-after")
    if retry_after:
        wait = min(int(retry_after), 60) + 1
        time.sleep(wait)
        return

    remaining = resp.headers.get("x-ratelimit-remaining")
    reset = resp.headers.get("x-ratelimit-reset")

    if remaining == "0" and reset:
        now = int(time.time())
        reset_ts = int(reset)
        wait = max(0, reset_ts - now + 2)
        time.sleep(min(wait, 180))
        return

    wait = min((2 ** attempt), 60)
    time.sleep(wait)


def gh_get(url: str, params: Optional[Dict[str, Any]] = None, attempts: int = 7) -> requests.Response:
    last: Optional[requests.Response] = None
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            resp = requests.get(url, headers=gh_headers(), params=params, timeout=40)
            last = resp
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(min(2 ** attempt, 30))
            continue

        if resp.status_code in (403, 429):
            _backoff_sleep(resp, attempt)
            continue

        return resp

    if last is not None:
        return last
    raise RuntimeError(f"GET failed after {attempts} attempts: {url}") from last_exc


def gh_head(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    attempts: int = 4,
) -> requests.Response:
    last: Optional[requests.Response] = None
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            resp = requests.head(url, headers=gh_headers(), params=params, timeout=30)
            last = resp
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(min(2 ** attempt, 30))
            continue

        if resp.status_code in (403, 429):
            _backoff_sleep(resp, attempt)
            continue

        return resp

    if last is not None:
        return last
    raise RuntimeError(f"HEAD failed after {attempts} attempts: {url}") from last_exc


# =========================
# GitHub Search
# =========================
def iter_typescript_repos() -> Iterator[Dict[str, Any]]:
    page = 1
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=MIN_LAST_PUSHED_DAYS)).date().isoformat()
    q = f"language:TypeScript stars:>={MIN_STARS} archived:false pushed:>={cutoff_date}"

    while page <= MAX_SEARCH_PAGES:
        resp = gh_get(
            f"{API_BASE}/search/repositories",
            params={
                "q": q,
                "sort": "stars",
                "order": "desc",
                "per_page": SEARCH_PER_PAGE,
                "page": page,
            },
        )
        if not resp.ok:
            raise RuntimeError(f"Search failed: {resp.status_code} {resp.text}")

        data = resp.json()
        items = data.get("items", [])
        if not items:
            break

        for item in items:
            if isinstance(item, dict):
                yield item
        page += 1
        time.sleep(PACE_SECONDS)


def split_full_name(full_name: str) -> Tuple[str, str]:
    owner, repo = full_name.split("/", 1)
    return owner, repo


# =========================
# Contents / Tree / Parsing
# =========================
def _decode_b64_json(content_b64: str) -> Optional[Dict[str, Any]]:
    try:
        raw = base64.b64decode(content_b64).decode("utf-8", errors="replace")
        return json.loads(raw)
    except Exception:
        return None


def fetch_contents_file(owner: str, repo: str, path: str, ref: Optional[str] = None) -> Optional[Dict[str, Any]]:
    # Contents API
    url = f"{API_BASE}/repos/{owner}/{repo}/contents/{path}"
    params = {}
    if ref:
        params["ref"] = ref

    resp = gh_get(url, params=params if params else None)
    if resp.status_code == 404:
        return None
    if not resp.ok:
        return None

    data = resp.json()
    if not isinstance(data, dict):
        return None
    if data.get("type") != "file":
        return None
    if data.get("encoding") != "base64":
        return None
    content = data.get("content")
    if not content:
        return None

    return _decode_b64_json(content)


def list_repo_tree(owner: str, repo: str, default_branch: str) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Use Git Trees API: /git/trees/{branch}?recursive=1
    Returns (tree_entries, is_truncated)
    """
    encoded_ref = quote(default_branch, safe="")
    url = f"{API_BASE}/repos/{owner}/{repo}/git/trees/{encoded_ref}"
    resp = gh_get(url, params={"recursive": "1"})
    if not resp.ok:
        return ([], False)

    data = resp.json()
    tree = data.get("tree", [])
    truncated = bool(data.get("truncated", False))

    if not isinstance(tree, list):
        return ([], truncated)

    if len(tree) > MAX_TREE_ENTRIES:
        # keep it bounded
        tree = tree[:MAX_TREE_ENTRIES]
        truncated = True

    return (tree, truncated)


def choose_package_json_paths(tree: List[Dict[str, Any]]) -> List[str]:
    """
    Find all package.json paths and rank them with heuristics.
    """
    pkg_paths = []
    for e in tree:
        if e.get("type") == "blob" and e.get("path", "").endswith("package.json"):
            pkg_paths.append(e["path"])

    def score(p: str) -> int:
        s = 0
        if p == "package.json":
            s += 1000
        # prefer app-ish areas
        for hint in PREFERRED_PKG_PATH_HINTS:
            if hint in f"/{p}":
                s += 50
        # shorter path tends to be primary package
        s += max(0, 60 - len(p))
        # penalize test/examples docs
        low = p.lower()
        if "/examples/" in f"/{low}":
            s -= 30
        if "/test" in low or "/tests" in low:
            s -= 20
        if "/docs/" in f"/{low}":
            s -= 10
        return s

    pkg_paths.sort(key=score, reverse=True)
    return pkg_paths


def check_path_exists(owner: str, repo: str, path: str, ref: Optional[str] = None) -> bool:
    url = f"{API_BASE}/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": ref} if ref else None
    resp = gh_head(url, params=params, attempts=3)
    # HEAD doesn't always behave perfectly with the API; fall back to GET-ish logic on weird statuses
    if resp.status_code in (200, 302):
        return True
    if resp.status_code == 404:
        return False

    r2 = gh_get(url, params=params)
    return r2.status_code != 404 and r2.ok


# =========================
# Dependency + Next.js Detection
# =========================
@dataclass
class PackageSignals:
    has_next_dep: bool
    has_react_dep: bool
    has_next_script: bool
    matched_ai_deps: Set[str]
    matched_fullstack_deps: Set[str]
    matched_pkg_paths: Set[str]


@dataclass
class RepoSignals:
    is_nextjs: bool
    is_ai_repo: bool
    is_fullstack_repo: bool
    qualifies: bool
    has_openai: bool
    has_vercel_ai_sdk: bool
    has_langchain: bool
    has_llamaindex: bool
    has_next_dep: bool
    has_next_script: bool
    matched_ai_deps: Set[str]
    matched_fullstack_deps: Set[str]
    matched_pkg_paths: List[str]
    has_next_config: bool
    has_app_dir: bool
    has_pages_dir: bool
    tree_truncated: bool


def _looks_like_next_script(script_value: str) -> bool:
    low = script_value.lower()
    if "next dev" in low or "next build" in low or "next start" in low:
        return True
    tokens = low.replace("&&", " ").replace("||", " ").replace(";", " ").split()
    return "next" in tokens


def analyze_packages(pkgs: List[Tuple[str, Dict[str, Any]]]) -> PackageSignals:
    matched_ai_deps: Set[str] = set()
    matched_fullstack_deps: Set[str] = set()
    matched_paths: Set[str] = set()
    has_next_dep = False
    has_react_dep = False
    has_next_script = False

    for pkg_path, pkg in pkgs:
        deps: Dict[str, Any] = {}
        deps.update(pkg.get("dependencies", {}) or {})
        deps.update(pkg.get("devDependencies", {}) or {})
        deps.update(pkg.get("peerDependencies", {}) or {})
        deps.update(pkg.get("optionalDependencies", {}) or {})

        dep_names = set(deps.keys())
        ai_matches = AI_DEP_KEYS.intersection(dep_names)
        fullstack_matches = FULLSTACK_DEP_KEYS.intersection(dep_names)

        if ai_matches or fullstack_matches or NEXT_DEP_KEYS.intersection(dep_names):
            matched_paths.add(pkg_path)

        if ai_matches:
            matched_ai_deps |= ai_matches
        if fullstack_matches:
            matched_fullstack_deps |= fullstack_matches

        if NEXT_DEP_KEYS.intersection(dep_names):
            has_next_dep = True
        if REACT_DEP_KEYS.intersection(dep_names):
            has_react_dep = True

        scripts = pkg.get("scripts", {}) or {}
        if isinstance(scripts, dict):
            for script_cmd in scripts.values():
                if isinstance(script_cmd, str) and _looks_like_next_script(script_cmd):
                    has_next_script = True
                    break

    return PackageSignals(
        has_next_dep=has_next_dep,
        has_react_dep=has_react_dep,
        has_next_script=has_next_script,
        matched_ai_deps=matched_ai_deps,
        matched_fullstack_deps=matched_fullstack_deps,
        matched_pkg_paths=matched_paths,
    )


def compute_signals(
    package_signals: PackageSignals,
    has_next_config: bool,
    has_app_dir: bool,
    has_pages_dir: bool,
    tree_truncated: bool,
) -> RepoSignals:
    dep_next = package_signals.has_next_dep
    dep_react = package_signals.has_react_dep
    nextish_fs = has_next_config or has_app_dir or has_pages_dir

    # Next.js signal should be strict enough to avoid random /app folders.
    is_nextjs = (
        dep_next
        or has_next_config
        or (package_signals.has_next_script and dep_react)
        or (nextish_fs and dep_next)
    )

    matched_ai_deps = set(sorted(package_signals.matched_ai_deps))
    matched_fullstack_deps = set(sorted(package_signals.matched_fullstack_deps))
    is_ai_repo = len(matched_ai_deps) > 0
    is_fullstack_repo = len(matched_fullstack_deps) > 0

    qualifies = (is_nextjs or not REQUIRE_NEXTJS) and (is_ai_repo or not REQUIRE_AI_PACKAGES)

    has_openai = (
        ("openai" in matched_ai_deps)
        or ("@ai-sdk/openai" in matched_ai_deps)
        or ("@langchain/openai" in matched_ai_deps)
    )
    has_vercel_ai_sdk = (
        ("ai" in matched_ai_deps)
        or ("@vercel/ai" in matched_ai_deps)
        or any(k.startswith("@ai-sdk/") for k in matched_ai_deps)
    )
    has_langchain = ("langchain" in matched_ai_deps) or any(
        k.startswith("@langchain/") for k in matched_ai_deps
    )
    has_llamaindex = "llamaindex" in matched_ai_deps

    return RepoSignals(
        is_nextjs=is_nextjs,
        is_ai_repo=is_ai_repo,
        is_fullstack_repo=is_fullstack_repo,
        qualifies=qualifies,
        has_openai=has_openai,
        has_vercel_ai_sdk=has_vercel_ai_sdk,
        has_langchain=has_langchain,
        has_llamaindex=has_llamaindex,
        has_next_dep=dep_next,
        has_next_script=package_signals.has_next_script,
        matched_ai_deps=matched_ai_deps,
        matched_fullstack_deps=matched_fullstack_deps,
        matched_pkg_paths=sorted(package_signals.matched_pkg_paths),
        has_next_config=has_next_config,
        has_app_dir=has_app_dir,
        has_pages_dir=has_pages_dir,
        tree_truncated=tree_truncated,
    )


# =========================
# Contributors + Enrichment
# =========================
_user_cache: Dict[str, Dict[str, Any]] = {}


def fetch_top_contributors(owner: str, repo: str, top_n: int) -> List[Dict[str, Any]]:
    url = f"{API_BASE}/repos/{owner}/{repo}/contributors"
    resp = gh_get(url, params={"per_page": top_n})

    if not resp.ok:
        return []

    data = resp.json()
    if not isinstance(data, list):
        return []

    out = []
    for c in data[:top_n]:
        login = c.get("login")
        if not login:
            continue
        out.append(
            {
                "login": login,
                "contributions": c.get("contributions", ""),
                "url": c.get("html_url", ""),
                "type": c.get("type", ""),
            }
        )
    return out


def fetch_user_profile(login: str) -> Dict[str, Any]:
    """
    Cache user profiles to minimize rate-limit usage.
    """
    if login in _user_cache:
        return _user_cache[login]

    resp = gh_get(f"{API_BASE}/users/{login}")
    if not resp.ok:
        _user_cache[login] = {}
        return {}

    data = resp.json()
    if not isinstance(data, dict):
        _user_cache[login] = {}
        return {}

    # pick common fields useful for CSV
    out = {
        "name": data.get("name", "") or "",
        "company": data.get("company", "") or "",
        "blog": data.get("blog", "") or "",
        "location": data.get("location", "") or "",
        "twitter_username": data.get("twitter_username", "") or "",
        "followers": data.get("followers", "") or "",
        "following": data.get("following", "") or "",
        "public_repos": data.get("public_repos", "") or "",
        "bio": data.get("bio", "") or "",
    }
    _user_cache[login] = out
    return out


# =========================
# CSV helpers
# =========================
def join_list(values: List[str]) -> str:
    return "; ".join([v for v in values if v])


def parse_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def is_probably_bot(login: str, account_type: str) -> bool:
    low_login = login.lower()
    return account_type.lower() == "bot" or low_login.endswith("[bot]")


def main() -> None:
    user_rollup: Dict[str, Dict[str, Any]] = {}
    qualifying_repo_count = 0
    scanned_repo_count = 0

    repos_fieldnames = [
        "repo",
        "url",
        "stars",
        "language",
        "default_branch",
        "is_nextjs",
        "is_ai_repo",
        "is_fullstack_repo",
        "qualifies_target_profile",
        "has_openai",
        "has_vercel_ai_sdk",
        "has_langchain",
        "has_llamaindex",
        "has_next_dep",
        "has_next_script",
        "matched_ai_deps",
        "matched_fullstack_deps",
        "matched_package_json_paths",
        "has_next_config",
        "has_app_dir",
        "has_pages_dir",
        "tree_truncated",
        "package_json_count_scanned",
    ]

    contrib_fieldnames = [
        "repo",
        "repo_url",
        "stars",
        "contributor_login",
        "contributor_type",
        "contributor_url",
        "contributions",
        "repo_matched_ai_deps",
        "repo_matched_fullstack_deps",
        # enrichment
        "name",
        "company",
        "blog",
        "location",
        "twitter_username",
        "followers",
        "following",
        "public_repos",
        "bio",
    ]

    top_users_fieldnames = [
        "rank",
        "login",
        "url",
        "repos_matched",
        "total_contributions",
        "weighted_score",
        "max_repo_stars",
        "name",
        "company",
        "blog",
        "location",
        "twitter_username",
        "followers",
        "following",
        "public_repos",
        "bio",
    ]

    with open(OUT_REPOS, "w", newline="", encoding="utf-8") as f_repos, open(
        OUT_CONTRIBS, "w", newline="", encoding="utf-8"
    ) as f_contrib, open(OUT_TOP_USERS, "w", newline="", encoding="utf-8") as f_top_users:
        w_repos = csv.DictWriter(f_repos, fieldnames=repos_fieldnames)
        w_repos.writeheader()

        w_contrib = csv.DictWriter(f_contrib, fieldnames=contrib_fieldnames)
        w_contrib.writeheader()

        w_top_users = csv.DictWriter(f_top_users, fieldnames=top_users_fieldnames)
        w_top_users.writeheader()

        for r in iter_typescript_repos():
            if qualifying_repo_count >= MAX_REPOS:
                break

            scanned_repo_count += 1
            full_name = r.get("full_name", "")
            if not full_name or "/" not in full_name:
                continue

            owner, repo = split_full_name(full_name)
            repo_url = r.get("html_url", "")
            stars = parse_int(r.get("stargazers_count", 0))
            language = r.get("language", "")
            default_branch = r.get("default_branch", "main") or "main"

            # 1) Tree scan
            tree, truncated = list_repo_tree(owner, repo, default_branch)
            pkg_paths = choose_package_json_paths(tree) if tree else []

            # 2) Pull up to N package.json files
            pkgs: List[Tuple[str, Dict[str, Any]]] = []
            for p in pkg_paths[:MAX_PACKAGE_JSON_PER_REPO]:
                pkg = fetch_contents_file(owner, repo, p, ref=default_branch)
                if pkg:
                    pkgs.append((p, pkg))
                time.sleep(PACE_SECONDS)

            package_signals = analyze_packages(pkgs)

            # 3) Next.js filesystem heuristics
            # next.config.* can be in many variants
            has_next_config = (
                check_path_exists(owner, repo, "next.config.js", ref=default_branch)
                or check_path_exists(owner, repo, "next.config.mjs", ref=default_branch)
                or check_path_exists(owner, repo, "next.config.ts", ref=default_branch)
            )

            # app/ and pages/ may exist at root in Next apps
            has_app_dir = check_path_exists(owner, repo, "app", ref=default_branch)
            has_pages_dir = check_path_exists(owner, repo, "pages", ref=default_branch)

            signals = compute_signals(
                package_signals=package_signals,
                has_next_config=has_next_config,
                has_app_dir=has_app_dir,
                has_pages_dir=has_pages_dir,
                tree_truncated=truncated,
            )

            # Skip immediately unless the repo is both Next.js and AI.
            if not (signals.is_nextjs and signals.is_ai_repo):
                if scanned_repo_count % 20 == 0:
                    print(
                        f"Scanned {scanned_repo_count} repos "
                        f"(qualifying={qualifying_repo_count}/{MAX_REPOS}). "
                        f"Cached users: {len(_user_cache)}"
                    )
                time.sleep(PACE_SECONDS)
                continue

            qualifying_repo_count += 1

            # Write repo row (qualifying repos only)
            w_repos.writerow(
                {
                    "repo": full_name,
                    "url": repo_url,
                    "stars": stars,
                    "language": language,
                    "default_branch": default_branch,
                    "is_nextjs": signals.is_nextjs,
                    "is_ai_repo": signals.is_ai_repo,
                    "is_fullstack_repo": signals.is_fullstack_repo,
                    "qualifies_target_profile": signals.qualifies,
                    "has_openai": signals.has_openai,
                    "has_vercel_ai_sdk": signals.has_vercel_ai_sdk,
                    "has_langchain": signals.has_langchain,
                    "has_llamaindex": signals.has_llamaindex,
                    "has_next_dep": signals.has_next_dep,
                    "has_next_script": signals.has_next_script,
                    "matched_ai_deps": join_list(sorted(list(signals.matched_ai_deps))),
                    "matched_fullstack_deps": join_list(sorted(list(signals.matched_fullstack_deps))),
                    "matched_package_json_paths": join_list(signals.matched_pkg_paths),
                    "has_next_config": signals.has_next_config,
                    "has_app_dir": signals.has_app_dir,
                    "has_pages_dir": signals.has_pages_dir,
                    "tree_truncated": signals.tree_truncated,
                    "package_json_count_scanned": len(pkgs),
                }
            )

            # 4) Contributors (only for repos that match your target profile)
            contributors = fetch_top_contributors(owner, repo, TOP_CONTRIBUTORS)

            # Write contributor rows
            for c in contributors:
                login = c.get("login", "")
                if not login:
                    continue
                account_type = c.get("type", "")
                if EXCLUDE_BOT_ACCOUNTS and is_probably_bot(login, account_type):
                    continue

                enriched = {}
                if ENRICH_CONTRIBUTORS:
                    enriched = fetch_user_profile(login)
                    time.sleep(PACE_SECONDS)

                contributions = parse_int(c.get("contributions", 0))
                score_boost = 1.0 + math.log10(max(1, stars))
                weighted_score = contributions * score_boost

                w_contrib.writerow(
                    {
                        "repo": full_name,
                        "repo_url": repo_url,
                        "stars": stars,
                        "contributor_login": login,
                        "contributor_type": account_type,
                        "contributor_url": c.get("url", ""),
                        "contributions": contributions,
                        "repo_matched_ai_deps": join_list(sorted(list(signals.matched_ai_deps))),
                        "repo_matched_fullstack_deps": join_list(
                            sorted(list(signals.matched_fullstack_deps))
                        ),
                        "name": enriched.get("name", ""),
                        "company": enriched.get("company", ""),
                        "blog": enriched.get("blog", ""),
                        "location": enriched.get("location", ""),
                        "twitter_username": enriched.get("twitter_username", ""),
                        "followers": enriched.get("followers", ""),
                        "following": enriched.get("following", ""),
                        "public_repos": enriched.get("public_repos", ""),
                        "bio": enriched.get("bio", ""),
                    }
                )

                if login not in user_rollup:
                    user_rollup[login] = {
                        "login": login,
                        "url": c.get("url", ""),
                        "repos": set(),
                        "total_contributions": 0,
                        "weighted_score": 0.0,
                        "max_repo_stars": 0,
                        "name": "",
                        "company": "",
                        "blog": "",
                        "location": "",
                        "twitter_username": "",
                        "followers": 0,
                        "following": 0,
                        "public_repos": 0,
                        "bio": "",
                    }

                user = user_rollup[login]
                user["repos"].add(full_name)
                user["total_contributions"] += contributions
                user["weighted_score"] += weighted_score
                user["max_repo_stars"] = max(user["max_repo_stars"], stars)

                # Fill enrichment fields once with non-empty data
                for field in [
                    "name",
                    "company",
                    "blog",
                    "location",
                    "twitter_username",
                    "bio",
                ]:
                    if not user[field]:
                        user[field] = enriched.get(field, "")
                for field in ["followers", "following", "public_repos"]:
                    if user[field] == 0:
                        user[field] = parse_int(enriched.get(field, 0))

            # progress + pacing
            if scanned_repo_count % 20 == 0:
                print(
                    f"Scanned {scanned_repo_count} repos "
                    f"(qualifying={qualifying_repo_count}/{MAX_REPOS}). "
                    f"Cached users: {len(_user_cache)}"
                )
            time.sleep(PACE_SECONDS)

        ranked_users = sorted(
            user_rollup.values(),
            key=lambda u: (u["weighted_score"], u["total_contributions"], len(u["repos"])),
            reverse=True,
        )

        for rank, u in enumerate(ranked_users, start=1):
            w_top_users.writerow(
                {
                    "rank": rank,
                    "login": u["login"],
                    "url": u["url"],
                    "repos_matched": len(u["repos"]),
                    "total_contributions": u["total_contributions"],
                    "weighted_score": round(u["weighted_score"], 2),
                    "max_repo_stars": u["max_repo_stars"],
                    "name": u["name"],
                    "company": u["company"],
                    "blog": u["blog"],
                    "location": u["location"],
                    "twitter_username": u["twitter_username"],
                    "followers": u["followers"],
                    "following": u["following"],
                    "public_repos": u["public_repos"],
                    "bio": u["bio"],
                }
            )

    if TREE_TRUNCATION_WARNING:
        print("Note: Some repo trees may be truncated by GitHub; 'tree_truncated' in repos.csv flags this.")

    print(
        "Done.\n"
        f"- Wrote {OUT_REPOS}\n"
        f"- Wrote {OUT_CONTRIBS}\n"
        f"- Wrote {OUT_TOP_USERS}\n"
        f"- Matching repos: {qualifying_repo_count}\n"
        f"- Total scanned repos: {scanned_repo_count}"
    )


if __name__ == "__main__":
    main()
