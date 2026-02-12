# Next.js + AI GitHub Scraper

This project contains a single script, `github.py`, that scans GitHub for popular, active TypeScript repositories and keeps only repos that look like **Next.js + AI** projects.

It then exports:
- matched repositories
- top contributors for each matched repo
- a ranked rollup of contributors across all matched repos

## What It Does (Default Behavior)

By default, the script:
- searches GitHub repositories with:
  - `language:TypeScript`
  - at least `300` stars
  - pushed within the last `365` days
  - not archived
- detects Next.js signals (`next` dependency, Next scripts, Next config, app/pages folders)
- detects AI-related dependencies (for example `openai`, `ai`, `@ai-sdk/*`, `langchain`, `llamaindex`)
- keeps only repos that match both Next.js and AI signals
- fetches top contributors for each matched repo
- optionally enriches contributors with profile data (name, company, location, etc.)

## Prerequisites

- Python 3.9+
- A GitHub token in `GITHUB_TOKEN`

The script exits immediately if `GITHUB_TOKEN` is missing.

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your token:

```bash
export GITHUB_TOKEN="your_token_here"
```

PowerShell:

```powershell
$env:GITHUB_TOKEN="your_token_here"
```

## Run

```bash
python3 github.py
```

When finished, it prints a summary and writes CSV files to the current directory.

## Output Files

- `repos.csv`: one row per matched repository with detection signals and matched dependencies
- `repo_contributors.csv`: contributor rows per matched repository (with optional profile enrichment)
- `top_users.csv`: contributor leaderboard across all matched repositories

## Configuration

There are no CLI flags right now. Configure behavior by editing constants at the top of `github.py`.

Common settings:
- `MAX_REPOS`: stop after this many qualifying repos
- `MIN_STARS`: minimum stars for search
- `MIN_LAST_PUSHED_DAYS`: recency filter (in days)
- `TOP_CONTRIBUTORS`: contributors fetched per repo
- `ENRICH_CONTRIBUTORS`: enable/disable user profile enrichment
- `EXCLUDE_BOT_ACCOUNTS`: skip likely bot users
- `REQUIRE_NEXTJS`: require Next.js signal to qualify
- `REQUIRE_AI_PACKAGES`: require AI dependency signal to qualify
- `PACE_SECONDS`: delay between API calls (helps with rate limits)

Output file names are also configurable:
- `OUT_REPOS`
- `OUT_CONTRIBS`
- `OUT_TOP_USERS`

## Troubleshooting

- `Missing GITHUB_TOKEN env var`:
  - Set `GITHUB_TOKEN` in your shell before running.
- Very slow runs:
  - This is expected when scanning many repos and enriching users.
  - Try lowering `MAX_REPOS` or setting `ENRICH_CONTRIBUTORS = False`.
- Too few or zero matches:
  - Lower `MIN_STARS` and/or loosen filters by setting `REQUIRE_NEXTJS` or `REQUIRE_AI_PACKAGES` to `False`.

## Notes

- The script uses GitHub REST API endpoints for search, repository tree/content checks, contributors, and user profiles.
- Some repository trees may be truncated by GitHub. The `tree_truncated` column in `repos.csv` flags this.
