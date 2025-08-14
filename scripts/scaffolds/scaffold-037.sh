#!/usr/bin/env bash
# scripts/scaffolds/scaffold_037.sh
# Non-breaking prep for LocalKeep: brand env fields, optional config hook, community files,
# docs, GitHub templates, manifest/icons, robots/sitemap, landing page.
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p .github/ISSUE_TEMPLATE docs static/icons templates scripts/scaffolds

# --- 1) Append brand env (idempotent) ---
touch .env.example
if ! grep -q "^BRAND_NAME=" .env.example; then
cat >> .env.example <<'ENV'

# ---- Branding (optional; templates may use these if present) ----
BRAND_NAME=LocalKeep
BRAND_TAGLINE=Local-first notes that think with you.
BRAND_PRIMARY_HEX=#6D28D9
BRAND_OG_IMAGE=/static/opengraph.svg
ENV
echo "• appended BRAND_* to .env.example"
fi

# --- 2) config.py: add optional brand fields (no breaking changes) ---
if [[ -f config.py ]] && ! grep -q "brand_name" config.py; then
  bk config.py
  cat >> config.py <<'PY'

try:
    from pydantic import BaseSettings
    class _BrandMixin(BaseSettings):
        brand_name: str = "LocalKeep"
        brand_tagline: str = "Local-first notes that think with you."
        brand_primary_hex: str = "#6D28D9"
        brand_og_image: str = "/static/opengraph.svg"
        class Config:
            env_file = ".env"
            env_prefix = ""
            fields = {
                "brand_name": {"env": ["BRAND_NAME"]},
                "brand_tagline": {"env": ["BRAND_TAGLINE"]},
                "brand_primary_hex": {"env": ["BRAND_PRIMARY_HEX"]},
                "brand_og_image": {"env": ["BRAND_OG_IMAGE"]},
            }
    # if Settings already loaded as `settings`, set defaults if missing
    try:
        for k,v in _BrandMixin().dict().items():
            setattr(settings, k, getattr(settings, k, v))
    except Exception:
        pass
except Exception:
    pass
PY
  echo "• appended optional brand fields to config.py"
fi

# --- 3) Community health files ---
if [[ ! -f CODE_OF_CONDUCT.md ]]; then
cat > CODE_OF_CONDUCT.md <<'MD'
# Code of Conduct
We use the Contributor Covenant 2.1. Be kind, assume good intent, and report incidents to maintainers.

## Enforcement
Email the maintainers (see SECURITY.md) with the subject “Code of Conduct”.
Full text: https://www.contributor-covenant.org/version/2/1/code_of_conduct/
MD
fi

if [[ ! -f CONTRIBUTING.md ]]; then
cat > CONTRIBUTING.md <<'MD'
# Contributing

Thanks for helping build LocalKeep!

## Dev setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8084

Branch & Commit
Branch: feat/<area>-<short>, fix/<area>-<short>

Commits: Conventional Commits (e.g., feat(ui): add command palette)

PR Quality
Focused diffs, testing notes, docs updates when needed.
MD
fi

if [[ ! -f SECURITY.md ]]; then
cat > SECURITY.md <<'MD'

Security Policy
Report vulnerabilities privately: security@localkeep.app (or GitHub Security Advisories).
We aim to acknowledge within 72 hours.

Supported: latest minor until v1.0.
MD
fi

#--- 4) GitHub templates & config ---
if [[ ! -f .github/ISSUE_TEMPLATE/bug_report.yml ]]; then
cat > .github/ISSUE_TEMPLATE/bug_report.yml <<'YML'
name: Bug report
description: File a bug report
labels: [bug]
body:

type: textarea
attributes: { label: What happened?, description: Steps to reproduce }
validations: { required: true }

type: textarea
attributes: { label: Expected behavior }

type: textarea
attributes: { label: Logs/screenshots }

type: input
attributes: { label: App version (/about) }
YML
fi

if [[ ! -f .github/ISSUE_TEMPLATE/feature_request.yml ]]; then
cat > .github/ISSUE_TEMPLATE/feature_request.yml <<'YML'
name: Feature request
description: Propose an enhancement
labels: [enhancement]
body:

type: textarea
attributes: { label: Problem or use case }
validations: { required: true }

type: textarea
attributes: { label: Proposal }

type: textarea
attributes: { label: Alternatives considered }
YML
fi

if [[ ! -f .github/ISSUE_TEMPLATE/config.yml ]]; then
cat > .github/ISSUE_TEMPLATE/config.yml <<'YML'
blank_issues_enabled: false
contact_links:

name: Questions & help
url: https://github.com/dhouchin1/second-brain-local/discussions
about: Ask questions and share ideas here.
YML
fi

if [[ ! -f .github/PULL_REQUEST_TEMPLATE.md ]]; then
cat > .github/PULL_REQUEST_TEMPLATE.md <<'MD'

Summary
<!-- What does this change and why? -->
User Impact
<!-- Screenshots or notes -->
Tests
 Unit/API tests added or adjusted

 Manual verification steps included

Checklist
 Conventional commit title

 Docs updated (if needed)

 No secrets in code or logs
MD
fi

if [[ ! -f .github/CODEOWNERS ]]; then
cat > .github/CODEOWNERS <<'TXT'

@dhouchin1
TXT
fi

if [[ ! -f .github/FUNDING.yml ]]; then
cat > .github/FUNDING.yml <<'YML'
github: []
custom: ["https://localkeep.app"]
YML
fi

# --- 5) Docs stubs ---
if [[ ! -f docs/roadmap.md ]]; then
cat > docs/roadmap.md <<'MD'

LocalKeep Roadmap (preview)
Quality gates (pre-commit + CI)

Embeddings-powered Similar

PWA offline capture

Importers (ENEX/Notion)

Two-way Obsidian sync (opt-in)
MD
fi

if [[ ! -f docs/architecture.md ]]; then
cat > docs/architecture.md <<'MD'

Architecture (high level)
FastAPI app (auth, capture, search)

SQLite + FTS5 (notes + FTS mirror + audit)

Optional LLM job queue

Optional vector index (future)

Static UI with templates + command palette
MD
fi

if [[ ! -f docs/api.md ]]; then
cat > docs/api.md <<'MD'

API Notes (brief)
POST /token — JWT via OAuth2 password flow

POST /capture — note, tags, optional file

GET /api/search?q=...

GET /export/json, GET /export/markdown.zip
Use Authorization: Bearer <token>.
MD
fi

if [[ ! -f CHANGELOG.md ]]; then
cat > CHANGELOG.md <<'MD'

Changelog
All notable changes to this project will be documented here.
MD
fi

# --- 6) Brand assets: manifest, icons, robots/sitemap, OG ---
if [[ ! -f static/manifest.webmanifest ]]; then
cat > static/manifest.webmanifest <<'JSON'
{
"name": "LocalKeep",
"short_name": "LocalKeep",
"start_url": "/",
"display": "standalone",
"background_color": "#ffffff",
"theme_color": "#6D28D9",
"icons": [
{ "src": "/static/icons/icon-192.svg", "type": "image/svg+xml", "sizes": "192x192" },
{ "src": "/static/icons/icon-512.svg", "type": "image/svg+xml", "sizes": "512x512" }
]
}
JSON
fi

if [[ ! -f static/icons/icon-192.svg ]]; then
cat > static/icons/icon-192.svg <<'SVG'
<svg xmlns="http://www.w3.org/2000/svg" width="192" height="192" viewBox="0 0 192 192">
<rect width="100%" height="100%" rx="24" fill="#6D28D9"/>
<rect x="32" y="44" width="128" height="104" rx="12" fill="#fff" opacity="0.96"/>
<text x="50%" y="56%" text-anchor="middle" font-family="Inter, Arial" font-size="36" fill="#6D28D9">LK</text>
</svg>
SVG
fi
if [[ ! -f static/icons/icon-512.svg ]]; then
cp static/icons/icon-192.svg static/icons/icon-512.svg
fi

if [[ ! -f static/opengraph.svg ]]; then
cat > static/opengraph.svg <<'SVG'
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
<defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#6D28D9"/><stop offset="1" stop-color="#4C1D95"/></linearGradient></defs>
<rect width="100%" height="100%" fill="url(#g)"/>
<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="Inter, Arial" font-size="84" fill="#fff">LocalKeep</text>
<text x="50%" y="62%" dominant-baseline="middle" text-anchor="middle" font-family="Inter, Arial" font-size="28" fill="#EDE9FE">Local-first notes that think with you.</text>
</svg>
SVG
fi

if [[ ! -f static/robots.txt ]]; then
cat > static/robots.txt <<'TXT'
User-agent: *
Allow: /
Sitemap: /static/sitemap.xml
TXT
fi

if [[ ! -f static/sitemap.xml ]]; then
cat > static/sitemap.xml <<'XML'

<?xml version="1.0" encoding="UTF-8"?> <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"> <url><loc>/</loc></url> <url><loc>/about</loc></url> <url><loc>/exports</loc></url> </urlset> XML fi
--- 7) Landing template (not linked unless you decide) ---
if [[ ! -f templates/landing.html ]]; then
cat > templates/landing.html <<'HTML'
{% extends "base.html" %}
{% block title %}LocalKeep — Local-first notes{% endblock %}
{% block content %}

<div class="card" style="max-width:840px; margin:2rem auto;"> <h1 style="margin:0 0 .5rem 0;">LocalKeep</h1> <p class="text-lg" style="color:#475569;">Local-first notes that think with you. Private by default. Fast search. Ollama-powered summaries (local models).</p> <div style="display:flex; gap:.6rem; flex-wrap:wrap; margin-top:.6rem;"> <a class="btn" href="/">Open app</a> <a class="btn secondary" href="/exports">Try an export</a> </div> <hr style="margin:1rem 0;"> <ul> <li>Local SQLite with FTS5</li> <li>Tagging, quick capture, audio uploads</li> <li>Ollama integration for titles/summaries/tags</li> </ul> </div> {% endblock %} HTML fi
--- 8) Base head: add manifest + og meta if missing (safe append) ---
if [[ -f templates/base.html ]] && ! grep -q "manifest.webmanifest" templates/base.html; then
bk templates/base.html
cat >> templates/base.html <<'HTML'

<!-- scaffold_037 meta/manifest --> <link rel="manifest" href="/static/manifest.webmanifest"> <link rel="icon" href="/static/icons/icon-192.svg"> <meta property="og:title" content="LocalKeep"> <meta property="og:description" content="Local-first notes that think with you."> <meta property="og:image" content="/static/opengraph.svg"> <meta name="theme-color" content="#6D28D9"> HTML echo "• appended manifest + OG meta to templates/base.html" fi
echo "✓ scaffold_037 complete (brand/docs/community assets added)."
echo "Next:"
echo " - Replace SVG icons/OG with final assets when ready."
echo " - (Optional) open a PR using your helper:"
echo " make ship S=scripts/scaffolds/scaffold_037.sh M="chore: brand+community prep""
XML
fi