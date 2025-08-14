#!/usr/bin/env bash
# scripts/scaffolds/scaffold_026.sh
# Testing suite (pytest + httpx) and Playwright auto-screenshots (Python)
set -euo pipefail
STAMP="$(date +%Y%m%d-%H%M%S)"; bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p tests/unit tests/api docs/screenshots scripts

# 1) dev requirements
cat > requirements-dev.txt <<'REQ'
pytest>=7.4
pytest-asyncio>=0.23
pytest-cov>=5.0
httpx>=0.27.2
anyio>=4.4
playwright>=1.45
REQ

# 2) basic API tests
cat > tests/api/conftest.py <<'PY'
import pytest, anyio
from fastapi.testclient import TestClient
from app import app

@pytest.fixture(scope="session")
def client():
    return TestClient(app)

@pytest.fixture
def user_and_token(client):
    import random
    uname = f"user{random.randint(1000,9999)}"
    r = client.post("/register", data={"username": uname, "password": "secret"})
    assert r.status_code in (200,201)
    t = client.post("/token", data={"username": uname, "password": "secret"})
    tok = t.json()["access_token"]
    return uname, tok
PY

cat > tests/api/test_auth_and_diag.py <<'PY'
def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert "ok" in r.json()

def test_diag_requires_auth(client):
    r = client.get("/api/diag")
    assert r.status_code in (401, 403)

def test_diag_with_auth(client, user_and_token):
    _, tok = user_and_token
    r = client.get("/api/diag", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    j = r.json()
    assert "version" in j and "db" in j
PY

# 3) Playwright screenshot script (Python)
cat > scripts/e2e_screenshots.py <<'PY'
import os, sys, asyncio, json
from playwright.async_api import async_playwright

BASE = os.environ.get("SB_BASE_URL","http://localhost:8084")
TOKEN = os.environ.get("SB_TOKEN","")

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True)
        ctx = await b.new_context()
        page = await ctx.new_page()
        # If we have a token, set Authorization header for same-origin
        if TOKEN:
            await ctx.set_extra_http_headers({"Authorization": f"Bearer {TOKEN}"})
        # Dashboard
        await page.goto(BASE + "/")
        await page.screenshot(path="docs/screenshots/dashboard.png", full_page=True)
        # About (may require auth)
        await page.goto(BASE + "/about")
        await page.screenshot(path="docs/screenshots/about.png", full_page=True)
        # Exports (if present)
        await page.goto(BASE + "/exports")
        await page.screenshot(path="docs/screenshots/exports.png", full_page=True)
        await b.close()

if __name__ == "__main__":
    asyncio.run(main())
PY

# 4) Makefile helpers
if [[ -f Makefile ]]; then bk Makefile; fi
cat >> Makefile <<'MK'

# === Tests & Screenshots ===
test:
	@. .venv/bin/activate && pytest -q

screenshots:
	@. .venv/bin/activate && python -m playwright install chromium
	@SB_BASE_URL=http://localhost:8084 . .venv/bin/activate && python scripts/e2e_screenshots.py
MK

echo "Done 026.
Next:
  source .venv/bin/activate
  pip install -r requirements.txt -r requirements-dev.txt
  uvicorn app:app --reload --host 0.0.0.0 --port 8084
  make test
  make screenshots   # set SB_TOKEN env if your pages require auth
"
