#!/usr/bin/env bash
# scripts/scaffolds/scaffold_030.sh
# GitHub Action to run the app, get a token, take screenshots, upload artifact
set -euo pipefail

mkdir -p .github/workflows docs/screenshots

# Ensure Python Playwright script exists (from earlier scaffolds)
if [[ ! -f scripts/e2e_screenshots.py ]]; then
cat > scripts/e2e_screenshots.py <<'PY'
import os, asyncio
from playwright.async_api import async_playwright

BASE = os.environ.get("SB_BASE_URL","http://localhost:8084")
TOKEN = os.environ.get("SB_TOKEN","")

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context()
        if TOKEN:
            await ctx.set_extra_http_headers({"Authorization": f"Bearer {TOKEN}"})
        page = await ctx.new_page()
        await page.goto(BASE + "/")
        await page.screenshot(path="docs/screenshots/dashboard.png", full_page=True)
        await page.goto(BASE + "/about")
        await page.screenshot(path="docs/screenshots/about.png", full_page=True)
        await page.goto(BASE + "/exports")
        await page.screenshot(path="docs/screenshots/exports.png", full_page=True)
        await b.close()

if __name__ == "__main__":
    asyncio.run(main())
PY
fi

# Workflow
cat > .github/workflows/screenshots.yml <<'YML'
name: E2E Screenshots

on:
  workflow_dispatch:
  push:
    branches: ["**"]

jobs:
  screenshots:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt || true
          pip install playwright
          python -m playwright install chromium

      - name: Start app (background)
        env:
          APP_HOST: "0.0.0.0"
          APP_PORT: "8084"
          DEV_LOGIN_ENABLED: "1"
        run: |
          nohup python -m uvicorn app:app --host 0.0.0.0 --port 8084 > uvicorn.log 2>&1 &
          for i in {1..30}; do
            curl -fsS http://localhost:8084/healthz && break
            sleep 1
          done
          curl -fsS http://localhost:8084/healthz

      - name: Create user & get token
        run: |
          set -e
          curl -s -X POST -F "username=ci" -F "password=ci" http://localhost:8084/register || true
          TOKEN=$(curl -s -X POST -H "Content-Type: application/x-www-form-urlencoded" \
            -d "username=ci&password=ci" http://localhost:8084/token | python - <<'PY'
import sys, json
print(json.load(sys.stdin)['access_token'])
PY
          )
          echo "SB_TOKEN=$TOKEN" >> $GITHUB_ENV

      - name: Take screenshots
        env:
          SB_BASE_URL: "http://localhost:8084"
          SB_TOKEN: ${{ env.SB_TOKEN }}
        run: |
          python scripts/e2e_screenshots.py

      - name: Upload screenshots artifact
        uses: actions/upload-artifact@v4
        with:
          name: screenshots_${{ github.sha }}
          path: docs/screenshots/*.png
YML

echo "Done 030.

Push and check Actions → E2E Screenshots. Artifacts will include docs/screenshots/*.png
"
