# Makefile for Second Brain
PY ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python

ENV ?= .env

.PHONY: venv install run-app run-bot export-json export-mdzip import-json import-md embeddings-rebuild fmt test clean

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt

run-app:
	bash -lc 'set -a; [ -f $(ENV) ] && . $(ENV); set +a; $(PYBIN) -m uvicorn app:app --reload --host 0.0.0.0 --port 8084'

run-bot:
	bash -lc 'set -a; [ -f $(ENV) ] && . $(ENV); set +a; $(PYBIN) discord_bot.py'

export-json: install
	@mkdir -p out
	$(PYBIN) sb.py export-json out/second_brain_export.json

export-mdzip: install
	@mkdir -p out
	$(PYBIN) sb.py export-mdzip out/second_brain_markdown.zip

# Usage: make import-json FILE=path/to/export.json
import-json: install
	@if [ -z "$(FILE)" ]; then echo "FILE= required"; exit 1; fi
	$(PYBIN) sb.py import-json "$(FILE)"

# Usage: make import-md FILE=dir/or.zip
import-md: install
	@if [ -z "$(FILE)" ]; then echo "FILE= required"; exit 1; fi
	$(PYBIN) sb.py import-md "$(FILE)"

# Usage: make embeddings-rebuild N=200 FORCE=1
embeddings-rebuild: install
	$(PYBIN) sb.py rebuild-embeddings --limit $${N:-100} $$( [ "$${FORCE:-0}" = "1" ] && echo --force || true )

fmt:
	$(PIP) install -q black==24.4.2
	$(VENV)/bin/black app.py discord_bot.py sb.py

test:
	$(PYBIN) -m compileall .

clean:
	rm -rf out __pycache__ */__pycache__
