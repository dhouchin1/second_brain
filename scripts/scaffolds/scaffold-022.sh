#!/usr/bin/env bash
# scripts/scaffolds/scaffold_022.sh
# Env-driven config (+ README refresh). Ollama is REQUIRED.
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p scripts/scaffolds docs/screenshots

# 1) .env.example
if [[ ! -f .env.example ]]; then
cat > .env.example <<'ENV'
# ---- Core paths ----
SB_DB_PATH=./second_brain.db
SB_AUDIO_DIR=./audio

# ---- Ollama (REQUIRED) ----
OLLAMA_HOST=http://localhost:11434
OLLAMA_SUMMARIZE_MODEL=mistral
OLLAMA_TITLE_MODEL=mistral
OLLAMA_TAG_MODEL=mistral
OLLAMA_TIMEOUT_SECS=60
OLLAMA_STREAM=0

# ---- Server ----
APP_HOST=0.0.0.0
APP_PORT=8084

# ---- Security ----
JWT_SECRET=super-secret-key
JWT_ALG=HS256
JWT_EXPIRE_MIN=30

# ---- Discord (optional, if bot enabled) ----
DISCORD_BOT_TOKEN=
DISCORD_BOT_PREFIX=!
DISCORD_ALLOWED_GUILDS=
DISCORD_FORWARD_URL=http://localhost:8084/capture
DISCORD_FORWARD_BEARER=

# ---- Dev login (cookie shim) ----
DEV_LOGIN_ENABLED=1
ENV
echo "• wrote .env.example"
fi

# 2) requirements entries
touch requirements.txt requirements-dev.txt
if ! grep -q "^python-dotenv" requirements.txt; then
  echo "python-dotenv>=1.0.1" >> requirements.txt
fi
if ! grep -q "^httpx" requirements.txt; then
  echo "httpx>=0.27.2" >> requirements.txt
fi

# 3) config.py (create or augment)
if [[ ! -f config.py ]]; then
  cat > config.py <<'PY'
from pydantic import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    base_dir: Path = Path(__file__).parent
    db_path: Path = Path("./second_brain.db")
    audio_dir: Path = Path("./audio")

    # Ollama (REQUIRED)
    ollama_host: str = "http://localhost:11434"
    ollama_summarize_model: str = "mistral"
    ollama_title_model: str = "mistral"
    ollama_tag_model: str = "mistral"
    ollama_timeout_secs: int = 60
    ollama_stream: int = 0

    # Server
    app_host: str = "0.0.0.0"
    app_port: int = 8084

    # Security
    jwt_secret: str = "super-secret-key"
    jwt_alg: str = "HS256"
    jwt_expire_min: int = 30

    # Dev login (cookie shim)
    dev_login_enabled: int = 1

    class Config:
        env_file = ".env"
        env_prefix = ""  # we use explicit names (SB_DB_PATH etc.)
        fields = {
            "db_path": {"env": ["SB_DB_PATH"]},
            "audio_dir": {"env": ["SB_AUDIO_DIR"]},
            "ollama_host": {"env": ["OLLAMA_HOST"]},
            "ollama_summarize_model": {"env": ["OLLAMA_SUMMARIZE_MODEL"]},
            "ollama_title_model": {"env": ["OLLAMA_TITLE_MODEL"]},
            "ollama_tag_model": {"env": ["OLLAMA_TAG_MODEL"]},
            "ollama_timeout_secs": {"env": ["OLLAMA_TIMEOUT_SECS"]},
            "ollama_stream": {"env": ["OLLAMA_STREAM"]},
            "app_host": {"env": ["APP_HOST"]},
            "app_port": {"env": ["APP_PORT"]},
            "jwt_secret": {"env": ["JWT_SECRET"]},
            "jwt_alg": {"env": ["JWT_ALG"]},
            "jwt_expire_min": {"env": ["JWT_EXPIRE_MIN"]},
            "dev_login_enabled": {"env": ["DEV_LOGIN_ENABLED"]},
        }

settings = Settings()
PY
  echo "• created config.py"
else
  # ensure .env loading via pydantic (noop if already present)
  if ! grep -q "BaseSettings" config.py; then
    bk config.py
    cat > config.py <<'PY'
from pydantic import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    base_dir: Path = Path(__file__).parent
    db_path: Path = Path("./second_brain.db")
    audio_dir: Path = Path("./audio")
    ollama_host: str = "http://localhost:11434"
    ollama_summarize_model: str = "mistral"
    ollama_title_model: str = "mistral"
    ollama_tag_model: str = "mistral"
    ollama_timeout_secs: int = 60
    ollama_stream: int = 0
    app_host: str = "0.0.0.0"
    app_port: int = 8084
    jwt_secret: str = "super-secret-key"
    jwt_alg: str = "HS256"
    jwt_expire_min: int = 30
    dev_login_enabled: int = 1
    class Config:
        env_file = ".env"
        env_prefix = ""
        fields = {
            "db_path": {"env": ["SB_DB_PATH"]},
            "audio_dir": {"env": ["SB_AUDIO_DIR"]},
            "ollama_host": {"env": ["OLLAMA_HOST"]},
            "ollama_summarize_model": {"env": ["OLLAMA_SUMMARIZE_MODEL"]},
            "ollama_title_model": {"env": ["OLLAMA_TITLE_MODEL"]},
            "ollama_tag_model": {"env": ["OLLAMA_TAG_MODEL"]},
            "ollama_timeout_secs": {"env": ["OLLAMA_TIMEOUT_SECS"]},
            "ollama_stream": {"env": ["OLLAMA_STREAM"]},
            "app_host": {"env": ["APP_HOST"]},
            "app_port": {"env": ["APP_PORT"]},
            "jwt_secret": {"env": ["JWT_SECRET"]},
            "jwt_alg": {"env": ["JWT_ALG"]},
            "jwt_expire_min": {"env": ["JWT_EXPIRE_MIN"]},
            "dev_login_enabled": {"env": ["DEV_LOGIN_ENABLED"]},
        }
settings = Settings()
PY
    echo "• upgraded config.py to env-driven settings"
  fi
fi

# 4) README refresh stub (only create if absent)
if [[ ! -f README.md ]]; then
cat > README.md <<'MD'
# Second Brain (FastAPI + SQLite + Ollama)

See docs in README—extended version provided via scaffolds (022–026).
Quick run:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --reload --host 0.0.0.0 --port 8084
MD
echo "• wrote minimal README.md"
fi

echo "Done 022.

Next:
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env # and edit if needed
"