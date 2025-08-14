
---

# scripts/scaffolds/scaffold_023.sh — LLM utils (Ollama REQUIRED) + tests

```bash
#!/usr/bin/env bash
# scripts/scaffolds/scaffold_023.sh
# Ollama-required LLM utils, provider interface, unit tests.
set -euo pipefail
STAMP="$(date +%Y%m%d-%H%M%S)"; bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p tests/unit

# 1) requirements
touch requirements.txt
grep -q "^httpx" requirements.txt || echo "httpx>=0.27.2" >> requirements.txt

# 2) llm_utils.py (replace or create)
bk llm_utils.py
cat > llm_utils.py <<'PY'
from __future__ import annotations
import httpx, json, re
from typing import Dict, List
from config import settings

class LLMError(RuntimeError): ...

def _ollama_generate(prompt: str, model: str, temperature: float = 0.2, timeout: int | None = None) -> str:
    if not settings.ollama_host:
        raise LLMError("OLLAMA_HOST not configured")
    url = f"{settings.ollama_host.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "temperature": temperature, "stream": bool(settings.ollama_stream)}
    to = timeout or settings.ollama_timeout_secs
    with httpx.Client(timeout=to) as cli:
        r = cli.post(url, json=payload)
        r.raise_for_status()
        # Non-stream response shape
        data = r.json()
    return data.get("response", "").strip()

def ollama_generate_title(text: str) -> str:
    text = (text or "").strip()
    snippet = text[:1000]
    prompt = (
        "Write a short, human title (max 8 words) for the following content.\n"
        "Return ONLY the title text.\n\n"
        f"{snippet}"
    )
    raw = _ollama_generate(prompt, settings.ollama_title_model)
    # clean
    title = raw.strip().strip('"').strip("'").replace("\n", " ")
    return re.sub(r"\s+", " ", title)[:80]

def ollama_summarize(text: str) -> Dict:
    text = (text or "").strip()
    snippet = text[:3000]
    prompt = (
        "Summarize the content in 3–5 bullet points and propose 3–7 tags and optional action items.\n"
        "Return JSON with keys: summary (string), tags (array of strings), actions (array of strings).\n\n"
        f"{snippet}"
    )
    raw = _ollama_generate(prompt, settings.ollama_summarize_model, temperature=0.1)
    # try parse JSON; fallback to naive extraction
    try:
        data = json.loads(raw)
        if not isinstance(data, dict): raise ValueError()
    except Exception:
        # naive fallback
        bullets = [line.strip("-• ").strip() for line in raw.splitlines() if line.strip()]
        summary = " ".join(bullets[:5])[:600]
        tags = [w for w in re.findall(r"\#?([a-zA-Z][\w\-]{3,})", raw)][:7]
        actions = [b for b in bullets if b.lower().startswith(("todo", "action", "next"))][:5]
        data = {"summary": summary, "tags": tags, "actions": actions}
    # normalize tags
    tags = []
    for t in (data.get("tags") or []):
        t = str(t).strip().lower().replace("#","").replace(" ", "-")
        if t and t not in tags: tags.append(t)
    data["tags"] = tags
    data["actions"] = [str(a).strip() for a in (data.get("actions") or []) if a and len(str(a).strip())<160][:10]
    data["summary"] = str(data.get("summary","")).strip()[:1000]
    return data
PY
echo "• wrote llm_utils.py"

# 3) unit test
cat > tests/unit/test_llm_utils.py <<'PY'
import types
import llm_utils as U

def test_title_cleaning(monkeypatch):
    def fake_gen(prompt, model, temperature=0.2, timeout=None):
        return '"A Test Title"'
    monkeypatch.setattr(U, "_ollama_generate", fake_gen)
    assert U.ollama_generate_title("some content") == "A Test Title"

def test_summarize_fallback(monkeypatch):
    def fake_gen(prompt, model, temperature=0.2, timeout=None):
        return "- todo: try this\n- Key insight one\n- Tag: research-notes"
    monkeypatch.setattr(U, "_ollama_generate", fake_gen)
    d = U.ollama_summarize("x")
    assert "summary" in d and "tags" in d and "actions" in d
    assert any("todo" in a.lower() for a in d["actions"])
PY

echo "Done 023. Run: source .venv/bin/activate && pip install -r requirements.txt && pytest -q tests/unit"
