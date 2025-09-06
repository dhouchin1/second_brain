#!/usr/bin/env python3
"""
Quick local Ollama diagnostics for Second Brain.

Checks:
- Server reachable at OLLAMA_URL (default http://localhost:11434)
- Lists available models (/api/tags)
- Verifies generate endpoint (/api/generate) for the configured model
- Verifies embeddings endpoint (/api/embeddings) for the configured embeddings model

Usage:
  python scripts/ollama_check.py --url http://localhost:11434          --gen-model llama3.2 --embed-model nomic-embed-text
"""
import argparse
import json
import sys
from urllib import request, error

def _post(url: str, payload: dict, timeout: int = 10):
    data = json.dumps(payload).encode('utf-8')
    req = request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), json.loads(resp.read().decode('utf-8'))

def _get(url: str, timeout: int = 5):
    with request.urlopen(url, timeout=timeout) as resp:
        return resp.getcode(), json.loads(resp.read().decode('utf-8'))

def main():
    ap = argparse.ArgumentParser(description='Ollama local diagnostics')
    ap.add_argument('--url', default='http://localhost:11434', help='Ollama base URL')
    ap.add_argument('--gen-model', default='llama3.2', help='Model for /api/generate')
    ap.add_argument('--embed-model', default='nomic-embed-text', help='Model for /api/embeddings')
    args = ap.parse_args()

    base = args.url.rstrip('/')
    ok = True

    print(f"Checking Ollama at {base} ...")
    # 1) Tags
    try:
        code, tags = _get(f"{base}/api/tags")
        print(f"- /api/tags: {code}; models: {[m.get('name') for m in tags.get('models', [])]}")
    except error.URLError as e:
        print(f"! Failed to reach Ollama: {e}")
        print("  - Ensure 'ollama serve' is running")
        return 2
    except Exception as e:
        print(f"! Error fetching tags: {e}")
        ok = False

    # 2) Generate
    try:
        code, out = _post(f"{base}/api/generate", {"model": args.gen_model, "prompt": "Say hello."})
        print(f"- /api/generate: {code}; ok: {bool(out)}")
    except error.HTTPError as e:
        print(f"! Generate failed ({e.code}): {e.reason}")
        print("  - Model may be missing; try: ollama pull", args.gen_model)
        ok = False
    except Exception as e:
        print(f"! Generate error: {e}")
        ok = False

    # 3) Embeddings
    try:
        code, out = _post(f"{base}/api/embeddings", {"model": args.embed_model, "input": "test embedding"})
        has = 'embedding' in out or (out.get('data') and out['data'][0].get('embedding')) or out.get('embeddings')
        print(f"- /api/embeddings: {code}; embedding returned: {bool(has)}")
    except error.HTTPError as e:
        print(f"! Embeddings failed ({e.code}): {e.reason}")
        print("  - Embedding model may be missing; try: ollama pull", args.embed_model)
        ok = False
    except Exception as e:
        print(f"! Embeddings error: {e}")
        ok = False

    if not ok:
        print("
Some checks failed. Tips:")
        print("- Start server: `ollama serve` (or the service on macOS)")
        print("- Pull models: `ollama pull", args.gen_model, "` and `ollama pull", args.embed_model, "`")
        print("- Verify firewall allows localhost:11434")
        return 1
    print("
All checks passed.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
