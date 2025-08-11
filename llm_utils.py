import requests
import json
from config import settings

def ollama_summarize(text, prompt=None):
    """Generate a summary using Ollama"""
    if not text or not text.strip():
        return ""
    
    system_prompt = prompt or "Summarize this text concisely:"
    data = {
        "model": settings.ollama_model,
        "prompt": f"{system_prompt}\n\n{text}\n\nSummary:",
        "stream": False
    }
    
    try:
        resp = requests.post(settings.ollama_api_url, json=data, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            return result.get("response", "").strip()
    except Exception as e:
        print(f"Ollama error: {e}")
    return ""

def ollama_generate_title(text):
    """Generate a title for the given text"""
    if not text or not text.strip():
        return "Untitled"
    
    prompt = f"Generate a short title (max 10 words) for:\n{text[:500]}\n\nTitle:"
    
    try:
        resp = requests.post(
            settings.ollama_api_url,
            json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
            timeout=30
        )
        if resp.status_code == 200:
            result = resp.json()
            return result.get("response", "").strip() or "Untitled"
    except Exception as e:
        print(f"Ollama error: {e}")
    return "Untitled"

def ollama_suggest_tags(text):
    """Suggest tags for the given text"""
    if not text or not text.strip():
        return []
    
    prompt = f"Suggest 3-5 tags for:\n{text[:500]}\n\nTags (comma-separated):"
    
    try:
        resp = requests.post(
            settings.ollama_api_url,
            json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
            timeout=30
        )
        if resp.status_code == 200:
            result = resp.json()
            tags_str = result.get("response", "").strip()
            return [tag.strip() for tag in tags_str.split(",") if tag.strip()][:5]
    except Exception as e:
        print(f"Ollama error: {e}")
    return []
