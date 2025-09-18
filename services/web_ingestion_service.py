"""
Web Content Ingestion Service

Intelligent web content extraction using Playwright with AI-powered processing.
Integrates with Smart Automation system for seamless URL-to-knowledge conversion.
"""

from __future__ import annotations
import asyncio
import base64
import json
import os
import re
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from urllib.parse import urlparse, urljoin
import uuid

try:
    from playwright.async_api import async_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("[web_ingestion] Playwright not available. Install with: pip install playwright")

import sqlite3

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel

from config import settings
from llm_utils import ollama_summarize, ollama_generate_title

try:
    import redis.asyncio as redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None


@dataclass
class IngestionArtifact:
    """Represents a captured artifact (pdf/html/media/etc.)."""
    type: str
    path: str
    size: int
    mime_type: Optional[str] = None
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class WebContent:
    """Extracted web content data"""
    url: str
    title: str
    content: str
    summary: str
    metadata: Dict[str, Any]
    screenshot_path: Optional[str] = None
    extracted_at: datetime = None
    content_hash: Optional[str] = None
    artifacts: List[IngestionArtifact] = field(default_factory=list)


def _truncate_content(content: str, max_length: int) -> Tuple[str, bool]:
    """Limit content length, returning truncated content and flag."""
    if max_length and len(content) > max_length:
        truncated = content[:max_length].rstrip()
        return truncated + "\n\n...[content truncated]", True
    return content, False


@dataclass
class ExtractionConfig:
    """Configuration for web content extraction"""
    capture_screenshot: bool = True
    capture_pdf: bool = False
    capture_html: bool = True
    download_original: bool = False
    download_media: bool = False
    fetch_captions: bool = False
    extract_images: bool = False
    follow_redirects: bool = True
    timeout: int = 30
    viewport_width: int = 1920
    viewport_height: int = 1080
    user_agent: str = "Second Brain Web Ingestion Bot 1.0"
    block_ads: bool = True
    extract_links: bool = True
    max_content_length: int = 50000

    def override(self, **kwargs) -> "ExtractionConfig":
        data = asdict(self)
        for key, value in kwargs.items():
            if value is not None and key in data:
                data[key] = value
        return ExtractionConfig(**data)

    @classmethod
    def from_settings(cls) -> "ExtractionConfig":
        return cls(
            capture_screenshot=settings.web_capture_screenshot_default,
            capture_pdf=settings.web_capture_pdf_default,
            capture_html=settings.web_capture_html_default,
            download_original=settings.web_download_original_default,
            download_media=settings.web_download_media_default,
            fetch_captions=settings.web_fetch_captions_default,
            extract_images=settings.web_extract_images_default,
            timeout=settings.web_timeout_default,
        )


class IngestionJobQueue:
    """Simple Redis-backed queue for ingestion jobs."""

    def __init__(self):
        self._client = None
        if redis and settings.redis_url:
            try:
                self._client = redis.from_url(settings.redis_url, decode_responses=False)
            except Exception as exc:
                print(f"[web_ingestion] Failed to connect to Redis: {exc}")
                self._client = None

    async def enqueue(self, key: str, payload: Dict[str, Any]):
        if not self._client:
            return
        try:
            await self._client.lpush(key, json.dumps(payload).encode("utf-8"))
        except Exception as exc:
            print(f"[web_ingestion] Redis enqueue failed: {exc}")

    async def dequeue(self, key: str, block: bool = True, timeout: int = 5) -> Optional[Dict[str, Any]]:
        if not self._client:
            await asyncio.sleep(timeout if block else 0)
            return None
        try:
            if block:
                result = await self._client.brpop(key, timeout)
            else:
                result = await self._client.rpop(key)
        except Exception as exc:
            print(f"[web_ingestion] Redis dequeue failed: {exc}")
            return None

        if not result:
            return None

        value = result[1] if isinstance(result, (list, tuple)) else result
        try:
            return json.loads(value.decode("utf-8"))
        except Exception as exc:
            print(f"[web_ingestion] Failed to decode job payload: {exc}")
            return None


class DomainHandler:
    """Base class for domain-specific ingestion handlers."""

    domains: Tuple[str, ...] = tuple()

    def matches(self, parsed_url) -> bool:
        return parsed_url.netloc.lower() in self.domains

    async def fetch(self, url: str, config: ExtractionConfig) -> Optional[WebContent]:
        raise NotImplementedError


class GitHubRepoHandler(DomainHandler):
    """Specialised handler for GitHub repositories."""

    domains = ("github.com", "www.github.com")
    API_BASE = "https://api.github.com"
    RAW_BASE = "https://raw.githubusercontent.com"

    def __init__(self):
        token = os.getenv("GITHUB_TOKEN")
        self._auth_headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "SecondBrain-WebIngestion/1.0"
        }
        if token:
            self._auth_headers["Authorization"] = f"Bearer {token}"

    def matches(self, parsed_url) -> bool:
        if not super().matches(parsed_url):
            return False
        parts = [p for p in parsed_url.path.strip('/').split('/') if p]
        return len(parts) >= 2

    async def fetch(self, url: str, config: ExtractionConfig) -> Optional[WebContent]:
        parsed = urlparse(url)
        parts = [p for p in parsed.path.strip('/').split('/') if p]
        if len(parts) < 2:
            return None

        owner, repo = parts[0], parts[1].replace('.git', '')

        # Ignore non-repository views (issues, pull requests, etc.) to allow fallback to generic extractor
        if len(parts) >= 3 and parts[2] in {"issues", "pull", "discussions", "actions", "projects"}:
            return None

        repo_data = await self._fetch_repo_metadata(owner, repo)
        if not repo_data:
            return None

        default_branch = repo_data.get("default_branch", "main")
        readme = await self._fetch_readme(owner, repo, default_branch)
        supplemental_files = await self._fetch_supplemental_files(owner, repo, default_branch)
        top_issues = await self._fetch_top_issues(owner, repo)
        languages = await self._fetch_languages(owner, repo)

        sections: List[str] = []
        sections.append(self._format_repo_snapshot(repo_data, languages))

        if readme:
            sections.append("## README\n" + readme.strip())

        for name, text in supplemental_files:
            sections.append(f"## {name}\n{text.strip()}")

        if top_issues:
            issue_lines = [
                f"- {issue['title']} (#{issue['number']}) — {issue['html_url']}"
                for issue in top_issues
            ]
            sections.append("## Recent Issues\n" + "\n".join(issue_lines))

        combined_content = "\n\n".join(section for section in sections if section)
        combined_content, truncated = _truncate_content(combined_content, config.max_content_length)

        summary = repo_data.get("description") or (readme[:200] if readme else "")
        metadata = {
            "handler": "github_repository",
            "source_url": url,
            "domain": parsed.netloc,
            "repo": {
                "full_name": repo_data.get("full_name"),
                "visibility": repo_data.get("visibility"),
                "stars": repo_data.get("stargazers_count"),
                "forks": repo_data.get("forks_count"),
                "watchers": repo_data.get("subscribers_count"),
                "open_issues": repo_data.get("open_issues_count"),
                "license": (repo_data.get("license") or {}).get("name"),
                "topics": repo_data.get("topics", []),
                "homepage": repo_data.get("homepage"),
                "default_branch": default_branch,
                "last_pushed_at": repo_data.get("pushed_at"),
            },
            "languages": languages,
            "supplemental_files": [name for name, _ in supplemental_files],
            "issues": [
                {
                    "title": issue["title"],
                    "number": issue["number"],
                    "url": issue["html_url"],
                    "created_at": issue.get("created_at"),
                    "labels": [label.get("name") for label in issue.get("labels", [])]
                }
                for issue in top_issues
            ],
            "content_truncated": truncated,
            "artifacts": [],
        }

        content_hash = hashlib.sha256((repo_data.get("full_name", "") + combined_content).encode()).hexdigest()[:16]

        return WebContent(
            url=url,
            title=repo_data.get("full_name", f"GitHub Repository: {owner}/{repo}"),
            content=combined_content,
            summary=summary or "GitHub repository overview",
            metadata=metadata,
            screenshot_path=None,
            extracted_at=datetime.now(),
            content_hash=content_hash,
        )

    async def _fetch_repo_metadata(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        endpoint = f"{self.API_BASE}/repos/{owner}/{repo}"
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(endpoint, headers=self._auth_headers)
            if response.status_code != 200:
                return None
            return response.json()

    async def _fetch_readme(self, owner: str, repo: str, branch: str) -> str:
        candidates = ["README.md", "README.MD", "README"]
        async with httpx.AsyncClient(timeout=20.0) as client:
            for candidate in candidates:
                raw_url = f"{self.RAW_BASE}/{owner}/{repo}/{branch}/{candidate}"
                response = await client.get(raw_url, headers={"User-Agent": "SecondBrain-WebIngestion/1.0"})
                if response.status_code == 200 and response.text.strip():
                    return response.text
        return ""

    async def _fetch_supplemental_files(self, owner: str, repo: str, branch: str) -> List[Tuple[str, str]]:
        supplemental_names = [
            "CONTRIBUTING.md",
            "CHANGELOG.md",
            "docs/overview.md",
            "docs/INTRODUCTION.md",
            "docs/index.md",
        ]
        found: List[Tuple[str, str]] = []
        async with httpx.AsyncClient(timeout=20.0) as client:
            for filename in supplemental_names:
                raw_url = f"{self.RAW_BASE}/{owner}/{repo}/{branch}/{filename}"
                response = await client.get(raw_url, headers={"User-Agent": "SecondBrain-WebIngestion/1.0"})
                if response.status_code == 200 and response.text.strip():
                    found.append((filename, response.text))
        return found

    async def _fetch_top_issues(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        endpoint = f"{self.API_BASE}/repos/{owner}/{repo}/issues"
        params = {"state": "open", "per_page": 3, "labels": ""}
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(endpoint, headers=self._auth_headers, params=params)
            if response.status_code != 200:
                return []
            issues = response.json()
            # Filter out pull requests (issues endpoint returns PRs too)
            return [issue for issue in issues if "pull_request" not in issue][:3]

    async def _fetch_languages(self, owner: str, repo: str) -> Dict[str, int]:
        endpoint = f"{self.API_BASE}/repos/{owner}/{repo}/languages"
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(endpoint, headers=self._auth_headers)
            if response.status_code != 200:
                return {}
            return response.json()

def _format_repo_snapshot(self, repo: Dict[str, Any], languages: Dict[str, int]) -> str:
        lines = [f"# {repo.get('full_name', repo.get('name', 'GitHub Repository'))}"]
        description = repo.get("description")
        if description:
            lines.append(description.strip())

        stats = [
            f"Stars: {repo.get('stargazers_count', 0)}",
            f"Forks: {repo.get('forks_count', 0)}",
            f"Watchers: {repo.get('subscribers_count', 0)}",
            f"Open Issues: {repo.get('open_issues_count', 0)}",
        ]
        if repo.get("license"):
            license_name = (repo["license"] or {}).get("name")
            if license_name:
                stats.append(f"License: {license_name}")

        if repo.get("topics"):
            stats.append("Topics: " + ", ".join(repo["topics"]))

        if languages:
            top_languages = sorted(languages.items(), key=lambda item: item[1], reverse=True)[:3]
            stats.append("Languages: " + ", ".join(lang for lang, _ in top_languages))

        lines.append("\n".join(stats))
        return "\n\n".join(filter(None, lines))


class MediumArticleHandler(DomainHandler):
    """Handler for Medium-hosted content."""

    domains = ("medium.com", "www.medium.com")

    def __init__(self):
        cookie = os.getenv("MEDIUM_SESSION_COOKIE")
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                           " AppleWebKit/537.36 (KHTML, like Gecko)"
                           " Chrome/123.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        self._cookies = {"sid": cookie} if cookie else {}

    def matches(self, parsed_url) -> bool:
        host = parsed_url.netloc.lower()
        if host.endswith(".medium.com") or host in self.domains:
            return True
        return "/@" in parsed_url.path

    async def fetch(self, url: str, config: ExtractionConfig) -> Optional[WebContent]:
        async with httpx.AsyncClient(headers=self._headers, cookies=self._cookies, timeout=config.timeout) as client:
            response = await client.get(url, follow_redirects=True)
            if response.status_code >= 400:
                raise RuntimeError(f"Medium request failed: HTTP {response.status_code}")
            html = response.text

        state = self._extract_state(html)
        if state:
            content, metadata = self._build_from_state(state)
        else:
            content, metadata = self._fallback_parse(html)

        truncated_content, truncated = _truncate_content(content, config.max_content_length)
        metadata.setdefault("handler", "medium_article")
        metadata.setdefault("source_url", url)
        metadata["content_truncated"] = truncated

        artifact = self._save_html_snapshot(html)
        artifacts = [artifact] if artifact else []

        summary = metadata.get("subtitle") or metadata.get("dek") or truncated_content[:240]
        content_hash = hashlib.sha256((url + truncated_content).encode()).hexdigest()[:16]

        return WebContent(
            url=url,
            title=metadata.get("title") or metadata.get("headline") or metadata.get("source_url") or "Medium Article",
            content=truncated_content,
            summary=summary,
            metadata=metadata,
            screenshot_path=None,
            extracted_at=datetime.now(),
            content_hash=content_hash,
            artifacts=artifacts,
        )

    def _extract_state(self, html: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"window\.__APOLLO_STATE__\s*=\s*({.*?})</script>", html, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

    def _build_from_state(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        post = next((value for key, value in state.items() if key.startswith("Post:") and isinstance(value, dict)), None)
        if not post:
            raise RuntimeError("Medium state missing Post entry")

        paragraphs = post.get("content", {}).get("bodyModel", {}).get("paragraphs", [])
        parts: List[str] = []
        for paragraph in paragraphs:
            text = paragraph.get("text", "").strip()
            if not text:
                continue
            p_type = paragraph.get("type")
            if p_type == "H2":
                parts.append(f"## {text}")
            elif p_type == "H3":
                parts.append(f"### {text}")
            else:
                parts.append(text)

        metadata = {
            "title": post.get("title"),
            "subtitle": post.get("content", {}).get("subtitle"),
            "dek": post.get("content", {}).get("dek"),
            "reading_time": post.get("virtuals", {}).get("readingTime"),
            "word_count": post.get("virtuals", {}).get("wordCount"),
            "claps": post.get("virtuals", {}).get("totalClapCount"),
            "tags": [tag.get("name") for tag in post.get("virtuals", {}).get("tags", []) if tag.get("name")],
            "published_at": post.get("firstPublishedAt"),
            "updated_at": post.get("latestPublishedAt"),
        }

        author_id = post.get("creatorId")
        author = state.get(author_id) if author_id else {}
        metadata["author"] = author.get("name")
        metadata["author_bio"] = author.get("bio")

        return "\n\n".join(parts), metadata

    def _fallback_parse(self, html: str) -> Tuple[str, Dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.text.strip() if soup.title else "Medium Article"
        article = soup.find("article")
        content = article.get_text("\n") if article else soup.get_text("\n")
        return content, {"title": title}

    def _save_html_snapshot(self, html: str) -> Optional[IngestionArtifact]:
        snapshots_dir = Path(settings.snapshots_dir) / "medium"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        path = snapshots_dir / f"medium_{uuid.uuid4().hex}.html"
        try:
            path.write_text(html, encoding="utf-8")
            return IngestionArtifact(
                type="html",
                path=str(path),
                size=path.stat().st_size,
                mime_type="text/html",
                label="Medium HTML snapshot",
            )
        except Exception:
            return None


DOMAIN_HANDLERS: List[DomainHandler] = [
    GitHubRepoHandler(),
    MediumArticleHandler(),
]


class WebContentExtractor:
    """Playwright-based web content extraction"""
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.playwright = None

    async def __aenter__(self):
        """Async context manager entry"""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not available")
            
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-extensions',
                '--disable-plugins',
                '--disable-images' if not True else '',  # Can be configured
            ]
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def extract_content(self, url: str, config: ExtractionConfig = None) -> WebContent:
        """Extract content from a web page"""
        config = config or ExtractionConfig()
        
        page = await self.browser.new_page(
            viewport={'width': config.viewport_width, 'height': config.viewport_height},
            user_agent=config.user_agent
        )
        
        try:
            # Block ads and trackers if requested
            if config.block_ads:
                await page.route("**/*", self._block_ads_handler)
            
            # Navigate to page
            response = await page.goto(
                url, 
                wait_until='networkidle',
                timeout=config.timeout * 1000
            )
            
            if not response or response.status >= 400:
                raise Exception(f"Failed to load page: HTTP {response.status if response else 'No response'}")
            
            # Get final URL after redirects
            final_url = page.url
            
            # Extract basic metadata
            title = await page.title() or ""
            
            # Try to get meta description
            description_element = await page.query_selector('meta[name="description"]')
            description = ""
            if description_element:
                description = await description_element.get_attribute('content') or ""
            
            artifacts: List[IngestionArtifact] = []

            # Extract main content using multiple strategies
            content = await self._extract_main_content(page)
            content, truncated = _truncate_content(content, config.max_content_length)
            
            # Extract additional metadata
            metadata = await self._extract_metadata(page, final_url, config, truncated)

            # Optionally persist original response body
            if config.download_original and response:
                try:
                    body_bytes = await response.body()
                    content_type = response.headers.get("content-type", "") if response.headers else ""
                    extension = ".bin"
                    if "html" in content_type:
                        extension = ".html"
                    elif "json" in content_type:
                        extension = ".json"
                    elif "text" in content_type:
                        extension = ".txt"
                    original_path = self._artifact_dir("original") / f"{uuid.uuid4().hex}{extension}"
                    original_path.write_bytes(body_bytes)
                    artifacts.append(
                        IngestionArtifact(
                            type="original",
                            path=str(original_path),
                            size=original_path.stat().st_size,
                            mime_type=content_type.split(";")[0] if content_type else None,
                            label="Original response",
                            metadata={"headers": dict(response.headers) if response.headers else {}}
                        )
                    )
                except Exception as exc:
                    print(f"[web_ingestion] Original download failed: {exc}")

            # Capture HTML snapshot if enabled
            if config.capture_html:
                try:
                    html_content = await page.content()
                    html_path, html_size = self._save_text_artifact(html_content, "html", ".html")
                    artifacts.append(
                        IngestionArtifact(
                            type="html",
                            path=html_path,
                            size=html_size,
                            mime_type="text/html",
                            label="HTML snapshot"
                        )
                    )
                except Exception as exc:
                    print(f"[web_ingestion] HTML capture failed: {exc}")

            # Capture PDF snapshot if enabled
            if config.capture_pdf:
                try:
                    pdf_path = self._artifact_dir("pdf") / f"{uuid.uuid4().hex}.pdf"
                    await page.pdf(path=str(pdf_path), format="A4", print_background=True)
                    artifacts.append(
                        IngestionArtifact(
                            type="pdf",
                            path=str(pdf_path),
                            size=pdf_path.stat().st_size,
                            mime_type="application/pdf",
                            label="PDF snapshot"
                        )
                    )
                except Exception as exc:
                    print(f"[web_ingestion] PDF capture failed: {exc}")

            # Take screenshot if requested
            screenshot_path = None
            if config.capture_screenshot:
                screenshot_path = await self._take_screenshot(page, final_url)
                if screenshot_path:
                    try:
                        artifacts.append(
                            IngestionArtifact(
                                type="image",
                                path=screenshot_path,
                                size=Path(screenshot_path).stat().st_size,
                                mime_type="image/png",
                                label="Screenshot"
                            )
                        )
                    except FileNotFoundError:
                        pass

            # Generate content hash
            content_hash = hashlib.sha256((title + content).encode()).hexdigest()[:16]
            
            metadata["artifacts"] = [asdict(artifact) for artifact in artifacts]

            return WebContent(
                url=final_url,
                title=title,
                content=content,
                summary=description,
                metadata=metadata,
                screenshot_path=screenshot_path,
                extracted_at=datetime.now(),
                content_hash=content_hash,
                artifacts=artifacts
            )
            
        finally:
            await page.close()
    
    async def _block_ads_handler(self, route):
        """Block ads and tracking requests"""
        url = route.request.url
        
        # Block known ad/tracking domains and resources
        blocked_patterns = [
            'googletagmanager.com',
            'google-analytics.com',
            'googlesyndication.com',
            'doubleclick.net',
            'facebook.com/tr',
            'twitter.com/i/adsct',
            'ads',
            'analytics',
            'tracking',
            '.gif',
            'pixel'
        ]
        
        if any(pattern in url.lower() for pattern in blocked_patterns):
            await route.abort()
        else:
            await route.continue_()

    def _artifact_dir(self, name: str) -> Path:
        directory = Path(settings.snapshots_dir) / name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _save_text_artifact(self, content: str, subdir: str, suffix: str) -> Tuple[str, int]:
        path = self._artifact_dir(subdir) / f"{uuid.uuid4().hex}{suffix}"
        path.write_text(content, encoding="utf-8")
        return str(path), path.stat().st_size
    
    async def _extract_main_content(self, page: Page) -> str:
        """Extract main content using multiple strategies"""
        content_selectors = [
            'article',
            '[role="main"]',
            'main',
            '.content',
            '.post-content',
            '.entry-content',
            '.article-content',
            '.story-body',
            '.post-body',
            '#content',
            '.container .row .col'
        ]
        
        # Try structured content extraction first
        for selector in content_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    content = await element.inner_text()
                    if len(content.strip()) > 200:  # Minimum content length
                        return self._clean_content(content)
            except:
                continue
        
        # Fallback: extract from body but try to avoid navigation/sidebar content
        try:
            # Remove navigation, sidebar, and footer elements
            await page.evaluate("""
                () => {
                    const selectorsToRemove = [
                        'nav', 'header', 'footer', '.nav', '.navigation', 
                        '.sidebar', '.menu', '.ads', '.advertisement',
                        '.social', '.comments', '.related', '.recommendations'
                    ];
                    selectorsToRemove.forEach(selector => {
                        document.querySelectorAll(selector).forEach(el => el.remove());
                    });
                }
            """)
            
            # Get remaining body content
            body_content = await page.evaluate("document.body.innerText")
            return self._clean_content(body_content)
            
        except:
            # Ultimate fallback
            return await page.inner_text('body') or ""
    
    def _clean_content(self, content: str) -> str:
        """Clean extracted content"""
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        content = content.strip()
        
        # Remove common footer/header patterns
        patterns_to_remove = [
            r'Cookie[s]?\s+Policy.*',
            r'Privacy\s+Policy.*',
            r'Terms\s+of\s+Service.*',
            r'Subscribe\s+to.*',
            r'Follow\s+us.*',
            r'Share\s+this.*',
        ]
        
        for pattern in patterns_to_remove:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content
    
    async def _extract_metadata(self, page: Page, url: str, config: ExtractionConfig, truncated: bool) -> Dict[str, Any]:
        """Extract metadata from the page"""
        metadata = {
            'domain': urlparse(url).netloc,
            'extracted_at': datetime.now().isoformat(),
            'final_url': url,
            'content_truncated': truncated,
        }
        
        # Extract Open Graph metadata
        og_tags = await page.evaluate("""
            () => {
                const og = {};
                const ogTags = document.querySelectorAll('meta[property^="og:"]');
                ogTags.forEach(tag => {
                    const property = tag.getAttribute('property');
                    const content = tag.getAttribute('content');
                    if (property && content) {
                        og[property.replace('og:', '')] = content;
                    }
                });
                return og;
            }
        """)
        
        metadata['open_graph'] = og_tags
        
        # Extract other meta tags
        meta_tags = await page.evaluate("""
            () => {
                const meta = {};
                const metaTags = document.querySelectorAll('meta[name]');
                metaTags.forEach(tag => {
                    const name = tag.getAttribute('name');
                    const content = tag.getAttribute('content');
                    if (name && content) {
                        meta[name] = content;
                    }
                });
                return meta;
            }
        """)
        
        metadata['meta_tags'] = meta_tags

        # Extract author information
        author_selectors = [
            '.author',
            '.byline',
            '[rel="author"]',
            '.post-author',
            '.article-author'
        ]
        
        for selector in author_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    author = await element.inner_text()
                    if author:
                        metadata['author'] = author.strip()
                        break
            except:
                continue
        
        # Extract publish date
        date_selectors = [
            'time[datetime]',
            '.published',
            '.post-date',
            '.article-date',
            '.date'
        ]
        
        for selector in date_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    # Try datetime attribute first
                    date_attr = await element.get_attribute('datetime')
                    if date_attr:
                        metadata['published_date'] = date_attr
                        break
                    # Fallback to text content
                    date_text = await element.inner_text()
                    if date_text:
                        metadata['published_date'] = date_text.strip()
                        break
            except:
                continue

        if config.extract_links:
            metadata['links'] = await self._extract_links(page, url)

        return metadata

    async def _extract_links(self, page: Page, url: str) -> Dict[str, List[Dict[str, str]]]:
        """Collect a snapshot of internal/external links for context."""
        try:
            raw_links = await page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                    href: a.href,
                    text: (a.textContent || '').trim()
                }))
            """)
        except Exception:
            return {'internal': [], 'external': []}

        domain = urlparse(url).netloc
        seen_internal = set()
        seen_external = set()
        internal_links: List[Dict[str, str]] = []
        external_links: List[Dict[str, str]] = []

        for link in raw_links:
            href = (link.get('href') or '').strip()
            if not href or href.startswith('mailto:') or href.startswith('javascript:'):
                continue

            parsed = urlparse(href)
            if not parsed.netloc:
                # Relative link; resolve against base
                href = urljoin(url, href)
                parsed = urlparse(href)

            entry = {
                'url': href,
                'text': link.get('text')[:120]
            }

            if parsed.netloc == domain:
                if href not in seen_internal and len(internal_links) < 15:
                    internal_links.append(entry)
                    seen_internal.add(href)
            else:
                if href not in seen_external and len(external_links) < 15:
                    external_links.append(entry)
                    seen_external.add(href)

        return {'internal': internal_links, 'external': external_links}
    
    async def _take_screenshot(self, page: Page, url: str) -> Optional[str]:
        """Take a screenshot of the page"""
        try:
            screenshots_dir = self._artifact_dir("screenshots")
            url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"web_{timestamp}_{url_hash}.png"
            screenshot_path = screenshots_dir / filename

            await page.screenshot(
                path=str(screenshot_path),
                full_page=True,
                type='png'
            )

            return str(screenshot_path)
            
        except Exception as e:
            print(f"[web_ingestion] Screenshot failed: {e}")
            return None


class WebIngestionService:
    """Main web content ingestion service"""
    
    def __init__(self, get_conn_func: Callable[[], sqlite3.Connection]):
        self.get_conn = get_conn_func
        self.default_config = ExtractionConfig.from_settings()
        self.job_queue = IngestionJobQueue()
        self._ensure_job_table()

    def _build_job_payload(self, url: str, user_id: int, note_id: Optional[int], config: ExtractionConfig) -> Dict[str, Any]:
        return {
            "job_id": uuid.uuid4().hex,
            "url": url,
            "user_id": user_id,
            "note_id": note_id,
            "config": asdict(config),
            "created_at": datetime.now().isoformat(),
            "status": "queued"
        }

    def _determine_async(self, async_mode: Optional[bool]) -> bool:
        if async_mode is not None:
            return async_mode
        return settings.web_async_ingestion_default

    async def _execute_ingestion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = payload["url"]
        user_id = payload["user_id"]
        note_id = payload.get("note_id")
        config = ExtractionConfig(**payload.get("config", {}))

        parsed = urlparse(url)
        handler = next((h for h in DOMAIN_HANDLERS if h.matches(parsed)), None)
        web_content: Optional[WebContent] = None
        handler_error: Optional[str] = None

        if handler:
            try:
                web_content = await handler.fetch(url, config)
            except Exception as handler_exc:
                handler_error = str(handler_exc)
                web_content = None

        if web_content is None:
            if not PLAYWRIGHT_AVAILABLE:
                raise RuntimeError(handler_error or "Playwright not available. Install with: pip install playwright")

            async with WebContentExtractor() as extractor:
                try:
                    web_content = await extractor.extract_content(url, config)
                except Exception as extractor_exc:
                    error_message = str(extractor_exc)
                    if handler_error:
                        error_message = f"{handler_error}; fallback failed: {extractor_exc}"
                    raise RuntimeError(error_message) from extractor_exc

        ai_results = await self._process_with_ai(web_content)
        note_id_created, file_metadata = await self._store_content(
            web_content, ai_results, user_id, note_id, config
        )

        return {
            "success": True,
            "note_id": note_id_created,
            "title": web_content.title,
            "content_length": len(web_content.content),
            "content_preview": web_content.content[:500],
            "summary": ai_results.get("summary", ""),
            "tags": ai_results.get("tags", []),
            "screenshot_path": web_content.screenshot_path,
            "metadata": file_metadata,
            "queued": False,
            "job_id": payload.get("job_id")
        }

    async def ingest_url(self, url: str, user_id: int = 1, note_id: Optional[int] = None, 
                        config: ExtractionConfig = None, async_mode: Optional[bool] = None) -> Dict[str, Any]:
        """Ingest content from a URL."""
        config = (config or self.default_config).override()
        job_payload = self._build_job_payload(url, user_id, note_id, config)
        self._record_job(job_payload)

        use_async = self._determine_async(async_mode) and self.job_queue and self.job_queue._client

        if use_async:
            await self.job_queue.enqueue("web_ingestion:jobs", job_payload)
            return {
                "success": True,
                "queued": True,
                "job_id": job_payload["job_id"],
                "note_id": None,
                "title": None,
                "content_length": None,
                "summary": None,
                "tags": [],
                "screenshot_path": None,
                "metadata": None
            }

        return await self._run_ingestion_job(job_payload)

    async def process_job_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._run_ingestion_job(payload)

    async def _run_ingestion_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        job_id = payload.get("job_id")
        self._update_job(job_id, status="processing", started_at=datetime.now().isoformat())
        try:
            result = await self._execute_ingestion(payload)
            completion_payload = {
                **payload,
                "status": "completed",
                "note_id": result.get("note_id"),
                "title": result.get("title"),
                "completed_at": datetime.now().isoformat()
            }
            await self.job_queue.enqueue("web_ingestion:completed", completion_payload)
            self._update_job(
                job_id,
                status="completed",
                completed_at=completion_payload["completed_at"],
                note_id=result.get("note_id"),
                title=result.get("title")
            )
            return result
        except Exception as exc:
            failure_payload = {
                **payload,
                "status": "failed",
                "error": str(exc),
                "failed_at": datetime.now().isoformat()
            }
            await self.job_queue.enqueue("web_ingestion:completed", failure_payload)
            self._update_job(
                job_id,
                status="failed",
                completed_at=failure_payload["failed_at"],
                error=str(exc)
            )
            raise

    def get_default_config(self) -> ExtractionConfig:
        return self.default_config

    def _ensure_job_table(self) -> None:
        conn = self.get_conn()
        try:
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS web_ingestion_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE,
                    user_id INTEGER,
                    url TEXT,
                    note_id INTEGER,
                    title TEXT,
                    status TEXT,
                    error TEXT,
                    payload TEXT,
                    created_at TEXT,
                    started_at TEXT,
                    completed_at TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _record_job(self, payload: Dict[str, Any]) -> None:
        conn = self.get_conn()
        try:
            c = conn.cursor()
            c.execute(
                """
                INSERT OR REPLACE INTO web_ingestion_jobs
                (job_id, user_id, url, note_id, title, status, error, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.get("job_id"),
                    payload.get("user_id"),
                    payload.get("url"),
                    payload.get("note_id"),
                    payload.get("title"),
                    payload.get("status", "queued"),
                    None,
                    json.dumps(payload),
                    payload.get("created_at") or datetime.now().isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _update_job(self, job_id: Optional[str], **fields) -> None:
        if not job_id:
            return
        allowed = {"status", "note_id", "title", "error", "started_at", "completed_at"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return

        set_clause = ", ".join(f"{key} = ?" for key in updates.keys())
        values = list(updates.values())
        values.append(job_id)

        conn = self.get_conn()
        try:
            c = conn.cursor()
            c.execute(f"UPDATE web_ingestion_jobs SET {set_clause} WHERE job_id = ?", values)
            conn.commit()
        finally:
            conn.close()

    def list_jobs(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self.get_conn()
        try:
            c = conn.cursor()
            rows = c.execute(
                """
                SELECT job_id, url, status, note_id, title, error, created_at, started_at, completed_at
                FROM web_ingestion_jobs
                WHERE user_id = ?
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            columns = [desc[0] for desc in c.description]
            return [dict(zip(columns, row)) for row in rows]
        finally:
            conn.close()
    
    async def _process_with_ai(self, web_content: WebContent) -> Dict[str, Any]:
        """Process web content with local LLM"""
        try:
            # Create a comprehensive prompt for the LLM
            prompt_content = f"""
            Title: {web_content.title}
            URL: {web_content.url}
            Domain: {web_content.metadata.get('domain', '')}
            
            Content:
            {web_content.content[:10000]}  # Limit content for LLM
            """
            
            # Get AI summary and analysis
            ai_result = ollama_summarize(prompt_content)
            
            # Enhanced processing for web content
            enhanced_prompt = f"""
            Analyze this web content and provide:
            1. A concise summary (2-3 sentences)
            2. Key topics/themes
            3. Relevant tags (3-5 tags)
            4. Content type (article, blog, tutorial, news, etc.)
            5. Notable quotes or key points
            
            Content: {prompt_content}
            """
            
            # Get enhanced analysis
            try:
                enhanced_result = ollama_summarize(enhanced_prompt)
                
                return {
                    "summary": enhanced_result.get("summary", ai_result.get("summary", "")),
                    "tags": enhanced_result.get("tags", ai_result.get("tags", [])),
                    "content_type": enhanced_result.get("content_type", "web_content"),
                    "key_points": enhanced_result.get("key_points", []),
                    "ai_analysis": enhanced_result
                }
            except:
                # Fallback to basic AI result
                return {
                    "summary": ai_result.get("summary", ""),
                    "tags": ai_result.get("tags", []),
                    "content_type": "web_content",
                    "key_points": [],
                    "ai_analysis": ai_result
                }
                
        except Exception as e:
            print(f"[web_ingestion] AI processing failed: {e}")
            return {
                "summary": web_content.summary or "Web content extracted",
                "tags": [],
                "content_type": "web_content",
                "key_points": [],
                "ai_analysis": {}
            }
    
    async def _store_content(self, web_content: WebContent, ai_results: Dict[str, Any], 
                           user_id: int, note_id: Optional[int] = None, config: ExtractionConfig | None = None) -> Tuple[int, Dict[str, Any]]:
        """Store web content in database"""
        conn = self.get_conn()
        try:
            c = conn.cursor()
            
            # Prepare data
            title = web_content.title or f"Web Content from {web_content.metadata.get('domain', 'Unknown')}"
            summary = ai_results.get("summary", "")
            tags = ",".join(ai_results.get("tags", []))
            content_type = ai_results.get("content_type", "web_content")

            artifact_dicts = [asdict(artifact) for artifact in web_content.artifacts]
            for artifact in artifact_dicts:
                artifact.setdefault("label", artifact.get("type", "artifact").title())

            total_new_storage = sum(a.get("size", 0) for a in artifact_dicts)
            self._enforce_storage_quota(user_id, total_new_storage)
            
            # Prepare metadata
            file_metadata = {
                "source_url": web_content.url,
                "domain": web_content.metadata.get("domain"),
                "extracted_at": web_content.extracted_at.isoformat(),
                "content_hash": web_content.content_hash,
                "screenshot_path": web_content.screenshot_path,
                "metadata": web_content.metadata,
                "ai_analysis": ai_results.get("ai_analysis", {}),
                "artifacts": artifact_dicts
            }
            
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if note_id:
                # Update existing note
                c.execute("""
                    UPDATE notes SET
                        title = ?, body = ?, content = ?, summary = ?, tags = ?, 
                        type = ?, file_metadata = ?, updated_at = datetime('now')
                    WHERE id = ? AND user_id = ?
                """, (
                    title, web_content.content, web_content.content, summary, tags,
                    content_type, json.dumps(file_metadata), note_id, user_id
                ))
                final_note_id = note_id
            else:
                # Create new note
                c.execute("""
                    INSERT INTO notes (
                        title, body, content, summary, tags, actions, type, timestamp,
                        file_metadata, status, user_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    title, web_content.content, web_content.content, summary, tags, "",
                    content_type, now, json.dumps(file_metadata),
                    "complete", user_id
                ))
                final_note_id = c.lastrowid
                # FTS triggers will index the note automatically
            
            conn.commit()

            manifest_path = self._write_manifest(
                final_note_id,
                user_id,
                web_content,
                ai_results,
                config or self.default_config,
                artifact_dicts
            )

            file_metadata["manifest_path"] = manifest_path
            c.execute(
                "UPDATE notes SET file_metadata = ? WHERE id = ?",
                (json.dumps(file_metadata), final_note_id)
            )
            conn.commit()

            return final_note_id, file_metadata
            
        finally:
            conn.close()
    
    def detect_urls(self, text: str) -> List[str]:
        """Detect URLs in text"""
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        return url_pattern.findall(text)
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and accessible"""
        try:
            parsed = urlparse(url)
            return parsed.scheme in ['http', 'https'] and parsed.netloc
        except:
            return False

    def _write_manifest(
        self,
        note_id: int,
        user_id: int,
        web_content: WebContent,
        ai_results: Dict[str, Any],
        config: ExtractionConfig,
        artifacts: List[Dict[str, Any]]
    ) -> str:
        manifest_dir = Path(settings.snapshots_dir) / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = manifest_dir / f"note_{note_id}_{timestamp}.json"

        manifest = {
            "note_id": note_id,
            "user_id": user_id,
            "url": web_content.url,
            "title": web_content.title,
            "content_hash": web_content.content_hash,
            "generated_at": datetime.now().isoformat(),
            "config": asdict(config),
            "artifacts": artifacts,
            "ai_summary": ai_results.get("summary"),
            "ai_tags": ai_results.get("tags", []),
            "metadata": web_content.metadata,
        }

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return str(manifest_path)

    def _calculate_storage_usage(self, user_id: int) -> int:
        conn = self.get_conn()
        try:
            c = conn.cursor()
            rows = c.execute(
                "SELECT file_metadata FROM notes WHERE user_id = ?",
                (user_id,)
            ).fetchall()

            total = 0
            for (metadata_json,) in rows:
                if not metadata_json:
                    continue
                try:
                    meta = json.loads(metadata_json)
                except Exception:
                    continue
                artifacts = meta.get("artifacts", [])
                for artifact in artifacts:
                    total += int(artifact.get("size", 0))
            return total
        finally:
            conn.close()

    def _enforce_storage_quota(self, user_id: int, additional_bytes: int) -> None:
        if additional_bytes <= 0:
            return
        limit_bytes = settings.web_storage_limit_mb * 1024 * 1024
        if not limit_bytes:
            return
        current_usage = self._calculate_storage_usage(user_id)
        if current_usage + additional_bytes > limit_bytes:
            raise ValueError(
                f"Storage quota exceeded: {current_usage + additional_bytes:,} bytes > {limit_bytes:,} bytes"
            )


# Integration with Smart Automation System
class UrlDetectionWorkflow:
    """Workflow for automatic URL detection and processing"""
    
    def __init__(self, web_ingestion_service: WebIngestionService):
        self.web_service = web_ingestion_service
    
    async def process_content_for_urls(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process content and extract any URLs found"""
        content = content_data.get("content", "")
        user_id = content_data.get("user_id")
        note_id = content_data.get("note_id")
        
        if not content or not user_id:
            return {"urls_processed": 0, "results": []}
        
        # Detect URLs
        urls = self.web_service.detect_urls(content)
        valid_urls = [url for url in urls if self.web_service.is_valid_url(url)]
        
        if not valid_urls:
            return {"urls_processed": 0, "results": []}
        
        # Process URLs
        results = []
        for url in valid_urls[:3]:  # Limit to first 3 URLs to avoid overload
            try:
                result = await self.web_service.ingest_url(url, user_id)
                results.append({
                    "url": url,
                    "success": result["success"],
                    "note_id": result.get("note_id"),
                    "title": result.get("title", ""),
                    "error": result.get("error")
                })
            except Exception as e:
                results.append({
                    "url": url,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "urls_processed": len(results),
            "results": results
        }


# API Models
class UrlIngestionRequest(BaseModel):
    url: str
    capture_screenshot: Optional[bool] = None
    capture_pdf: Optional[bool] = None
    capture_html: Optional[bool] = None
    download_original: Optional[bool] = None
    download_media: Optional[bool] = None
    fetch_captions: Optional[bool] = None
    extract_images: Optional[bool] = None
    timeout: Optional[int] = None
    # Legacy alias support
    take_screenshot: Optional[bool] = None
    async_mode: Optional[bool] = None


class UrlIngestionResponse(BaseModel):
    success: bool
    note_id: Optional[int] = None
    title: Optional[str] = None
    content_length: Optional[int] = None
    content_preview: Optional[str] = None
    summary: Optional[str] = None
    tags: List[str] = []
    screenshot_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    job_id: Optional[str] = None
    queued: Optional[bool] = None
    error: Optional[str] = None


def get_web_ingestion_service(get_conn_func):
    """Factory function to get web ingestion service."""
    return WebIngestionService(get_conn_func)
