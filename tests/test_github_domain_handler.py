import pytest

from services.web_ingestion_service import (
    GitHubRepoHandler,
    ExtractionConfig,
    WebContent,
)


@pytest.mark.asyncio
async def test_github_handler_matches_github_repo():
    handler = GitHubRepoHandler()
    parsed = __import__('urllib.parse').urlparse('https://github.com/owner/repo')

    assert handler.matches(parsed) is True

    parsed_other = __import__('urllib.parse').urlparse('https://example.com/foo')
    assert handler.matches(parsed_other) is False


@pytest.mark.asyncio
async def test_github_handler_fetches_combined_content(monkeypatch):
    handler = GitHubRepoHandler()

    repo_payload = {
        'full_name': 'owner/repo',
        'name': 'repo',
        'description': 'Test repository',
        'default_branch': 'main',
        'stargazers_count': 42,
        'forks_count': 5,
        'subscribers_count': 3,
        'open_issues_count': 7,
        'topics': ['python', 'async'],
        'license': {'name': 'MIT'},
        'visibility': 'public',
        'pushed_at': '2024-01-01T00:00:00Z',
        'homepage': 'https://example.com',
    }

    async def fake_repo(owner, repo):
        return repo_payload

    async def fake_readme(owner, repo, branch):
        return "# Heading\nSome README content"

    async def fake_supp(owner, repo, branch):
        return [("CONTRIBUTING.md", "Contribution guidelines"), ("docs/overview.md", "Overview doc")]

    async def fake_issues(owner, repo):
        return [
            {
                'title': 'First issue',
                'number': 1,
                'html_url': 'https://github.com/owner/repo/issues/1',
                'created_at': '2024-01-02T00:00:00Z',
                'labels': [{'name': 'bug'}],
            },
            {
                'title': 'Second issue',
                'number': 2,
                'html_url': 'https://github.com/owner/repo/issues/2',
                'created_at': '2024-01-03T00:00:00Z',
                'labels': [{'name': 'enhancement'}],
            },
        ]

    async def fake_languages(owner, repo):
        return {'Python': 1000, 'JavaScript': 500}

    monkeypatch.setattr(handler, '_fetch_repo_metadata', fake_repo)
    monkeypatch.setattr(handler, '_fetch_readme', fake_readme)
    monkeypatch.setattr(handler, '_fetch_supplemental_files', fake_supp)
    monkeypatch.setattr(handler, '_fetch_top_issues', fake_issues)
    monkeypatch.setattr(handler, '_fetch_languages', fake_languages)

    config = ExtractionConfig(max_content_length=10000)
    result = await handler.fetch('https://github.com/owner/repo', config)

    assert isinstance(result, WebContent)
    assert 'README content' in result.content
    assert 'Contribution guidelines' in result.content
    assert 'Recent Issues' in result.content
    assert result.metadata['repo']['full_name'] == 'owner/repo'
    assert len(result.metadata['supplemental_files']) == 2
    assert result.metadata['issues'][0]['title'] == 'First issue'
    assert result.metadata['content_truncated'] is False
    assert isinstance(result.metadata.get('artifacts'), list)


@pytest.mark.asyncio
async def test_github_handler_respects_max_length(monkeypatch):
    handler = GitHubRepoHandler()

    async def fake_repo(owner, repo):
        return {
            'full_name': 'owner/repo',
            'name': 'repo',
            'description': 'Test repository',
            'default_branch': 'main',
            'stargazers_count': 1,
            'forks_count': 0,
            'subscribers_count': 0,
            'open_issues_count': 0,
            'topics': [],
            'license': None,
            'visibility': 'public',
            'pushed_at': '2024-01-01T00:00:00Z',
            'homepage': None,
        }

    async def fake_readme(owner, repo, branch):
        return 'A' * 500

    async def fake_supp(owner, repo, branch):
        return []

    async def fake_issues(owner, repo):
        return []

    async def fake_languages(owner, repo):
        return {}

    monkeypatch.setattr(handler, '_fetch_repo_metadata', fake_repo)
    monkeypatch.setattr(handler, '_fetch_readme', fake_readme)
    monkeypatch.setattr(handler, '_fetch_supplemental_files', fake_supp)
    monkeypatch.setattr(handler, '_fetch_top_issues', fake_issues)
    monkeypatch.setattr(handler, '_fetch_languages', fake_languages)

    config = ExtractionConfig(max_content_length=120)
    result = await handler.fetch('https://github.com/owner/repo', config)

    assert result.metadata['content_truncated'] is True
    assert result.content.endswith('...[content truncated]')
