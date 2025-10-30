from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices, model_validator
from typing import Any, Optional




BASE_DIR = Path(__file__).parent.resolve()

class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    base_dir: Path = BASE_DIR
    db_path: Path = BASE_DIR / "notes.db"
    vault_path: Path = BASE_DIR
    audio_dir: Path = BASE_DIR / "audio"
    uploads_dir: Path = BASE_DIR / "uploads"
    media_dir: Path = BASE_DIR / "media"
    snapshots_dir: Path = BASE_DIR / "snapshots"
    videos_dir: Path = BASE_DIR / "videos"
    # Maximum size (in bytes) for any uploaded file (audio/images/pdfs)
    # Can be overridden via env var MAX_FILE_SIZE
    max_file_size: int = 200 * 1024 * 1024  # 200MB default
    whisper_cpp_path: Path = BASE_DIR / "build/bin/whisper-cli"
    whisper_model_path: Path = BASE_DIR / "models/ggml-base.en.bin"
    # Transcription backend: 'whisper' (whisper.cpp) or 'vosk' (lightweight, CPU-only)
    transcriber: str = "whisper"
    # Path to Vosk ASR model directory when using transcriber='vosk'
    vosk_model_path: Optional[Path] = None
    ollama_api_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.2"
    # Optional Ollama performance/resource knobs
    ollama_num_ctx: Optional[int] = None
    ollama_num_predict: Optional[int] = None
    ollama_temperature: Optional[float] = None
    ollama_top_p: Optional[float] = None
    ollama_num_gpu: Optional[int] = None
    # AI processing controls (to reduce local CPU/RAM usage)
    ai_processing_enabled: bool = True
    ai_chunk_size_chars: int = 1500
    ai_throttle_delay_seconds: int = 2
    # Processing concurrency for background audio transcription/LLM jobs
    processing_concurrency: int = 2
    # Transcription concurrency: allow more than one whisper job at once
    transcription_concurrency: int = 1
    # Batch processing mode: queue multiple files without immediate processing
    batch_mode_enabled: bool = False
    # Number of audio files to queue before starting batch processing
    batch_size_threshold: int = 5
    # Time to wait (seconds) before processing partial batches
    batch_timeout_seconds: int = 300  # 5 minutes
    # Split long WAVs into segments (seconds) to avoid timeouts/CPU spikes
    transcription_segment_seconds: int = 600
    # Max seconds to process a single note before marking failed:timeout
    # Increase this if you plan to upload longer audio recordings
    processing_timeout_seconds: int = 1800  # 30 minutes

    # Web ingestion defaults and quotas
    web_capture_screenshot_default: bool = Field(
        default=True,
        validation_alias=AliasChoices('web_capture_screenshot_default', 'WEB_CAPTURE_SCREENSHOT_DEFAULT')
    )
    web_capture_pdf_default: bool = Field(
        default=False,
        validation_alias=AliasChoices('web_capture_pdf_default', 'WEB_CAPTURE_PDF_DEFAULT')
    )
    web_capture_html_default: bool = Field(
        default=True,
        validation_alias=AliasChoices('web_capture_html_default', 'WEB_CAPTURE_HTML_DEFAULT')
    )
    web_download_original_default: bool = Field(
        default=False,
        validation_alias=AliasChoices('web_download_original_default', 'WEB_DOWNLOAD_ORIGINAL_DEFAULT')
    )
    web_download_media_default: bool = Field(
        default=False,
        validation_alias=AliasChoices('web_download_media_default', 'WEB_DOWNLOAD_MEDIA_DEFAULT')
    )
    web_fetch_captions_default: bool = Field(
        default=False,
        validation_alias=AliasChoices('web_fetch_captions_default', 'WEB_FETCH_CAPTIONS_DEFAULT')
    )
    web_extract_images_default: bool = Field(
        default=False,
        validation_alias=AliasChoices('web_extract_images_default', 'WEB_EXTRACT_IMAGES_DEFAULT')
    )
    web_timeout_default: int = Field(
        default=30,
        validation_alias=AliasChoices('web_timeout_default', 'WEB_TIMEOUT_DEFAULT')
    )
    web_storage_limit_mb: int = Field(
        default=4096,
        validation_alias=AliasChoices('web_storage_limit_mb', 'WEB_STORAGE_LIMIT_MB')
    )
    web_async_ingestion_default: bool = Field(
        default=False,
        validation_alias=AliasChoices('web_async_ingestion_default', 'WEB_ASYNC_INGESTION_DEFAULT')
    )

    redis_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices('redis_url', 'REDIS_URL')
    )
    
    # Security settings
    secret_key: str = Field(
        default="generate-secure-key-in-production",
        validation_alias=AliasChoices('secret_key', 'SECRET_KEY')
    )
    webhook_token: str = Field(
        default="generate-secure-webhook-token-in-production",
        validation_alias=AliasChoices('webhook_token', 'WEBHOOK_TOKEN')
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Production security settings
    environment: str = Field(
        default="development",
        validation_alias=AliasChoices('environment', 'ENVIRONMENT', 'ENV')
    )
    
    # CORS configuration
    cors_origins: str = Field(
        default="http://localhost:8082,http://127.0.0.1:8082",
        validation_alias=AliasChoices('cors_origins', 'CORS_ORIGINS')
    )
    cors_credentials: bool = Field(
        default=True,
        validation_alias=AliasChoices('cors_credentials', 'CORS_CREDENTIALS')
    )
    
    # Rate limiting
    rate_limit_per_minute: int = Field(
        default=100,
        validation_alias=AliasChoices('rate_limit_per_minute', 'RATE_LIMIT_PER_MINUTE')
    )
    auth_rate_limit_per_minute: int = Field(
        default=5,
        validation_alias=AliasChoices('auth_rate_limit_per_minute', 'AUTH_RATE_LIMIT_PER_MINUTE')
    )
    
    # Cookie security
    cookie_secure: bool = Field(
        default_factory=lambda: os.environ.get('ENVIRONMENT', 'development') == 'production',
        validation_alias=AliasChoices('cookie_secure', 'COOKIE_SECURE')
    )
    cookie_samesite: str = Field(
        default="strict",
        validation_alias=AliasChoices('cookie_samesite', 'COOKIE_SAMESITE')
    )
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() in ('production', 'prod')
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as list."""
        return [origin.strip() for origin in self.cors_origins.split(',') if origin.strip()]

 # NEW: Obsidian + Raindrop
    obsidian_vault_path: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices('obsidian_vault_path','OBSIDIAN_VAULT_PATH')
    )
    obsidian_projects_root: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices('obsidian_projects_root','OBSIDIAN_PROJECTS_ROOT')
    )
    obsidian_per_project: bool = Field(
        default=True,
        validation_alias=AliasChoices('obsidian_per_project','OBSIDIAN_PER_PROJECT')
    )
    raindrop_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices('raindrop_token','RAINDROP_TOKEN')
    )

    # Advanced Search Configuration
    search_rerank_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices('search_rerank_enabled', 'SEARCH_RERANK_ENABLED')
    )
    search_rerank_top_k: int = Field(
        default=20,
        validation_alias=AliasChoices('search_rerank_top_k', 'SEARCH_RERANK_TOP_K')
    )
    search_rerank_final_k: int = Field(
        default=8,
        validation_alias=AliasChoices('search_rerank_final_k', 'SEARCH_RERANK_FINAL_K')
    )
    search_rerank_weight: float = Field(
        default=0.7,
        validation_alias=AliasChoices('search_rerank_weight', 'SEARCH_RERANK_WEIGHT')
    )
    search_original_weight: float = Field(
        default=0.3,
        validation_alias=AliasChoices('search_original_weight', 'SEARCH_ORIGINAL_WEIGHT')
    )
    search_cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias=AliasChoices('search_cross_encoder_model', 'SEARCH_CROSS_ENCODER_MODEL')
    )

    # Email Configuration for Magic Links
    email_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices('email_enabled', 'EMAIL_ENABLED')
    )
    email_service: str = Field(
        default="resend",  # resend, sendgrid, mailgun, smtp
        validation_alias=AliasChoices('email_service', 'EMAIL_SERVICE')
    )
    email_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices('email_api_key', 'EMAIL_API_KEY', 'RESEND_API_KEY')
    )
    email_from: str = Field(
        default="noreply@localhost",
        validation_alias=AliasChoices('email_from', 'EMAIL_FROM')
    )
    email_from_name: str = Field(
        default="Second Brain",
        validation_alias=AliasChoices('email_from_name', 'EMAIL_FROM_NAME')
    )
    # SMTP Configuration (if using email_service=smtp)
    smtp_host: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices('smtp_host', 'SMTP_HOST')
    )
    smtp_port: int = Field(
        default=587,
        validation_alias=AliasChoices('smtp_port', 'SMTP_PORT')
    )
    smtp_username: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices('smtp_username', 'SMTP_USERNAME')
    )
    smtp_password: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices('smtp_password', 'SMTP_PASSWORD')
    )
    smtp_use_tls: bool = Field(
        default=True,
        validation_alias=AliasChoices('smtp_use_tls', 'SMTP_USE_TLS')
    )

    # === LOCAL-FIRST AI CONFIGURATION ===
    # Master switch: when False, ALL external AI services are disabled
    # When True, allows fallback to external AI when local models fail
    ai_allow_external: bool = Field(
        default=False,
        validation_alias=AliasChoices('ai_allow_external', 'AI_ALLOW_EXTERNAL')
    )
    
    # Local AI preferences (used when ai_allow_external=True for fallbacks)
    ai_prefer_local: bool = Field(
        default=True,
        validation_alias=AliasChoices('ai_prefer_local', 'AI_PREFER_LOCAL')
    )
    
    # Embedding provider priority: local models first, external as fallback
    ai_embeddings_provider_priority: str = Field(
        default="sentence_transformers,ollama,openai",
        validation_alias=AliasChoices('ai_embeddings_provider_priority', 'AI_EMBEDDINGS_PROVIDER_PRIORITY')
    )
    
    # LLM provider priority: local models first, external as fallback  
    ai_llm_provider_priority: str = Field(
        default="ollama,openai,anthropic",
        validation_alias=AliasChoices('ai_llm_provider_priority', 'AI_LLM_PROVIDER_PRIORITY')
    )
    
    @property
    def embeddings_providers(self) -> list[str]:
        """Get embedding providers as list."""
        return [p.strip() for p in self.ai_embeddings_provider_priority.split(',') if p.strip()]
    
    @property  
    def llm_providers(self) -> list[str]:
        """Get LLM providers as list."""
        return [p.strip() for p in self.ai_llm_provider_priority.split(',') if p.strip()]
    
    # External AI API keys (only used when ai_allow_external=True)
    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices('openai_api_key', 'OPENAI_API_KEY')
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices('anthropic_api_key', 'ANTHROPIC_API_KEY')
    )


    # Capture deduplication (prevent duplicate notes based on content hash)
    capture_dedup_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices('capture_dedup_enabled', 'CAPTURE_DEDUP_ENABLED')
    )
    # If > 0, consider duplicates only within this window (days). 0 = no window.
    capture_dedup_window_days: int = Field(
        default=30,
        validation_alias=AliasChoices('capture_dedup_window_days', 'CAPTURE_DEDUP_WINDOW_DAYS')
    )

    # Memory System Settings
    memory_extraction_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices('memory_extraction_enabled', 'MEMORY_EXTRACTION_ENABLED')
    )
    memory_extraction_threshold: int = Field(
        default=4,  # Min messages before extraction
        validation_alias=AliasChoices('memory_extraction_threshold', 'MEMORY_EXTRACTION_THRESHOLD')
    )
    memory_retention_days: int = Field(
        default=365,
        validation_alias=AliasChoices('memory_retention_days', 'MEMORY_RETENTION_DAYS')
    )
    memory_consolidation_batch_size: int = Field(
        default=10,
        validation_alias=AliasChoices('memory_consolidation_batch_size', 'MEMORY_CONSOLIDATION_BATCH_SIZE')
    )

    # Model Selection (can be overridden at runtime)
    chat_model: str = Field(
        default="llama3.2",
        validation_alias=AliasChoices('chat_model', 'CHAT_MODEL')
    )
    memory_extraction_model: str = Field(
        default="llama3.1:8b",
        validation_alias=AliasChoices('memory_extraction_model', 'MEMORY_EXTRACTION_MODEL')
    )
    summarization_model: str = Field(
        default="llama3.2",
        validation_alias=AliasChoices('summarization_model', 'SUMMARIZATION_MODEL')
    )
    title_generation_model: str = Field(
        default="llama3.2",
        validation_alias=AliasChoices('title_generation_model', 'TITLE_GENERATION_MODEL')
    )

    # Memory Retrieval Settings
    max_episodic_memories: int = Field(
        default=5,
        validation_alias=AliasChoices('max_episodic_memories', 'MAX_EPISODIC_MEMORIES')
    )
    max_semantic_memories: int = Field(
        default=10,
        validation_alias=AliasChoices('max_semantic_memories', 'MAX_SEMANTIC_MEMORIES')
    )
    max_document_results: int = Field(
        default=5,
        validation_alias=AliasChoices('max_document_results', 'MAX_DOCUMENT_RESULTS')
    )
    episodic_importance_threshold: float = Field(
        default=0.3,
        validation_alias=AliasChoices('episodic_importance_threshold', 'EPISODIC_IMPORTANCE_THRESHOLD')
    )

    # Vector Search Settings for Memories
    memory_vector_enabled: bool = Field(
        default=True,  # Set to False if sqlite-vec not available
        validation_alias=AliasChoices('memory_vector_enabled', 'MEMORY_VECTOR_ENABLED')
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"   # prevents crashes if other stray keys exist
    )
    
    @model_validator(mode='after')
    def generate_secure_keys(self) -> 'Settings':
        """Generate secure keys if defaults are still being used."""
        if self.secret_key == "generate-secure-key-in-production":
            self.secret_key = os.urandom(32).hex()
            print("⚠️  Generated random SECRET_KEY. Set SECRET_KEY env var for production!")
        
        if self.webhook_token == "generate-secure-webhook-token-in-production":
            self.webhook_token = os.urandom(32).hex()
            print("⚠️  Generated random WEBHOOK_TOKEN. Set WEBHOOK_TOKEN env var for production!")
        
        return self

settings = Settings()

def get_settings() -> Settings:
    """Get application settings instance"""
    return settings
