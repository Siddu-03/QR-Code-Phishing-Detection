"""
Centralized application configuration.
Loads values from environment variables / .env file using pydantic-settings.
"""
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


APP_VERSION = "1.0.0"


class Settings(BaseSettings):
    # App
    APP_NAME: str = "QR Shield API"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security
    API_KEY: str = "change-me-in-production"
    SECRET_KEY: str = "change-me-to-a-random-secret"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    # Authentication must be explicitly disabled; it is never silently
    # bypassed just because APP_ENV=="development". Only set this to
    # true in a local/dev environment that is not internet-reachable.
    AUTH_DISABLED: bool = False

    # Upload / decoded-image safety limits (Change 12 - security hardening)
    MAX_IMAGE_DIMENSION_PX: int = 6000

    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # Uploads
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_TYPES: str = "image/png,image/jpeg,image/jpg,image/webp"

    # QR Shield core (Weeks 1-4 CV modules: image_loader, tamper_analysis, dataset_management)
    QR_SHIELD_CORE_PATH: str = "../qr_shield_core"

    # Storage
    UPLOAD_DIR: str = "./storage/uploads"
    REPORT_DIR: str = "./storage/reports"
    HISTORY_FILE: str = "./storage/history.json"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./storage/logs/backend.log"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def allowed_image_types_list(self) -> List[str]:
        return [t.strip() for t in self.ALLOWED_IMAGE_TYPES.split(",") if t.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    def ensure_dirs(self) -> None:
        for path in (self.UPLOAD_DIR, self.REPORT_DIR, Path(self.HISTORY_FILE).parent, Path(self.LOG_FILE).parent):
            Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
