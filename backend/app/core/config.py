import os
import secrets
from typing import Any, Dict, List, Optional, Union

from pydantic import AnyHttpUrl, EmailStr, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Base
    APP_NAME: str = "Media Transcription Service"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:8000", "http://localhost:3000"]

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Database
    DATABASE_URL: str
    DATABASE_TEST_URL: Optional[str] = None

    # JWT
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 365  # 1 year
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # First superuser
    FIRST_SUPERUSER_EMAIL: EmailStr = "avpmanager@gmail.com"
    FIRST_SUPERUSER_USERNAME: str = "admin"
    FIRST_SUPERUSER_PASSWORD: str = "admin"

    # File storage
    UPLOAD_DIR: str = "/backend/app/uploads"
    TEMP_DIR: str = "/backend/app/temp"
    MAX_UPLOAD_SIZE: int = 1024 * 1024 * 1024  # 1 GB

    # RabbitMQ
    RABBITMQ_HOST: str
    RABBITMQ_PORT: int
    RABBITMQ_USER: str
    RABBITMQ_PASSWORD: str
    RABBITMQ_VHOST: str = "/"
    RABBITMQ_QUEUE_TRANSCRIPTION: str = "transcription_queue_with_ttl"

    # OpenAI
    OPENAI_API_KEY: str

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma3:12b"

    # Email
    SMTP_HOST: str
    SMTP_PORT: int
    SMTP_USER: str
    SMTP_PASSWORD: str
    SMTP_TLS: bool = True
    EMAIL_FROM: EmailStr

    # LDAP (for future use)
    LDAP_SERVER: Optional[str] = None
    LDAP_BIND_DN: Optional[str] = None
    LDAP_BIND_PASSWORD: Optional[str] = None
    LDAP_USER_SEARCH_BASE: Optional[str] = None
    LDAP_USER_SEARCH_FILTER: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Create directories if they don't exist
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Получаем путь к корню проекта
UPLOAD_DIR = os.path.join(BASE_DIR, settings.UPLOAD_DIR)
TEMP_DIR = os.path.join(BASE_DIR, settings.TEMP_DIR,)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
