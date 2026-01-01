"""Configuration management for the Gemini RAG system."""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # API Keys
    fi_api_key: str = Field(default="", alias="FI_API_KEY")
    fi_secret_key: str = Field(default="", alias="FI_SECRET_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    
    # Project Configuration
    project_name: str = Field(default="gemini-document-qa", alias="PROJECT_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    
    # Model Configuration
    embedding_model: str = "models/text-embedding-004"
    generation_model: str = "gemini-2.5-flash"
    
    # RAG Parameters
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k_results: int = 5
    
    # Evaluation Settings
    enable_evaluation: bool = True
    evaluation_templates: list[str] = Field(
        default_factory=lambda: ["hallucination", "relevance", "toxicity", "tone"]
    )
    
    # Directory Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent / "data" / "documents")
    chroma_dir: Path = Field(default_factory=lambda: Path(__file__).parent / "chroma_db")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings."""
    return settings
