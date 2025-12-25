"""
Configuration management with environment variable validation
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Required API Keys
    anthropic_api_key: str = Field(..., description="Anthropic API key for Claude")
    openai_api_key: str = Field(..., description="OpenAI API key for embeddings")
    
    # Server Configuration
    environment: str = Field(default="development", description="Environment: development, staging, production")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # CORS Configuration
    cors_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated list of allowed CORS origins"
    )
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=10, ge=1, le=1000, description="Rate limit per minute per IP")
    rate_limit_per_hour: int = Field(default=100, ge=1, le=10000, description="Rate limit per hour per IP")
    
    # Security Settings
    max_message_length: int = Field(default=10000, ge=100, le=50000, description="Maximum message length in characters")
    max_sessions_per_user: int = Field(default=50, ge=1, le=1000, description="Maximum sessions per user")
    
    # Database
    db_path: str = Field(default="data/db/agent.db", description="Path to SQLite database")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # Allow extra fields for flexibility
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value"""
        allowed = ['development', 'staging', 'production']
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of: {', '.join(allowed)}")
        return v.lower()
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level"""
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of: {', '.join(allowed)}")
        return v.upper()
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list"""
        return [origin.strip() for origin in self.cors_origins.split(",")]


# Global settings instance
try:
    settings = Settings()
except Exception as e:
    # Fallback settings if .env is missing
    print(f"Warning: Could not load settings: {e}")
    print("Using fallback configuration (development mode)")
    
    settings = Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        environment="development",
        debug=True,
        cors_origins="http://localhost:3000,http://localhost:8080",
        rate_limit_per_minute=10,
        rate_limit_per_hour=100,
        max_message_length=10000,
        max_sessions_per_user=50
    )
