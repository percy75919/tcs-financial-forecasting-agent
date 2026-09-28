from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    ollama_model: str = "qwen3:8b"
    ollama_base_url: str = "http://localhost:11434"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_collection_name: str = "tcs_earnings_transcripts_fastembed_bge_small"

    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_database: str = "tcs_forecast"
    mysql_user: str = "tcs_user"
    mysql_password: str = "change_me"

    data_dir: Path = Path("data")
    chroma_dir: Path = Path("data/chroma")
    top_k_transcript_chunks: int = 8
    request_timeout_seconds: int = 180

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def mysql_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.chroma_dir.mkdir(parents=True, exist_ok=True)
    return s
