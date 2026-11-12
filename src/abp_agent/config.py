"""Settings, read from environment variables (and .env when running outside Docker)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Postgres (ParadeDB)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "abp"
    postgres_password: str = ""
    postgres_db: str = "abp"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # LLM (any OpenAI-compatible server)
    llm_base_url: str = "http://host.docker.internal:1234/v1"
    llm_model: str = "qwen3.8-27b"
    llm_api_key: str = "not-needed"

    # Embeddings: bge-m3 in LM Studio, beside the chat model (D41)
    embed_base_url: str = "http://host.docker.internal:1234/v1"
    embed_model: str = "text-embedding-bge-m3"

    # Platform services
    mlflow_tracking_uri: str = "http://localhost:5000"
    prefect_api_url: str = "http://localhost:4200/api"

    # ABP repository (bind-mounted read-only; see compose.yaml)
    abp_repo_dir: str = "/sources/abp"

    # MCP server: clients send this as a bearer token. Empty means nobody gets in.
    mcp_token: str = ""
    mcp_public_url: str = "http://localhost:8011"

    @property
    def database_url(self) -> str:
        """SQLAlchemy URL using the psycopg 3 driver."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
