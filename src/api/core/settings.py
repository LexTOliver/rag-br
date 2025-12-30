"""
API settings for the RAG-BR application.

Defines configuration parameters such as API metadata,
and application-specific settings like Vector Index and Qdrant paths.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class APISettings(BaseSettings):
    """Configurações para API do RAG-BR."""

    # -- API SETTINGS --
    # Metadata
    title: str = "RAG-BR"
    version: str = "0.1.0"
    description: str = "Sistema de Recuperação e Resposta Baseada em Documentos em Português com Modelo de Reranking Treinado."

    # API
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")

    # -- APPLICATION SETTINGS
    # Vector Index
    index_config_path: str = Field(
        default="configs/index_config.yaml", env="INDEX_CONFIG_PATH"
    )

    # Qdrant
    # TODO: Avaliar se deixar configuração pelo VectorIndex ou separado
    # TODO: Implementar conexão remota via Qdrant Cloud ou Docker
    # qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    # qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_path: str = Field(default="data/qdrant", env="QDRANT_PATH")

    class Config:
        """Configuração do Pydantic Settings."""

        # TODO: Mudar para .env em produção
        env_file = ".env.example"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = APISettings()
