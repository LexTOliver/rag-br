"""
Configuration module for vectorization settings.

Centralizes all configuration parameters for text chunking, embedding,
and vector storage operations used in document vectorization tasks.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """
    Definição de configurações para modelos de embeddings pré-treinados.

    Attributes:
        model_name (str): Nome do modelo pré-treinado.
        device (str): Dispositivo para computação ('cpu' ou 'cuda').
    """

    model_name: str = "intfloat/multilingual-e5-small"
    device: str = "cpu"


@dataclass
class ChunkerConfig:
    """
    Definição de configurações para chunking dos textos.

    Attributes:
        model_name (str): Nome do modelo de tokenização.
        chunk_size (int): Tamanho máximo de cada pedaço em tokens.
        overlap (int): Quantidade de sobreposição de tokens entre chunks.
    """

    chunk_size: int = 256
    overlap: int = 64

    def __post_init__(self):
        """Validações após inicialização."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size deve ser positivo")

        if self.overlap >= self.chunk_size:
            raise ValueError("overlap deve ser menor que chunk_size")

        if self.overlap < 0:
            raise ValueError("overlap não pode ser negativo")


@dataclass
class EmbedderConfig:
    """
    Definição de configurações para geração de embeddings.

    Attributes:
        model_name (str): Nome do modelo pré-treinado.
        batch_size (int): Tamanho do lote para processamento em batch.
        local_cache_dir (Optional[str]): Diretório para cache local de embeddings.
        enable_local_cache (bool): Habilita ou desabilita o cache local de embeddings.
    """

    batch_size: int = 32
    local_cache_dir: Optional[str] = "data/embeddings_cache"
    enable_local_cache: bool = True

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size deve ser positivo")


@dataclass
class VectorStoreConfig:
    """
    Definição de configurações para o armazenamento vetorial.

    Attributes:
        collection_name (str): Nome da coleção no VectorStore.
        path (str): Caminho para armazenamento local do VectorStore.
    """

    collection_name: str = "quati_chunks"
    path: str = "data/qdrant"
    distance_metric: str = "cosine"

    def __post_init__(self):
        valid_metrics = ["cosine", "euclidean", "dot"]
        if self.distance_metric not in valid_metrics:
            raise ValueError(f"distance_metric deve ser um de: {valid_metrics}")


@dataclass
class IndexingConfig:
    """
    Definição de configurações para o pipeline de indexação.

    Attributes:
        dataset (Dataset): Dataset a ser indexado.
        metadata_fields (list): Campos de metadados a extrair.
    """

    source: str = "parquet"  # Source type: parquet, HF_dataset, etc.
    dataset: dict = field(
        default_factory=lambda: {
            "data_path": "data/processed/quati_reranker_eval_vi.parquet"
        }
    )
    metadata_fields: Optional[List[str]] = field(default_factory=lambda: ["passage_id"])
    batch_size: int = 16
    id_field: str = "passage_id"
    text_field: str = "passage"


@dataclass
class VectorIndexConfig:
    """
    Configuração mestre do pipeline de indexação vetorial.

    Attributes:
        dataset (Dataset): Dataset a ser indexado.
        metadata_fields (list): Campos de metadados a extrair.
        chunker (ChunkerConfig): Configurações do Chunker.
        embedder (EmbedderConfig): Configurações do Embedder.
        vector_store (VectorStoreConfig): Configurações do VectorStore.
    """

    model: ModelConfig
    chunker: ChunkerConfig
    embedder: EmbedderConfig
    vector_store: VectorStoreConfig
    indexing: IndexingConfig

    @classmethod
    def from_dict(cls, config_dict: dict) -> "VectorIndexConfig":
        """
        Cria uma instância de VectorIndexConfig a partir de um dicionário de configurações.

        Params:
            config_dict (dict): Dicionário contendo as configurações.
            dataset (Dataset): Dataset a ser indexado.

        Returns:
            VectorIndexConfig: Instância configurada.
        """
        model_cfg = ModelConfig(**config_dict.get("model", {}))
        chunker_cfg = ChunkerConfig(**config_dict.get("chunker", {}))
        embedder_cfg = EmbedderConfig(**config_dict.get("embedder", {}))
        vector_store_cfg = VectorStoreConfig(**config_dict.get("vector_store", {}))
        indexing_cfg = IndexingConfig(**config_dict.get("indexing", {}))

        return cls(
            model=model_cfg,
            chunker=chunker_cfg,
            embedder=embedder_cfg,
            vector_store=vector_store_cfg,
            indexing=indexing_cfg,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "VectorIndexConfig":
        """
        Cria uma instância de VectorIndexConfig a partir de um arquivo YAML.

        Params:
            yaml_path (str): Caminho para o arquivo YAML contendo as configurações.

        Returns:
            VectorIndexConfig: Instância configurada.
        """
        import yaml

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    def to_dict(self) -> dict:
        """
        Converte a configuração para um dicionário.

        Returns:
            dict: Dicionário representando a configuração.
        """
        return {
            "model": self.model.__dict__,
            "chunker": self.chunker.__dict__,
            "embedder": self.embedder.__dict__,
            "vector_store": self.vector_store.__dict__,
            "indexing": self.indexing.__dict__,
        }
