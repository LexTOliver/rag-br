"""
Configuration module for vectorization settings.

Centralizes all configuration parameters for text chunking, embedding,
and vector storage operations used in document vectorization tasks.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass
class ModelConfig:
    """
    Definição de configurações para modelos de embeddings pré-treinados.

    Attributes:
        model_name (str): Nome do modelo pré-treinado.
        device (str): Dispositivo para computação ('cpu' ou 'cuda').
    """

    model_name: str = "intfloat/multilingual-e5-small"
    device: Literal["cpu", "cuda"] = "cpu"

    def __post_init__(self):
        valid_devices = ["cpu", "cuda"]
        if self.device not in valid_devices:
            raise ValueError(f"device deve ser um de: {valid_devices}")


@dataclass
class ChunkerConfig:
    """
    Definição de configurações para chunking dos textos.

    Attributes:
        chunk_size (int): Tamanho máximo de cada pedaço em tokens.
        overlap (int): Quantidade de sobreposição de tokens entre chunks.
    """

    chunk_size: int = 256
    overlap: int = 64

    def __post_init__(self):
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
        batch_size (int): Tamanho do lote para processamento em batch.
        local_cache_dir (Optional[str]): Diretório para cache local de embeddings.
        enable_local_cache (bool): Habilita ou desabilita o cache local de embeddings.
    """

    batch_size: int = 32
    local_cache_dir: Optional[str] = "data/embeddings_cache"
    cache_limit_size: int = 100000  # in Bytes
    enable_local_cache: bool = False

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size deve ser positivo")

        if self.enable_local_cache and not self.local_cache_dir:
            raise ValueError(
                "local_cache_dir deve ser definido se enable_local_cache for True"
            )

        if self.enable_local_cache and self.cache_limit_size <= 0:
            raise ValueError("cache_limit_size deve ser positivo")


@dataclass
class VectorStoreConfig:
    """
    Definição de configurações para o armazenamento vetorial.

    Attributes:
        collection_name (str): Nome da coleção no VectorStore.
        path (str): Caminho para armazenamento local do VectorStore.
        distance_metric (str): Métrica de distância para buscas vetoriais.
    """

    collection_name: str = "documents"
    path: str = "data/qdrant_db"
    distance_metric: Literal["cosine", "euclidean", "dot"] = "cosine"

    def __post_init__(self):
        valid_metrics = ["cosine", "euclidean", "dot"]
        if self.distance_metric not in valid_metrics:
            raise ValueError(f"distance_metric deve ser um de: {valid_metrics}")


@dataclass
class IndexingConfig:
    """
    Definição de configurações para o pipeline de indexação.

    Attributes:
        source (str): Tipo de fonte dos dados (ex: parquet, hf, etc.).
        data_path (str): Caminho para os dados.
        id_field (str): Nome do campo que contém o ID do documento.
        text_field (str): Nome do campo que contém o texto do documento.
        metadata_fields (list): Campos de metadados a extrair.
        version (Optional[str]): Versão do dataset, se aplicável.
        batch_size (int): Tamanho do lote para processamento em batch.
    """

    source: str  # Source type: parquet, hf, etc.
    data_path: str
    id_field: str
    text_field: str
    metadata_fields: Optional[List[str]]
    version: Optional[str] = None
    batch_size: int = 16
    skip_existing: bool = True
    force_reindex: bool = False

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size deve ser positivo")

        if not self.source:
            raise ValueError("source não pode estar vazio")

        valid_sources = ["parquet", "hf"]
        if self.source not in valid_sources:
            raise ValueError(f"source deve ser um de: {valid_sources}")

        if not self.data_path:
            raise ValueError("data_path não pode estar vazio")

        if not self.id_field:
            raise ValueError("id_field não pode estar vazio")

        if not self.text_field:
            raise ValueError("text_field não pode estar vazio")

        if self.metadata_fields is None:
            self.metadata_fields = []

    @classmethod
    def from_dict(cls, config_dict: dict) -> "IndexingConfig":
        """
        Cria uma instância de IndexingConfig a partir de um dicionário de configurações.

        Params:
            config_dict (dict): Dicionário contendo as configurações.

        Returns:
            IndexingConfig: Instância configurada.
        """
        return cls(
            source=config_dict.get("source", ""),
            data_path=config_dict.get("data_path", ""),
            id_field=config_dict.get("id_field", ""),
            text_field=config_dict.get("text_field", ""),
            metadata_fields=config_dict.get("metadata_fields", []),
            version=config_dict.get("version"),
            batch_size=config_dict.get("batch_size", 16),
            skip_existing=config_dict.get("skip_existing", True),
            force_reindex=config_dict.get("force_reindex", False),
        )

    def to_dict(self) -> dict:
        """
        Converte a configuração para um dicionário.

        Returns:
            dict: Dicionário representando a configuração.
        """
        return {
            "source": self.source,
            "data_path": self.data_path,
            "id_field": self.id_field,
            "text_field": self.text_field,
            "metadata_fields": self.metadata_fields,
            "version": self.version,
            "batch_size": self.batch_size,
            "skip_existing": self.skip_existing,
            "force_reindex": self.force_reindex,
        }


@dataclass
class IndexResult:
    """
    Resultado da operação de indexação.

    Attributes:
        indexed_count (int): Número de documentos indexados com sucesso.
        skipped_count (int): Número de documentos que foram pulados.
        errors (List[str]): Lista de mensagens de erro encontradas durante a indexação.
    """

    doc_id: str
    status: Literal["indexed", "skipped", "failed"]
    message: Optional[str] = None
    chunks_total: int = 0
    chunks_indexed: int = 0


@dataclass
class VectorIndexConfig:
    """
    Configuração mestre do pipeline de indexação vetorial.

    Attributes:
        model (ModelConfig): Configurações do modelo de embeddings.
        chunker (ChunkerConfig): Configurações do Chunker.
        embedder (EmbedderConfig): Configurações do Embedder.
        vector_store (VectorStoreConfig): Configurações do VectorStore.
    """

    model: ModelConfig
    chunker: ChunkerConfig
    embedder: EmbedderConfig
    vector_store: VectorStoreConfig

    @classmethod
    def from_dict(cls, config_dict: dict) -> "VectorIndexConfig":
        """
        Cria uma instância de VectorIndexConfig a partir de um dicionário de configurações.

        Params:
            config_dict (dict): Dicionário contendo as configurações.

        Returns:
            VectorIndexConfig: Instância configurada.
        """
        model_cfg = ModelConfig(**config_dict.get("model", {}))
        chunker_cfg = ChunkerConfig(**config_dict.get("chunker", {}))
        embedder_cfg = EmbedderConfig(**config_dict.get("embedder", {}))
        vector_store_cfg = VectorStoreConfig(**config_dict.get("vector_store", {}))

        return cls(
            model=model_cfg,
            chunker=chunker_cfg,
            embedder=embedder_cfg,
            vector_store=vector_store_cfg,
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
        }
