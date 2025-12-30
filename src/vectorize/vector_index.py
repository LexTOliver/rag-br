"""
Module for managing Vector Indexing operations.

Provides a class that orchestrates chunking, embedding, and storage of text data
into a vector store for efficient retrieval.
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, Literal, Optional

from vectorize.chunking import Chunker
from vectorize.config import VectorIndexConfig
from vectorize.embed import Embedder
from vectorize.vector_store import VectorStore


def _hash_document(text: str) -> str:
    """
    Gera um hash MD5 para o texto do documento fornecido.

    Params:
        text (str): Texto do documento a ser hasheado.

    Returns:
        str: Hash MD5 do texto.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


@dataclass
class IndexResult:
    """
    Resultado da operação de indexação.

    Attributes:
        doc_id (str): ID do documento indexado, criado a partir do hash do texto.
        status (Literal["indexed", "skipped", "failed"]): Status da indexação
        message (Optional[str]): Mensagem adicional sobre o resultado da indexação.
        chunks_total (int): Número total de chunks gerados para o documento.
        chunks_indexed (int): Número de chunks efetivamente indexados.
    """

    doc_id: str
    status: Literal["indexed", "skipped", "failed"]
    message: Optional[str] = None
    chunks_total: int = 0
    chunks_indexed: int = 0


class VectorIndex:
    """
    Classe mestre para gerenciamento da vetorização e indexação vetorial de dados textuais.
    Orquestra o pipeline de indexação vetorial, incluindo:
        - Chunking textual
        - Geração de embeddings
        - Armazenamento em VectorStore

    Attributes:
        config (VectorIndexConfig): Configurações completas do pipeline de indexação vetorial.
        chunker (Chunker): Responsável por dividir textos em chunks.
        embedder (Embedder): Responsável por gerar embeddings dos textos.
        vector_store (VectorStore): Armazena os embeddings e permite buscas vetoriais.
    """

    def __init__(self):
        self.chunker: Optional[Chunker] = None
        self.embedder: Optional[Embedder] = None
        self.vector_store: Optional[VectorStore] = None
        self.config: Optional[VectorIndexConfig] = None
        self._initialized: bool = False

    def initialize(self, config: VectorIndexConfig):
        """
        Inicializa os componentes do pipeline de indexação vetorial com base na configuração fornecida.
        Deve inicializar apenas uma vez.

        Params:
            config (VectorIndexConfig): Configurações completas do pipeline de indexação vetorial.
        """
        if self._initialized:
            return

        # Store the configuration
        self.config = config

        # Initialize Chunker
        self.chunker = Chunker(config.model, config.chunker)

        # Initialize Embedder
        self.embedder = Embedder(config.model, config.embedder)

        # Initialize VectorStore
        dim = self.embedder.model.get_sentence_embedding_dimension()
        vector_store_config = config.vector_store
        self.vector_store = VectorStore(
            collection_name=vector_store_config.collection_name,
            path=vector_store_config.path,
            vector_size=dim,
            distance_metric=vector_store_config.distance_metric,
        )
        self._initialized = True

    def assert_initialized(self):
        if not self._initialized:
            raise RuntimeError(
                "VectorIndex não inicializado. Chame `initialize()` antes de usar."
            )
        else:
            return True

    def index_document(
        self,
        text: str,
        metadata: Dict,
        skip_existing: bool = True,
        force_reindex: bool = False,
    ) -> IndexResult:
        """
        Indexa um único documento com base no texto e metadados fornecidos.

        Params:
            text (str): Texto do documento a ser indexado.
            metadata (Dict): Metadados associados ao documento.
            skip_existing (bool): Se deve pular a indexação se o documento já existir.
            force_reindex (bool): Se deve forçar a reindexação mesmo que o documento já exista.
        Returns:
            IndexResult: Resultado da operação de indexação.
        """
        # Assert initialization of components
        self.assert_initialized()

        # Validate text
        if not text or not text.strip():
            return IndexResult(
                doc_id="N/A",
                status="skipped",
                message="Empty or whitespace-only text",
            )

        # Generate document ID based on text hash
        # TODO: Improve document ID strategy if needed
        doc_id = _hash_document(text)

        # Check if document is already indexed
        if skip_existing and not force_reindex:
            chunks_found = self.vector_store.document_exists(doc_id, metadata)
            if chunks_found > 0:
                return IndexResult(
                    doc_id=doc_id,
                    status="skipped",
                    message="Document already indexed",
                    chunks_indexed=chunks_found,
                )

        # Start indexing process
        try:
            # -- Chunking --
            chunks = self.chunker.chunk(text)

            if not chunks:
                return IndexResult(
                    doc_id=doc_id,
                    status="skipped",
                    message="Chunking returned empty",
                )

            # Define payloads for each chunk
            # TODO: Remover chunk_text e text; manter apenas as referências dos documentos,
            payloads = []
            for idx, chunk in enumerate(chunks):
                payloads.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": idx,
                        "chunk_text": chunk,
                        **metadata,
                    }
                )

            # -- Embedding with caching --
            hashes, embeddings = self.embedder.embed_with_cache(chunks)

            # -- Upsert into Vector Store --
            # TODO: Improve id definition: concat doc_id + chunk_id + hash?
            points = []
            for h, emb, payload in zip(hashes, embeddings, payloads):
                points.append(
                    {
                        "id": h,
                        "vector": emb,
                        "payload": payload,
                    }
                )

            if not points:
                return IndexResult(
                    doc_id=doc_id,
                    status="skipped",
                    message="No points to upsert",
                    chunks_total=len(chunks),
                )

            self.vector_store.upsert(points)

            return IndexResult(
                doc_id=doc_id,
                status="indexed",
                message="Document indexed successfully",
                chunks_total=len(chunks),
                chunks_indexed=len(points),
            )

        except Exception as e:
            return IndexResult(
                doc_id=doc_id if "doc_id" in locals() else "N/A",
                status="failed",
                message=f"{e.__class__.__name__}: {str(e)}",
            )

    def search(self, query: str, top_k: int = 5):
        """
        Busca vetorial a partir de uma query.

        Params:
            query (str): Texto da query para busca.
            top_k (int): Número de resultados a retornar.

        Returns:
            Lista de pontos similares encontrados.
        """
        self.assert_initialized()

        query_emb = self.embedder.embed([query], batch_size=1)[0]

        return self.vector_store.query_search(vector=query_emb, limit=top_k)
