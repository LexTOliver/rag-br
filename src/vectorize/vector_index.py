"""
Module for managing Vector Indexing operations.

Provides a class that orchestrates chunking, embedding, and storage of text data
into a vector store for efficient retrieval.
"""

from typing import Dict, Optional

from vectorize.chunking import Chunker
from vectorize.config import IndexingConfig, IndexResult, VectorIndexConfig
from vectorize.embed import Embedder
from vectorize.vector_store import VectorStore


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

    def _assert_initialized(self):
        if not self._initialized:
            raise RuntimeError(
                "VectorIndex não inicializado. "
                "Chame `initialize()` antes de usar."
            )

    def index_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict,
        options: Optional[IndexingConfig] = None,
    ) -> IndexResult:
        """
        Indexa um único documento com base no texto e metadados fornecidos.

        Params:
            doc_id (str): Identificador único do documento.
            text (str): Texto do documento a ser indexado.
            metadata (Dict): Metadados associados ao documento.
            options (Optional[IndexingConfig]): Opções adicionais para controle de indexação.
        Returns:
            IndexResult: Resultado da operação de indexação.
        """
        # Assert initialization of components
        self._assert_initialized()

        # Get options or use defaults
        options = options or IndexingConfig()
        if not text or not text.strip():
            return IndexResult(
                doc_id=doc_id,
                status="skipped",
                message="Empty or whitespace-only text",
            )

        # Check if document is already indexed
        if options.skip_existing and not options.force_reindex:
            if self.vector_store.document_exists(doc_id, options.id_field):
                return IndexResult(
                    doc_id=doc_id,
                    status="skipped",
                    message="Document already indexed",
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
            payloads = []
            for idx, chunk in enumerate(chunks):
                payloads.append(
                    {
                        options.id_field: doc_id,
                        options.text_field: chunk,
                        "chunk_id": idx,
                        **metadata,
                    }
                )

            # -- Embedding with caching --
            hashes, embeddings = self.embedder.embed_with_cache(chunks)

            # -- Upsert into Vector Store --
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
                doc_id=doc_id,
                status="failed",
                message=f"{e.__class__.__name__}: {str(e)}",
            )

    def search(self, query: str, top_k: int = 5):
        """
        Busca vetorial a partir de uma query.

        Params:
            query (str): Texto da query para busca.
            top_k (int): Número de resultados a retornar.
        """
        self._assert_initialized()
        
        query_emb = self.embedder.embed([query], batch_size=1)[0]

        return self.vector_store.query_search(vector=query_emb, limit=top_k)
