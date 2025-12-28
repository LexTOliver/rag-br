"""
Module for managing Vector Indexing operations.

Provides a class that orchestrates chunking, embedding, and storage of text data
into a vector store for efficient retrieval.
"""

from typing import Dict

import numpy as np

from vectorize.chunking import Chunker
from vectorize.config import VectorIndexConfig
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

    def __init__(
        self,
        config: VectorIndexConfig,
    ):
        """
        Inicializa o VectorIndex com os componentes necessários, configurados conforme as especificações.

        Params:
            config (VectorIndexConfig): Configurações completa do pipeline.
        """
        self.config = config

        # Set indexing config
        self.indexing = config.indexing

        # Initialize chunker
        self.chunker = Chunker(
            model_name=config.model.model_name,
            config=config.chunker,
            device=config.model.device,
        )

        # Initialize embedder
        self.embedder = Embedder(
            model_name=config.model.model_name,
            config=config.embedder,
            device=config.model.device,
        )

        # Get embedding dimension
        dim = self.embedder.model.get_sentence_embedding_dimension()

        # Initialize vector store
        self.vector_store = VectorStore(
            collection_name=config.vector_store.collection_name,
            vector_size=dim,
            distance_metric=config.vector_store.distance_metric,
            path=config.vector_store.path,
        )

    def index_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict,
    ):
        """
        Indexa um único documento.

        Params:
            doc_id (str): Identificador único do documento.
            text (str): Texto do documento a ser indexado.
            metadata (Dict): Metadados associados ao documento.
        """
        try:
            # Chunk the document text
            chunks = self.chunker.chunk(text)

            if not chunks:
                return

            # Prepare payloads for embedding and storage
            payloads = []
            for idx, chunk in enumerate(chunks):
                payloads.append(
                    {"doc_id": doc_id, "chunk_id": idx, "text": chunk, **metadata}
                )

            # Generate embeddings with caching
            hashes, embeddings = self.embedder.embed_with_cache(
                chunks, self.vector_store
            )

            # Filter invalid points before upsert
            valid_points = []
            dropped = 0

            for h, emb, payload in zip(hashes, embeddings, payloads):
                # VVerify if embedding is None
                if emb is None:
                    dropped += 1
                    continue

                # Convert to numpy array
                emb_np = np.asarray(emb)

                # Verify if empty
                if emb_np.size == 0 or emb_np.ndim == 0:
                    dropped += 1
                    continue

                # Add valid point
                valid_points.append(
                    {
                        "id": h,
                        "vector": emb_np,
                        "payload": payload,
                    }
                )

            # If no valid points, skip upsert
            if not valid_points:
                return

            # Upsert into vector store
            self.vector_store.upsert(
                points=[
                    {
                        "id": hash,
                        "vector": embedding,
                        "payload": payload,
                    }
                    for hash, embedding, payload in zip(hashes, embeddings, payloads)
                ]
            )
        except Exception as e:
            raise RuntimeError(f"Error indexing document {doc_id}: {e}") from e

    def search(self, query: str, top_k: int = 5):
        """
        Busca vetorial a partir de uma query.

        Params:
            query (str): Texto da query para busca.
            top_k (int): Número de resultados a retornar.
        """
        query_emb = self.embedder.embed([query], batch_size=1)[0]

        return self.vector_store.query_search(vector=query_emb, limit=top_k)
