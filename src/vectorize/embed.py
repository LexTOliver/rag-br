"""
Embedding module for document vectorization tasks.

This module provides a class to generate embeddings for text data
using pre-trained transformer models.
"""

import hashlib
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from vectorize.config import EmbedderConfig
from vectorize.vector_store import VectorStore


def text_hash(text: str) -> str:
    """
    Gera hash estável baseado no conteúdo textual.
    Útil para identificar textos de forma única.
    Utiliza o algoritmo MD5 para gerar o hash.

    Params:
        text (str): Texto de entrada para gerar o hash.

    Returns:
        str: Hash MD5 do texto fornecido.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class Embedder:
    """
    Classe para geração de embeddings de textos utilizando modelos pré-treinados.
    Aplica cache de embeddings localmente ou via VectorStore para evitar recomputação de embeddings.

    Attributes:
        model (SentenceTransformer): Modelo pré-treinado para geração de embeddings.
        batch_size (int): Tamanho do lote para processamento em batch.
        local_cache_dir (str): Diretório para cache local de embeddings.
        enable_local_cache (bool): Habilita ou desabilita o cache local de embeddings

    Methods:
        embed(texts: List[str]) -> np.ndarray:
            Gera embeddings para uma lista de textos.
        embed_with_cache(
            texts: List[str],
            vector_store: VectorStore,
        ) -> Dict[str, np.ndarray]:
            Gera embeddings de textos, utilizando cache (local e VectorStore) para evitar recomputação.
        clear_local_cache():
            Limpa todo o cache local de embeddings.
        clear_cached_embedding(text_hash: str):
            Remove um embedding específico do cache local.
        get_cache_size() -> int:
            Retorna o tamanho total do cache local em bytes.
    """

    def __init__(
        self,
        model_name: str,
        config: EmbedderConfig,
        device: str = "cpu",
    ):
        """
        Inicializa o Embedder com um modelo específico.

        Params:
            model_name (str): Nome do modelo pré-treinado.
            config (EmbedderConfig): Configurações do Embedder.
            device (str): Dispositivo para computação ('cpu' ou 'cuda').
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = config.batch_size
        self.enable_local_cache = config.enable_local_cache

        if self.enable_local_cache:
            self.local_cache_dir = config.local_cache_dir
            os.makedirs(self.local_cache_dir, exist_ok=True)

    def embed(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """
        Gera embeddings para uma lista de textos.

        Params:
            texts (List[str]): Lista de textos a serem embutidos.

        Returns:
            np.ndarray: Matriz de embeddings onde cada linha corresponde a um texto.
        """
        if batch_size is None:
            batch_size = self.batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # For equalizing search methods
        )
        return embeddings

    def _load_existing_embeddings(
        self,
        hashes: List[str],
        vector_store: VectorStore,
    ) -> Dict[str, np.ndarray]:
        """
        Verifica e recupera embeddings existentes no VectorStore.

        Params:
            hashes (List[str]): Hashes correspondentes aos textos.
            vector_store (VectorStore): Store vetorial (Qdrant).

        Returns:
            Dict[str, np.ndarray]:
                - embeddings recuperados mapeados por seus hashes
        """
        embeddings_dict = {}

        # For each hash, check if it exists in the vector store
        for h in hashes:
            result = vector_store.search_by_id(h)
            if result and len(result) > 0:
                vector = result[0].vector
                if vector is not None:
                    emb = np.asarray(vector)
                    if emb.size > 0 and emb.ndim > 0:
                        embeddings_dict[h] = emb

        return embeddings_dict

    def _load_cached_embeddings(
        self,
        hashes: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Verifica e recupera embeddings armazenados localmente no diretório de cache.

        Params:
            hashes (List[str]): Lista de hashes correspondentes aos textos.

        Returns:
            Dict[str, np.ndarray]:
                - embeddings carregados mapeados por seus hashes
        """
        embeddings_dict = {}

        # For each hash, check if a cached file exists
        for h in hashes:
            cache_path = os.path.join(self.local_cache_dir, f"{h}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    emb = pickle.load(f)
                    emb_np = np.asarray(emb)
                    if emb_np.size > 0:
                        embeddings_dict[h] = emb_np

        return embeddings_dict

    def _cache_embeddings_locally(
        self,
        hashes: List[str],
        embeddings: np.ndarray,
    ):
        """
        Armazena embeddings localmente no diretório especificado para cache.

        Params:
            hashes (List[str]): Lista de hashes correspondentes aos textos.
            embeddings (np.ndarray): Matriz de embeddings a serem armazenados.
        """
        for h, emb in zip(hashes, embeddings):
            cache_path = os.path.join(self.local_cache_dir, f"{h}.pkl")
            with open(cache_path, "wb") as f:
                pickle.dump(emb, f)

    def clear_local_cache(self):
        """
        Limpa todo o cache local de embeddings.
        Útil para liberar espaço ou forçar a recomputação de embeddings.
        """
        if os.path.exists(self.local_cache_dir):
            import shutil

            shutil.rmtree(self.local_cache_dir)
            os.makedirs(self.local_cache_dir, exist_ok=True)

    def clear_cached_embedding(self, text_hash: str):
        """
        Remove um embedding específico do cache local.
        Útil para forçar a recomputação de um embedding específico.

        Params:
            text_hash (str): Hash do texto a remover do cache.
        """
        cache_path = os.path.join(self.local_cache_dir, f"{text_hash}.pkl")
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def get_cache_size(self) -> int:
        """
        Retorna o tamanho total do cache local em bytes.
        Pode ser usado para monitorar o uso de espaço em disco.

        Returns:
            int: Tamanho em bytes.
        """
        if not os.path.exists(self.local_cache_dir):
            return 0
        return sum(
            os.path.getsize(os.path.join(self.local_cache_dir, f))
            for f in os.listdir(self.local_cache_dir)
        )

    def embed_with_cache(
        self,
        texts: List[str],
        vector_store: VectorStore,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Gera embeddings de textos, de maneira eficiente, utilizando divisão em lotes e cache (local e VectorStore).
        Busca e reutiliza embeddings existentes no VectorStore e no cache local antes de computar novos embeddings.
        Computa embeddings apenas para textos que não possuem embeddings armazenados.

        Params:
            texts (List[str]): Textos de entrada.
            vector_store (VectorStore): Store vetorial (Qdrant).

        Returns:
            Tuple[List[str], np.ndarray]:
                - "hashes": Hashes correspondentes aos textos.
                - "embeddings": Embeddings gerados para os textos.
        """
        # Get hashes from texts
        hashes = [text_hash(text) for text in texts]

        # Verify embeddings existing in VectorStore
        existing_embeddings = self._load_existing_embeddings(hashes, vector_store)

        # Get remaining hashes to check in local cache
        remaining_hashes = [h for h in hashes if h not in existing_embeddings]
        # Verify embeddings existing in local cache
        cached_embeddings = (
            self._load_cached_embeddings(remaining_hashes)
            if self.enable_local_cache
            else {}
        )

        # Get remaining hashes to embed
        hashes_to_embed = [h for h in remaining_hashes if h not in cached_embeddings]

        # Embed remaining texts in batches
        new_embeddings = {}
        for i in range(
            0, len(hashes_to_embed), self.batch_size
        ):  # Iterate over batches
            # -- Get hashes and corresponding texts for the current batch
            batch_hashes = hashes_to_embed[i : i + self.batch_size]
            batch_texts = [texts[hashes.index(h)] for h in batch_hashes]

            # -- Compute embeddings for the current batch
            batch_embeddings = self.embed(batch_texts)

            # -- Cache newly computed embeddings locally
            if self.enable_local_cache:
                self._cache_embeddings_locally(batch_hashes, batch_embeddings)

            # -- Store new embeddings
            for h, emb in zip(batch_hashes, batch_embeddings):
                new_embeddings[h] = emb

        # Combine all embeddings keeping the original order
        embeddings = []
        for h in hashes:
            if h in existing_embeddings:
                embeddings.append(existing_embeddings[h])
            elif h in cached_embeddings:
                embeddings.append(cached_embeddings[h])
            else:
                embeddings.append(new_embeddings[h])

        return hashes, np.stack(embeddings)
