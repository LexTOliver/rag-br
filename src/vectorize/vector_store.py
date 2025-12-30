"""
Module for managing Vector DB operations, including creation, insertion, and search.

Provides a class to handle vector store operations such as checking
for the existence of vectors based on their IDs.
"""

from typing import Any, Dict, List, Literal

import numpy as np
from qdrant_client import QdrantClient, models


class VectorStore:
    """
    Classe para gerenciamento de operações em banco de dados vetorial.
    Abstração sobre Qdrant para persistência e consulta de embeddings.
    # TODO: Adicionar persistência remota via Qdrant Cloud ou via Docker

    Attributes:
        client (QdrantClient): Cliente do Qdrant para interagir com o banco de dados vetorial.
        collection_name (str): Nome da coleção no Qdrant onde os vetores são armazenados.

    Methods:
        exists(vector_id: str) -> bool:
            Verifica se um vetor com o ID especificado existe na coleção.
        upsert(ids: List[str], vectors, payloads: List[Dict[str, Any]]):
            Insere ou atualiza pontos na coleção vetorial.
        search(vector, limit: int = 5):
            Realiza uma busca por vetores similares na coleção, retornando os mais próximos.
    """

    def __init__(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: Literal["cosine", "euclidean", "dot"] = "cosine",
        path: str = "data/qdrant",
    ):
        """
        Inicializa o VectorStore com um cliente Qdrant local e o nome da coleção.
        TODO: Adicionar suporte para conexão remota via Qdrant Cloud ou Docker.

        Params:
            collection_name (str): Nome da coleção no Qdrant.
            vector_size (int): Tamanho dos vetores na coleção.
            distance_metric (str): Métrica de distância para similaridade ('cosine', 'euclidean', 'dot').
            path (str): Caminho para armazenamento local do Qdrant.
        """
        self.collection_name = collection_name
        self.client = QdrantClient(path=path)

        self._ensure_collection(vector_size, distance_metric)

    def _ensure_collection(self, vector_size: int, distance_metric: str):
        """
        Garante que a coleção especificada exista no Qdrant.
        Verifica se a coleção já existe; se não, cria uma nova coleção com os parâmetros fornecidos.

        Params:
            vector_size (int): Tamanho dos vetores na coleção.
            distance_metric (str): Métrica de distância para similaridade ('cosine', 'euclidean', 'dot').
        """
        # Validate distance metric
        valid_metrics = ["cosine", "euclidean", "dot"]
        if distance_metric not in valid_metrics:
            raise ValueError(f"distance_metric deve ser um de: {valid_metrics}")

        # Check if collection exists; if not, create it
        if self.collection_name not in [
            c.name for c in self.client.get_collections().collections
        ]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance[distance_metric.upper()]
                ),
            )

    def upsert(self, points: List[Dict[str, Any]]):
        """
        Insere ou atualiza pontos na coleção vetorial.

        Params:
            points (List[Dict[str, Any]]): Lista de pontos a serem inseridos/atualizados.
                Cada ponto deve conter 'id', 'vector' e 'payload'.
        """
        # Convert dict points to PointStruct
        points = [
            models.PointStruct(
                id=point["id"],
                vector=point["vector"].tolist(),
                payload=point["payload"],
            )
            for point in points
        ]

        # Upsert points into the collection
        self.client.upsert(collection_name=self.collection_name, points=points)

    def query_search(
        self,
        vector: np.ndarray,
        limit: int = 5,
        with_vectors: bool = False,
        with_payload: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Realiza uma busca por vetores similares na coleção, retornando os mais próximos.
        TODO: Adicionar filtros de payload para buscas mais refinadas.

        Params:
            vector (np.ndarray): Vetor de consulta.
            limit (int): Número máximo de resultados a serem retornados.
            with_vectors (bool): Se deve retornar os vetores junto com os resultados.
            with_payload (bool): Se deve retornar os payloads junto com os resultados.

        Returns:
            List[Dict[str, Any]]: Lista de pontos similares encontrados.
        """
        return self.client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            limit=limit,
            with_vectors=with_vectors,
            with_payload=with_payload,
        ).points

    def search_by_ids(
        self,
        vector_ids: List[str],
        with_vectors: bool = False,
        with_payload: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Busca vetores na coleção pelos seus IDs.

        Params:
            vector_ids (List[str]): Lista de IDs dos vetores a serem buscados.
            with_vectors (bool): Se deve retornar os vetores junto com os resultados.
            with_payload (bool): Se deve retornar os payloads junto com os resultados.

        Returns:
            Lista de pontos encontrados com os IDs especificados.
        """
        return self.client.retrieve(
            collection_name=self.collection_name,
            ids=vector_ids,
            with_vectors=with_vectors,
            with_payload=with_payload,
        )

    def search_by_filter(
        self,
        filter: Dict[str, Any],
        limit: int = 5,
        with_vectors: bool = False,
        with_payload: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Busca vetores na coleção com base em um filtro de payload.

        Params:
            filter (Dict[str, Any]): Dicionário representando o filtro de payload.
            limit (int): Número máximo de resultados a serem retornados.
            with_vectors (bool): Se deve retornar os vetores junto com os resultados.
            with_payload (bool): Se deve retornar os payloads junto com os resultados.

        Returns:
            List[Dict[str, Any]]: Lista de pontos encontrados que correspondem ao filtro.
        """
        # Create Qdrant filter from the provided dictionary
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
                for key, value in filter.items()
            ]
        )

        # Perform the scroll query with the constructed filter
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_vectors=with_vectors,
            with_payload=with_payload,
        )

        return results[0]

    def document_exists(self, doc_id: str, metadata: Dict[str, Any]) -> int:
        """
        Verifica se um documento já possui um vetor indexado no VectorStore.
        A verificação é feita buscando o hash e metadados associados ao documento.

        Params:
            doc_id (str): ID do documento a ser verificado.
            metadata (Dict[str, Any]): Metadados do documento a ser verificado.
        Returns:
            int: Número de vetores encontrados correspondentes ao documento.

        """
        # Create filter with doc_id and metadata
        filter_dict = {"doc_id": doc_id}
        filter_dict.update(metadata)
        results = self.search_by_filter(filter=filter_dict, limit=50)

        return len(results)
