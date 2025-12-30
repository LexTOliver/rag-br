"""
Service for handling query operations.

Provides functionality to perform searches on the vector index.
"""

from functools import lru_cache
from typing import Tuple

from api.schemas.query import QueryResult


class QueryService:
    """
    Service class for handling query operations on the vector index.

    Attributes:
        vector_index: The vector index instance used for searching.

    Methods:
        search(query: str, top_k: int) -> Tuple[QueryResult]:
            Performs a search on the vector index and returns the top_k results.
    """

    def __init__(self, vector_index):
        """
        Initializes the QueryService with the provided vector index.

        Params:
            vector_index: The vector index instance to be used for searching.
        """
        self.vector_index = vector_index

    @lru_cache(maxsize=128)
    def search(self, query: str, top_k: int) -> Tuple[QueryResult]:
        """
        Performs a search on the vector index and returns the top_k results.

        Params:
            query (str): The search query string.
            top_k (int): The number of top results to return.

        Returns:
            Tuple[QueryResult]: A tuple containing the top_k search results.
        """
        result = self.vector_index.search(query, top_k=top_k)

        return [
            QueryResult(
                id=str(r.id),
                score=r.score,
                payload=r.payload,
            )
            for r in result
        ]
