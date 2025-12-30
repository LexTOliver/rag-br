"""
Dependency injection for QueryService.

Provides a function to create and inject QueryService instances.
"""

from api.dependencies.vector_index import get_vector_index
from api.services.query import QueryService


def query_search_service() -> QueryService:
    """Provides an instance of QueryService with the vector index injected."""
    vector_index = get_vector_index()
    return QueryService(vector_index=vector_index)
