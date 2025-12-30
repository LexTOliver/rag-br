"""
Dependency injection for IndexService.

Provides a function to create and inject IndexService instances.
"""
from api.dependencies.vector_index import get_vector_index
from api.services.index import IndexService


def get_index_service() -> IndexService:
    """Provides an instance of IndexService with the vector index injected."""
    vector_index = get_vector_index()
    return IndexService(vector_index=vector_index)