from fastapi import Request

from vectorize.vector_index import VectorIndex


def get_vector_index(request: Request) -> VectorIndex:
    """Dependência para obter a instância do VectorIndex."""
    return request.app.state.vector_index
