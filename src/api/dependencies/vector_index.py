from api.core.lifespan import vector_index
from vectorize.vector_index import VectorIndex


def get_vector_index() -> VectorIndex:
    """Dependência para obter a instância do VectorIndex."""
    return vector_index
