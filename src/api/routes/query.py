from fastapi import APIRouter, Depends

from api.dependencies.vector_index import get_vector_index
from api.schemas.query import QueryRequest, QueryResponse, QueryResult
from vectorize.vector_index import VectorIndex

# Define router
router = APIRouter(prefix="/query")


# Define query endpoint
@router.post(
    "/",
    tags=["Search"],
    summary="Realiza uma consulta na base de dados vetorial.",
    response_model=QueryResponse,
)
def query_search(
    request: QueryRequest,
    vector_index: VectorIndex = Depends(get_vector_index),
) -> QueryResponse:
    """
    Realiza uma consulta na base de dados vetorial usando o VectorIndex.
    """
    # TODO: Implementar service de search
    # Perform search
    res = vector_index.search(request.query, top_k=request.top_k)

    # Parse results into QueryResponse
    parsed = [
        QueryResult(
            id=str(r.id),
            score=r.score,
            payload=r.payload,
        )
        for r in res
    ]

    return QueryResponse(results=parsed)
