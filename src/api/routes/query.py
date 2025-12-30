"""
API routes for handling query operations.

Defines the endpoint for performing searches on the vector index.
"""

from fastapi import APIRouter, Depends

from api.dependencies.query_service import query_search_service
from api.schemas.query import QueryRequest, QueryResponse
from api.services.query import QueryService

# Define router
router = APIRouter(prefix="/query")


# Define query endpoint
@router.post(
    "/",
    tags=["Search"],
    summary="Realiza uma consulta na base de dados vetorial.",
    response_model=QueryResponse,
)
async def query_search(
    request: QueryRequest,
    service: QueryService = Depends(query_search_service),
) -> QueryResponse:
    """Realiza uma consulta na base de dados vetorial."""
    results = service.search(query=request.query, top_k=request.top_k)

    return QueryResponse(results=results)
