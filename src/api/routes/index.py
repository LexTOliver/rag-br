from fastapi import APIRouter, Depends

from api.dependencies.index_service import get_index_service
from api.schemas.index import IndexRequest, IndexResponse
from api.services.index import IndexService

# Initialize API router
router = APIRouter(prefix="/index")


# Define routes
@router.post(
    "/",
    tags=["Index"],
    summary="Vetoriza e indexa novo documento.",
    response_model=IndexResponse,
)
async def index_document(
    request: IndexRequest,
    service: IndexService = Depends(get_index_service),
) -> IndexResponse:
    """
    Vetoriza e indexa um novo documento na base de dados vetorial usando o VectorIndex.
    """
    res = service.index_document(
        document=request.document, metadata=request.metadata
    )

    # Return response
    return IndexResponse(
        doc_id=res["doc_id"], status=res["status"], message=res["message"], num_chunks=res["chunks_indexed"]
    )
