from fastapi import APIRouter, Depends

from api.dependencies.vector_index import get_vector_index

# from api.services.indexing import index_document_service
from api.schemas.index import IndexRequest, IndexResponse
from vectorize.vector_index import VectorIndex

# Initialize API router
router = APIRouter(prefix="/index")


# Define routes
@router.post(
    "/",
    tags=["Index"],
    summary="Vetoriza e indexa novo documento.",
    response_model=IndexResponse,
)
def index_document(
    request: IndexRequest,
    vector_index: VectorIndex = Depends(get_vector_index),
) -> IndexResponse:
    """
    Vetoriza e indexa um novo documento na base de dados vetorial usando o VectorIndex.
    """
    # TODO: Implementar index service
    # Vectorize and index document
    res = vector_index.index_document(request.document, request.metadata)

    # Parse status and message
    status = "success" if res.status == "indexed" else "failure"
    message = f"Document {res.status}. {res.message or ''}".strip()

    # Return response
    return IndexResponse(doc_id=res.doc_id, status=status, message=message, num_chunks=res.chunks_indexed)