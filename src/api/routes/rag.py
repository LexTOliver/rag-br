"""
RAG (Retrieval-Augmented Generation) related routes.

Defines the API endpoints for RAG operations.
"""

from fastapi import APIRouter

from api.schemas.rag import RAGRequest, RAGResponse

# Define the router
router = APIRouter(prefix="/rag")


# Define the RAG endpoint
@router.post(
    "/",
    response_model=RAGResponse,
    tags=["RAG"],
    summary="Pipeline RAG Completo",
)
def generate_rag_response(request: RAGRequest) -> RAGResponse:
    """
    Gera uma resposta RAG com base na requisição fornecida.
    """
    return {
        "answer": "RAG não implementado ainda",
        "sources": [],
        "context_used": False,
    }
