"""
Schemas for RAG (Retrieval-Augmented Generation) API requests and responses.

Defines the data models for RAG requests and responses using Pydantic.
"""

from pydantic import BaseModel, Field


# TODO: Add more fields as necessary
class RAGRequest(BaseModel):
    """Schema para requisição do pipeline RAG."""

    query: str = Field(..., description="Consulta do usuário para o processamento RAG.")
    context: str = Field(
        ..., description="Informações de contexto para o processamento RAG."
    )


class RAGResponse(BaseModel):
    """Schema para resposta do pipeline RAG."""

    answer: str = Field(..., description="Resposta gerada pelo pipeline RAG.")
    sources: list[str] = Field(
        ..., description="Fontes utilizadas para gerar a resposta."
    )
    context_used: bool = Field(
        False, description="Indica se o contexto foi utilizado no processamento."
    )
