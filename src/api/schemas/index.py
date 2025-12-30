"""
Schemas for document indexing requests and responses.

This module defines the Pydantic models used for validating and serializing
the request and response data related to document indexing operations in the API.
"""
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    """Schema para a solicitação de indexação de documento."""

    document: str = Field(
        ...,
        description="O conteúdo do documento a ser indexado na base de dados vetorial",
    )
    metadata: Dict[str, Any] = Field(
        ..., description="Metadados adicionais associados ao documento"
    )


class IndexResponse(BaseModel):
    """Schema para a resposta de indexação de documento."""

    doc_id: str = Field(..., description="ID do documento indexado")
    status: Literal["success", "failure"] = Field(
        ..., description="Status da operação de indexação"
    )
    message: Optional[str] = Field(
        None, description="Mensagem adicional sobre o resultado da indexação"
    )
    num_chunks: Optional[int] = Field(
        None, description="Número de chunks gerados para o documento indexado"
    )
