"""
Schemas for query requests and responses.

Defines the data models for handling query operations in the API.
"""

from typing import Any, Dict, Tuple

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Schema para requisição de consulta."""

    query: str = Field(
        ..., description="A consulta de texto para buscar na base de dados vetorial"
    )
    top_k: int = Field(
        5, description="Número de resultados principais a serem retornados"
    )


class QueryResult(BaseModel):
    """Schema para resultado individual da consulta."""

    id: str = Field(..., description="ID do trecho do documento retornado")
    score: float = Field(..., description="Pontuação de similaridade do documento")
    payload: Dict[str, Any] = Field(
        ..., description="Dados adicionais associados ao documento"
    )


class QueryResponse(BaseModel):
    """Schema para resposta da consulta contendo os resultados."""

    results: Tuple[QueryResult] = Field(
        ..., description="Lista de resultados da consulta"
    )
