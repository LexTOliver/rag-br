from typing import Any, Dict, List

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ..., description="A consulta de texto para buscar na base de dados vetorial"
    )
    top_k: int = Field(
        5, description="Número de resultados principais a serem retornados"
    )


class QueryResult(BaseModel):
    id: str = Field(..., description="ID do trecho do documento retornado")
    score: float = Field(..., description="Pontuação de similaridade do documento")
    payload: Dict[str, Any] = Field(
        ..., description="Dados adicionais associados ao documento"
    )


class QueryResponse(BaseModel):
    results: List[QueryResult] = Field(
        ..., description="Lista de resultados da consulta"
    )
