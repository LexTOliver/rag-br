from typing import Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema para a resposta do health check da API."""

    status: Literal["healthy", "error"] = Field(
        ..., description="Status geral da API (e.g., 'healthy' ou 'error')"
    )
    version: str = Field(..., description="Versão atual da API")
    components: dict = Field(
        ..., description="Status dos componentes individuais da API"
    )
    message: Optional[str] = Field(
        None, description="Mensagem adicional em caso de erro"
    )
