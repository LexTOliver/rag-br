from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ComponentHealth(BaseModel):
    """Schema para o status de um componente individual da API."""

    name: str = Field(..., description="Nome do componente")
    status: Literal["Running", "Error", "Degraded"] = Field(
        ..., description="Status do componente"
    )
    latency_ms: Optional[float] = Field(
        None, description="Latência do componente em milissegundos"
    )
    details: Optional[str] = Field(
        None, description="Detalhes adicionais sobre o status do componente"
    )

    def to_dict(self) -> dict:
        """Converte o objeto ComponentHealth em um dicionário."""
        return {
            "name": self.name,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "details": self.details,
        }


class HealthResponse(BaseModel):
    """Schema para a resposta do health check da API."""

    status: Literal["Healthy", "Degraded", "Error"] = Field(
        ..., description="Status geral da API (e.g., 'Healthy' ou 'Error')"
    )
    version: str = Field(..., description="Versão atual da API")
    components: List[ComponentHealth] = Field(
        ..., description="Status dos componentes individuais da API"
    )
    message: Optional[str] = Field(
        None, description="Mensagem adicional em caso de erro"
    )
