from fastapi import APIRouter

from api.core.settings import settings
from api.dependencies.vector_index import get_vector_index
from api.schemas.health import HealthResponse

# Initialize API router
router = APIRouter(prefix="/health")


# Define routes
@router.get(
    "/",
    tags=["Health System Check"],
    summary="Checa o status de execução da API.",
    response_model=HealthResponse,
)
def health_check() -> HealthResponse:
    """Verifica o status de execução dos componentes da API."""
    try:
        vector_index = get_vector_index()
        components_status = {
            "Vector Index": "Running"
            if vector_index._assert_initialized() is None
            else "Not Running",
            # TODO: Add other components health checks here (RAG, DB, etc.)
        }

        return HealthResponse(
            status="healthy", version=settings.version, components=components_status
        )
    except Exception as e:
        return HealthResponse(
            status="error", version="0.1.0", components={}, message=str(e)
        )
        # TODO: Implement Error Handling
        # raise HTTPException(
        #     status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        #     detail="Service unavailable"
        # )
