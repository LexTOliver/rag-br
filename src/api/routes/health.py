import asyncio
from datetime import datetime

from fastapi import APIRouter

from api.core.settings import settings
from api.dependencies.query_service import query_search_service
from api.dependencies.vector_index import get_vector_index
from api.schemas.health import ComponentHealth, HealthResponse

COMPONENT_TIMEOUT = 2.0  # seconds

# Initialize API router
router = APIRouter(prefix="/health")


# Helper function to check component health
async def check_component(name: str, coro) -> ComponentHealth:
    """
    Helper function to check the health of a component asynchronously.

    Params:
        name (str): The name of the component.
        coro: The coroutine to execute for the health check.

        Returns: ComponentHealth object representing the health status of the component.
    """
    started = datetime.now()
    try:
        _ = await asyncio.wait_for(coro, timeout=COMPONENT_TIMEOUT)
        duration_ms = (datetime.now() - started).total_seconds() * 1000
        return ComponentHealth(
            name=name,
            status="Running",
            latency_ms=duration_ms,
            details="Ok",
        )
    except asyncio.TimeoutError:
        duration_ms = (datetime.now() - started).total_seconds() * 1000
        return ComponentHealth(
            name=name,
            status="Degraded",
            latency_ms=duration_ms,
            details="Component check timed out",
        )
    except Exception as e:
        duration_ms = (datetime.now() - started).total_seconds() * 1000
        return ComponentHealth(
            name=name,
            status="Error",
            latency_ms=duration_ms,
            details=str(e),
        )


# Define routes
@router.get(
    "/",
    tags=["Health System Check"],
    summary="Checa o status de execução da API.",
    response_model=HealthResponse,
)
async def health_check() -> HealthResponse:
    """Verifica o status de execução dos componentes da API."""
    try:
        components_status = []

        # -- Check Vector Index Health --
        components_status.append(
            await check_component(
                "Vector Index",
                asyncio.to_thread(lambda: get_vector_index().assert_initialized()),
            )
        )

        # -- Check Query Service Health --
        components_status.append(
            await check_component(
                "Query Service",
                asyncio.to_thread(
                    lambda: query_search_service().search("health check", top_k=1)
                ),
            )
        )

        # TODO: Search how to add Index Service health check carefully without indexing data each time
        # TODO: Add more component health checks as needed

        # Determine overall status
        overall_status = (
            "Healthy"
            if all(component.status == "Running" for component in components_status)
            else "Degraded"
        )

        return HealthResponse(
            status=overall_status,
            version=settings.version,
            components=components_status,
        )
    except Exception as e:
        return HealthResponse(
            status="Error",
            version=settings.version,
            components=[],
            message=f"Health check failed: {str(e)}",
        )
