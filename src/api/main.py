"""
Main application file for the RAG-BR API.
Sets up the FastAPI app, middleware, and routes.

Includes CORS middleware and integrates the lifespan context manager.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.lifespan import lifespan
from api.core.settings import settings
from api.routes import health, index, query, rag  # debug

# Initialize FastAPI app
app = FastAPI(
    title=settings.title,
    version=settings.version,
    description=settings.description,
    lifespan=lifespan,
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Include routes
app.include_router(health.router)
app.include_router(index.router)
app.include_router(query.router)
app.include_router(rag.router)
# app.include_router(debug.router)


@app.get("/", tags=["Root"], summary="RAG-BR Root Endpoint")
async def read_root():
    """Root endpoint da API RAG-BR."""
    return {"message": "Welcome to the RAG-BR API!"}
