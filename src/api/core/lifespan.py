"""
Lifecycle management for the FastAPI application.

Defines startup and shutdown procedures, including initialization steps for:
- Loading configuration
- Setting up the VectorIndex
"""

from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI

from api.core.settings import settings
from vectorize.config import VectorIndexConfig
from vectorize.vector_index import VectorIndex

# from utils.logger import get_logger
# logger = get_logger("logs/api.log")
# TODO: Implementar logger para api


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação FastAPI."""
    # STARTUP
    try:
        # logger.info("Starting RAG-BR API...")
        vector_index = VectorIndex()

        # Load configuration
        config_path = settings.index_config_path
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Initialize VectorIndex
        vector_index_config = VectorIndexConfig.from_dict(config["vector_index"])
        vector_index.initialize(vector_index_config)
        app.state.vector_index = vector_index
        # logger.info("VectorIndex initialized!")

        # TODO: Initialize handlers
        yield
    except Exception as e:
        print(f"Failed to start API: {str(e)}")
        # logger.error(f"Failed to start API: {str(e)}", exc_info=True)
        raise
    # SHUTDOWN
    finally:
        # Perform any necessary cleanup
        if getattr(vector_index, "embedder", None):
            vector_index.embedder.clear_local_cache()
        # logger.info("Shutting down RAG-BR API...")
