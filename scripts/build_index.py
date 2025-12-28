"""
Vectorizing and indexing pipeline for RAG-BR using Quati dataset.

This script orchestrates the process of loading the Quati dataset,
chunking the documents, generating embeddings using a specified model,
and storing the resulting vectors in a vector database (Qdrant) for
efficient retrieval.

This module should be executed as a script.
"""

import argparse
import logging
from typing import Dict

import yaml
from datasets import Dataset, load_dataset
from tqdm import tqdm

from ingestion.load_dataset import load_parquet_dataset
from utils.logger import get_logger
from vectorize.config import IndexingConfig, VectorIndexConfig
from vectorize.vector_index import VectorIndex

logger = get_logger("logs/build_index_pipeline.log", level=logging.INFO)


def create_args() -> argparse.ArgumentParser:
    """
    Cria o parser de argumentos para a linha de comando.

    Returns:
        ArgumentParser: Parser with defined arguments.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline para vetorização e indexação do dataset Quati."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/index_config.yaml",
        help="Caminho para o arquivo de configuração YAML do índice.",
    )
    return parser.parse_args()


def load_quati_documents(indexing_config: Dict[str, str]) -> Dataset:
    """
    Carrega os documentos do dataset Quati de uma fonte específica e seleciona os dados necessários.

    Params:
        indexing_config (Dict[str, str]): Configurações de indexação contendo informações do dataset.

    Returns:
        Dataset: Dataset carregado do dataset Quati.
    """
    try:
        # Get dataset configuration
        data_path = indexing_config["data_path"]
        source = indexing_config["source"]

        # Load dataset based on source type
        if source == "parquet":  # Load from Parquet file
            dataset = load_parquet_dataset(data_path)
        elif source == "hf":  # Load from Hugging Face dataset
            dataset = load_dataset(
                indexing_config["dataset_name"],
                f"quati_{indexing_config['version']}_passages",
            )[f"quati_{indexing_config['version']}_passages"]
        else:
            raise ValueError(f"Unsupported data source: {source}")
    except Exception as e:
        logger.error(f"Error loading Quati documents: {e}", exc_info=True)
        logger.error(f"Dataset config: {indexing_config}, Source: {source}")
        logger.error(
            "Please ensure the dataset is available and the source type is correct."
        )
        raise

    # Select only necessary columns
    columns_to_keep = [
        indexing_config["id_field"],
        indexing_config["text_field"],
    ] + indexing_config.get("metadata_fields", [])

    dataset = dataset.select_columns(columns_to_keep)

    return dataset


def build_index(config: dict):
    """
    Pipeline for vectorizing and indexing the Quati dataset.

    Params:
        config (dict): Configurations for the pipeline extracted from the YAML file.
    """
    # Load configuration
    vector_index_config = VectorIndexConfig.from_dict(config["vector_index"])
    indexing_config = IndexingConfig.from_dict(config["indexing"])

    # Initialize VectorIndex
    vector_index = VectorIndex()
    vector_index.initialize(vector_index_config)

    # Load documents from Quati dataset
    logger.info(f"Loading Quati dataset from source: {indexing_config.source}, path: {indexing_config.data_path}.")
    dataset = load_quati_documents(indexing_config=indexing_config.to_dict())
    total_docs = len(dataset)
    logger.info(f"Loaded {total_docs} documents from Quati dataset.")

    # Calculate total number of batches for tqdm
    total_batches = (
        total_docs + indexing_config.batch_size - 1
    ) // indexing_config.batch_size

    # Index documents
    doc_count = 0
    indexed_doc_count = 0
    skipped_doc_count = 0
    error_doc_count = 0
    indexed_chunk_count = 0
    batch = []

    # Iterate over batches of documents for indexing
    logger.info("Starting indexing process...")
    logger.info("Indexing configuration:")
    logger.info(f"Force reindex: {indexing_config.force_reindex}")
    logger.info(f"Skip existing: {indexing_config.skip_existing}")
    logger.info(f"Batch size: {indexing_config.batch_size}")
    if vector_index.embedder.enable_local_cache:
        logger.info("Local cache for embeddings is enabled.")
    with tqdm(
        total=total_batches,
        desc="Indexing documents",
        unit="batch",
    ) as pbar:
        for batch in dataset.iter(batch_size=indexing_config.batch_size):
            # -- Process each document in the batch --
            batch_size = len(batch[indexing_config.id_field])
            for i in range(batch_size):
                # Extract document data
                doc_id = str(batch[indexing_config.id_field][i])
                text = batch[indexing_config.text_field][i]

                # Extract metadata safely
                metadata_fields = indexing_config.metadata_fields or []
                metadata = {
                    field: batch[field][i]
                    for field in metadata_fields
                    if field in batch
                }

                # Index the document
                res = vector_index.index_document(
                    doc_id=doc_id,
                    text=text,
                    metadata=metadata,
                    options=indexing_config,
                )

                # Update results
                if res.status == "indexed":
                    indexed_doc_count += 1
                    indexed_chunk_count += res.chunks_indexed
                    logger.debug(f"Indexed document ID {doc_id} successfully.")
                    logger.debug(f"Message: {res.message}.")
                    logger.debug(f"Chunks indexed for this doc: {res.chunks_indexed}/{res.chunks_total}.")
                elif res.status == "failed":
                    error_doc_count += 1
                    logger.error(f"Failed to index document ID {doc_id}.")
                    logger.debug(res.message)
                elif res.status == "skipped":
                    skipped_doc_count += 1
                    logger.debug(f"Skipped document ID {doc_id}: {res.message}.")
                    logger.debug(f"Message: {res.message}.")
                
            # Check and clear local cache if enabled
            if vector_index.embedder.enable_local_cache:
                cache_limit = vector_index.embedder.cache_limit_size
                current_cache_size = vector_index.embedder.get_cache_size()
                if current_cache_size > cache_limit:
                    logger.info(
                        f"Local cache size {current_cache_size} bytes exceeds limit of {cache_limit} bytes. Clearing cache."
                    )
                    vector_index.embedder.clear_local_cache()
            
            # Update progress bar after each batch
            pbar.update(1)
            doc_count += batch_size

            # Log progress every 500 documents
            if doc_count % 500 == 0:
                logger.info(f"Total chunks indexed so far: {indexed_chunk_count}.")
                logger.info(f"Total documents indexed so far {indexed_doc_count}.")
                logger.info(f"Total documents skipped so far: {skipped_doc_count}.")
                logger.info(f"Total documents failed so far: {error_doc_count}.")

    logger.info(f"Indexing complete! Total documents indexed: {indexed_doc_count}.")
    logger.info(f"Total chunks indexed: {indexed_chunk_count}.")

def main():
    """
    Pipeline de vetorização e indexação do dataset Quati.
    Utiliza configurações definidas em um arquivo YAML para carregar o
    dataset, gerar embeddings e armazenar os vetores em um banco de dados
    vetorial (Qdrant).

    Por limitações de capacidade computacional, utilizou-se o subconjunto
    de passages do dataset Quati que possui avaliações de relevância (reranker eval).
    """
    # LOAD CONFIGURATION
    args = create_args()
    with open(args.config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    # BUILD INDEX EXECUTION
    try:
        build_index(config_dict)
    except Exception as e:
        logger.error(f"Error building index: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
