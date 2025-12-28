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
from vectorize.config import VectorIndexConfig
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
        dataset_config = indexing_config["dataset"]
        source = indexing_config["source"]

        # Load dataset based on source type
        if source == "parquet":  # Load from Parquet file
            dataset = load_parquet_dataset(dataset_config["data_path"])
        elif source == "HF dataset":  # Load from Hugging Face dataset
            dataset = load_dataset(
                dataset_config["dataset_name"],
                f"quati_{dataset_config['version']}_passages",
            )[f"quati_{dataset_config['version']}_passages"]
        else:
            raise ValueError(f"Unsupported data source: {source}")
    except Exception as e:
        logger.error(f"Error loading Quati documents: {e}", exc_info=True)
        logger.error(f"Dataset config: {dataset_config}, Source: {source}")
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
    index_config = VectorIndexConfig.from_dict(config)

    # Initialize VectorIndex
    vector_index = VectorIndex(config=index_config)

    # Load documents from Quati dataset
    logger.info("Loading Quati dataset...")
    dataset = load_quati_documents(vector_index.indexing.__dict__)
    total_docs = len(dataset)
    logger.info(f"Loaded {total_docs} documents from Quati dataset.")

    # Calculate total number of batches for tqdm
    total_batches = (
        total_docs + vector_index.indexing.batch_size - 1
    ) // vector_index.indexing.batch_size

    # Index documents
    indexed_count = 0
    batch = []

    # Iterate over batches of documents for indexing
    with tqdm(
        total=total_batches,
        desc="Indexing documents",
        unit="batch",
    ) as pbar:
        for batch in dataset.iter(batch_size=vector_index.indexing.batch_size):
            # -- Process each document in the batch --
            batch_size = len(batch[vector_index.indexing.id_field])
            for i in range(batch_size):
                # Extract document data
                doc_id = str(batch[vector_index.indexing.id_field][i])
                text = batch[vector_index.indexing.text_field][i]

                # Extract metadata safely
                metadata_fields = vector_index.indexing.metadata_fields or []
                metadata = {
                    field: batch[field][i]
                    for field in metadata_fields
                    if field in batch
                }

                # Index the document
                vector_index.index_document(
                    doc_id=doc_id,
                    text=text,
                    metadata=metadata,
                )
                indexed_count += 1
                tqdm.write(f"Indexed document ID: {doc_id}")

            # Update progress bar after each batch
            pbar.update(1)

            # Log progress every 500 documents
            if indexed_count % 500 == 0:
                tqdm.write(f"Indexed {indexed_count}/{total_docs} documents")
                # logger.info(f"Indexed {indexed_count}/{total_docs} documents")

    logger.info(f"Indexing complete! Total documents indexed: {indexed_count}")


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
