"""
Dataset ingestion and loading pipelines for RAG-BR.

This script orchestrates the ingestion and preprocessing of
external datasets (MS MARCO and Quati), producing standardized
Parquet artifacts for downstream tasks such as reranker training
and evaluation.

This module should be executed as a script.
"""

import sys
import yaml
from argparse import ArgumentParser
from datasets import load_dataset, Dataset
from src.ingest.preprocess import preprocess_dataset, format_msmarco, format_quati
from src.utils.logger import get_logger

logger = get_logger("ingestion.log")


def load_msmarco(
    dataset_name: str = "microsoft/ms_marco",
    version: str = "v2.1",
    split: str = "train",
    num_proc: int = 1,
) -> Dataset:
    """
    Carrega o dataset MS MARCO da Hugging Face.

    Params:
        dataset_name (str): Nome do dataset na Hugging Face.
        version (str): Versão do MS MARCO ("v1" ou "v2.1").
        split (str): Divisão do dataset ("train", "validation", "test").
        num_proc (int): Número de processos para pré-processamento paralelo.

    Returns:
        Dataset: Objeto Dataset carregado.
    """
    # Load MS MARCO dataset from Hugging Face
    ds = load_dataset(dataset_name, version, split=split)

    # Format the dataset to standard schema
    ds = format_msmarco(ds, num_proc=num_proc)

    # Apply preprocessing
    ds = preprocess_dataset(ds, num_proc=num_proc)

    return ds


def load_quati(
    dataset_name: str = "unicamp-dl/quati", version: str = "1M", num_proc: int = 1
) -> Dataset:
    """
    Carrega o dataset Quati da Hugging Face.

    Params:
        dataset_name (str): Nome do dataset na Hugging Face.
        version (str): Versão do Quati ("1M" para 1 milhão de trechos).

    Returns:
        Dataset: Objeto Dataset carregado.
    """
    # Load Quati dataset from Hugging Face
    passages_ds = load_dataset(
        dataset_name, f"quati_{version}_passages", trust_remote_code=True
    )
    queries_ds = load_dataset(dataset_name, "quati_all_topics", trust_remote_code=True)
    qrels_ds = load_dataset(
        dataset_name, f"quati_{version}_qrels", trust_remote_code=True
    )

    # Format the dataset to standard schema
    ds = format_quati(
        passages_ds[f"quati_{version}_passages"],
        queries_ds["quati_all_topics"],
        qrels_ds[f"quati_{version}_qrels"],
        num_proc=num_proc,
    )

    # Apply preprocessing
    ds = preprocess_dataset(ds, label_divisor=3.0, num_proc=num_proc)

    return ds


def create_args() -> ArgumentParser:
    """
    Cria o parser de argumentos para a linha de comando.

    Returns:
        ArgumentParser: Parser de argumentos configurado.
    """
    parser = ArgumentParser(
        description="Pipeline de ingestão e pré-processamento dos datasets MS MARCO e Quati."
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/ingest_config.yaml",
        help="Caminho para o arquivo de configuração YAML.",
    )

    return parser.parse_args()


def main():
    """
    Pipeline de ingestão e pré-processamento dos datasets.
    Datasets utilizados: MS MARCO e Quati.
    """
    # LOAD CONFIGURATION
    args = create_args()
    with open(args.config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    # Get number of CPU cores for parallel processing
    if "number_of_processes" not in config:
        CPU_COUNT = 1
    else:
        from multiprocessing import cpu_count

        CPU_COUNT = min(config["number_of_processes"], cpu_count())
    logger.info(f"Using {CPU_COUNT} CPU cores for processing.")

    # ----------------------------------------------------------------
    # LOAD AND PROCESS MS MARCO DATASET
    try:
        msmarco_config = config["ms_marco"]
        logger.info(
            f"Loading and preprocessing MS MARCO dataset, version {msmarco_config['version']}, split {msmarco_config['split']}."
        )
        msmarco_ds = load_msmarco(
            dataset_name=msmarco_config["dataset_name"],
            version=msmarco_config["version"],
            split=msmarco_config["split"],
            num_proc=CPU_COUNT,
        )
        logger.info(
            f"MS MARCO dataset loaded and preprocessed. Saving to {msmarco_config['output_path']}."
        )
        logger.debug(f"Length of MS MARCO dataset: {len(msmarco_ds)}")
        logger.debug(f"MS MARCO dataset sample: {msmarco_ds[0]}")
        msmarco_ds.to_parquet(msmarco_config["output_path"])
    except Exception as e:
        logger.error("Error processing MS MARCO dataset.")
        logger.error(str(e))
        sys.exit(1)

    # ----------------------------------------------------------------
    # LOAD AND PROCESS QUATI DATASET
    try:
        quati_config = config["quati"]
        logger.info(
            f"Loading and preprocessing Quati dataset, version {quati_config['version']}."
        )
        quati_ds = load_quati(
            dataset_name=quati_config["dataset_name"],
            version=quati_config["version"],
            num_proc=CPU_COUNT,
        )
        logger.info(
            f"Quati dataset loaded and preprocessed. Saving to {quati_config['output_path']}."
        )
        logger.debug(f"Length of Quati dataset: {len(quati_ds)}")
        logger.debug(f"Quati dataset sample: {quati_ds[0]}")
        quati_ds.to_parquet(quati_config["output_path"])
    except Exception as e:
        logger.error("Error processing Quati dataset.")
        logger.error(str(e))
        sys.exit(1)

    logger.info("Ingestion and preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    main()
