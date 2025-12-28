"""
Dataset ingestion and loading utilities for RAG-BR.

This module provides functions to load and preprocess the MS MARCO and Quati datasets
from Hugging Face, applying necessary formatting and preprocessing steps.
"""

from typing import Dict

from datasets import Dataset, load_dataset


def load_msmarco(
    dataset_name: str = "microsoft/ms_marco",
    version: str = "v2.1",
    split: str = "train",
) -> Dataset:
    """
    Carrega o dataset MS MARCO da Hugging Face.

    Params:
        dataset_name (str): Nome do dataset na Hugging Face.
        version (str): Versão do MS MARCO ("v1" ou "v2.1").
        split (str): Divisão do dataset ("train", "validation", "test").

    Returns:
        Dataset: Objeto Dataset carregado.
    """
    # Load MS MARCO dataset from Hugging Face
    ds = load_dataset(dataset_name, version, split=split)

    return ds


def load_quati(
    dataset_name: str = "unicamp-dl/quati", version: str = "1M"
) -> Dict[str, Dataset]:
    """
    Carrega o dataset Quati da Hugging Face.

    Params:
        dataset_name (str): Nome do dataset na Hugging Face.
        version (str): Versão do Quati ("1M" para 1 milhão de trechos).

    Returns:
        Dict[str, Dataset]: Dicionário contendo os objetos Dataset carregados.
    """
    # Load Quati dataset passages
    passages_ds = load_dataset(
        dataset_name, f"quati_{version}_passages", trust_remote_code=True
    )[f"quati_{version}_passages"]

    # Load Quati dataset queries
    queries_ds = load_dataset(dataset_name, "quati_all_topics", trust_remote_code=True)[
        "quati_all_topics"
    ]

    # Load Quati dataset qrels
    qrels_ds = load_dataset(
        dataset_name, f"quati_{version}_qrels", trust_remote_code=True
    )[f"quati_{version}_qrels"]

    return {"passages": passages_ds, "queries": queries_ds, "qrels": qrels_ds}


def load_parquet_dataset(file_path: str) -> Dataset:
    """
    Carrega um dataset a partir de um arquivo Parquet.

    Params:
        file_path (str): Caminho para o arquivo Parquet.

    Returns:
        Dataset: Objeto Dataset carregado.
    """
    ds = load_dataset("parquet", data_files=file_path)["train"]
    return ds
