"""
Text preprocessing utilities for data ingestion.

This module contains functions for:
- Text normalization and cleaning
- Label normalization
- Dataset-level preprocessing operations

These functions are designed to be framework-agnostic and
used both in notebooks and production pipelines.
"""

import re
import unicodedata
from multiprocessing import cpu_count

from datasets import Dataset

HTML_TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"\S+@\S+")
MULTISPACE = re.compile(r"\s+")
CPU_COUNT = 1 if cpu_count() is None else cpu_count()  # TODO: Adjust


def clean_text(text: str) -> str:
    """
    Realiza limpeza textual baseada em expressões regulares.
    As transformações aplicadas são:
    - Normalização Unicode
    - Remoção de ruídos (como sequência de "===" ou " ---")
    - Remoção de HTML, URLs e E-mails
    - Remoção de caracteres não-principais e de controle
    - Normalização de espaços

    Params:
        text (str): Texto a ser limpo.

    Returns:
        str: Texto limpo.
    """
    if not text:
        return ""

    # Unicode normalization (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # Remove noise (repeated sequences such as "=====" or "----")
    text = re.sub(r"([^a-zA-Z0-9\s])\1{3,}", " ", text)

    # Remove HTML, URLs, E-mails
    text = HTML_TAG.sub("", text)
    text = URL.sub("", text)
    text = EMAIL.sub("", text)

    # Clean non-printable characters and control
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

    # Spaces normalization (useful for keeping cleaned chunks in RAG)
    text = MULTISPACE.sub(" ", text).strip()

    return text.lower()


def normalize_label(label: float, divisor: float = 1.0) -> float:
    """
    Normaliza o rótulo dividindo pelo divisor fornecido.

    Params:
        label (float): Rótulo original.
        divisor (float): Valor para dividir o rótulo.

    Returns:
        float: Rótulo normalizado.
    """
    if divisor == 0:
        raise ValueError("Divisor não pode ser zero.")
    return label / divisor


def preprocess_dataset(
    dataset: Dataset, label_divisor: float = 1.0, num_proc: int = CPU_COUNT
) -> Dataset:
    """
    Aplica os passos para pré-processamento dos dados.
    Passos aplicados:
    - Limpeza textual para os campos 'query' e 'passage'
    - Normalização dos rótulos dividindo pelo divisor fornecido

    Params:
        dataset (Dataset): Dataset a ser processado.
        label_divisor (float): Divisor para normalização dos rótulos.
        num_proc (int): Número de processos para execução paralela.

    Returns:
        Dataset: Dataset com textos limpos.
    """

    # Function to clean text in batches
    def _clean_in_batch(batch: dict) -> dict:
        batch["query"] = [clean_text(text) for text in batch["query"]]
        batch["passage"] = [clean_text(text) for text in batch["passage"]]
        return batch

    # Function to normalize labels in batches
    def _normalize_labels_in_batch(batch: dict) -> dict:
        batch["label"] = [
            normalize_label(label, divisor=label_divisor) for label in batch["label"]
        ]
        return batch

    # Apply text cleaning in parallel
    dataset_cleaned = dataset.map(
        _clean_in_batch, batched=True, batch_size=10_000, num_proc=num_proc
    )

    # Apply label normalization in parallel
    dataset_cleaned = dataset_cleaned.map(
        _normalize_labels_in_batch, batched=True, batch_size=10_000, num_proc=num_proc
    )

    return dataset_cleaned


# MS MARCO-specific preprocessing
# ----------------------------------------------------------


def format_msmarco(msmarco_ds: Dataset, num_proc: int = CPU_COUNT) -> Dataset:
    """
    Formata o dataset MS MARCO para o esquema padrão:
    - query_id: ID da consulta
    - passage_id: ID do trecho
    - query: Texto da consulta
    - passage: Texto do trecho
    - label: Rótulo binário (1 se relevante, 0 caso contrário)

    Params:
        msmarco_ds (Dataset): Dataset MS MARCO bruto.
    Returns:
        Dataset: Dataset formatado.
    """

    # Function to format the dataset in batches
    def _format_in_batch(batch: dict) -> dict:
        queries = []
        queries_ids = []
        passages = []
        passages_ids = []
        scores = []

        # Iterate query and the dictionary containing passages and scores for that query
        for query_id, query_text, passage_info_dict in zip(
            batch["query_id"], batch["query"], batch["passages"]
        ):
            # Iterate over the individual passages and their corresponding scores
            for idx, p_text, is_selected_score in zip(
                range(
                    len(passage_info_dict["passage_text"])
                ),  # This is a list of indexes
                passage_info_dict["passage_text"],  # This is a list of passage strings
                passage_info_dict["is_selected"],  # This is a list of scores (0 or 1)
            ):
                queries_ids.append(query_id)
                passages_ids.append(str(query_id) + "_" + str(idx))
                queries.append(query_text)
                passages.append(p_text)
                scores.append(float(is_selected_score))

        return {
            "query_id": queries_ids,
            "passage_id": passages_ids,
            "query": queries,
            "passage": passages,
            "label": scores,
        }

    # Apply formatting in parallel
    dataset_formatted = msmarco_ds.map(
        _format_in_batch,
        batched=True,
        batch_size=10_000,
        num_proc=num_proc,
        remove_columns=msmarco_ds.column_names,
    )

    return dataset_formatted


# Quati-specific preprocessing
# ----------------------------------------------------------


def format_quati(
    passages_ds: Dataset,
    queries_ds: Dataset,
    qrels_ds: Dataset,
    num_proc: int = CPU_COUNT,
) -> Dataset:
    """
    Formata o dataset Quati para o esquema padrão:
    - query_id: ID da consulta
    - passage_id: ID do trecho
    - query: Texto da consulta
    - passage: Texto do trecho
    - label: Rótulo binário (1 se relevante, 0 caso contrário)

    Params:
        passages_ds (Dataset): Dataset de trechos.
        queries_ds (Dataset): Dataset de consultas.
        qrels_ds (Dataset): Dataset de relevâncias.

    Returns:
        Dataset: Dataset formatado.
    """
    # Create dictionary lookup to get query by id
    topics_lookup = {}
    for q in queries_ds:
        topics_lookup[q["query_id"]] = q["query"]

    # Create dictionary lookup to get passage by id
    passages_lookup = {}
    for p in passages_ds:
        passages_lookup[p["passage_id"]] = p["passage"]

    # Function to format the dataset in batches
    def _format_in_batch(batch):
        # Get info
        queries = [topics_lookup[query_id] for query_id in batch["query_id"]]
        passages = [passages_lookup[passage_id] for passage_id in batch["passage_id"]]
        scores = [score for score in batch["score"]]
        query_ids = [query_id for query_id in batch["query_id"]]
        passage_ids = [passage_id for passage_id in batch["passage_id"]]

        return {
            "query_id": query_ids,
            "passage_id": passage_ids,
            "query": queries,
            "passage": passages,
            "label": scores,
        }

    # Apply formatting in parallel
    dataset_formatted = qrels_ds.map(
        _format_in_batch,
        batched=True,
        batch_size=10_000,
        num_proc=num_proc,
        remove_columns=qrels_ds.column_names,
    )

    return dataset_formatted
