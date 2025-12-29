"""
Chunking module for data indexing and retrieval tasks.

This module provides a class to perform chunking of text data
based on tokenization, with configurable overlap between chunks.
"""

from typing import List

from transformers import AutoTokenizer

from vectorize.config import ChunkerConfig, ModelConfig


class Chunker:
    """
    Classe para aplicação de chunking em textos.
    Realiza chunking de textos longos baseado em tokens,
    com overlap configurável.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizador pré-treinado para segmentação de texto.
        chunk_size (int): Tamanho máximo de cada chunk em tokens.
        overlap (int): Número de tokens que se sobrepõem entre chunks consecutivos.

    Methods:
        chunk(text: str) -> List[str]:
            Divide um texto em trechos (chunks) com base em tokens,
            mantendo um overlap entre os trechos.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        config: ChunkerConfig,
    ):
        """
        Inicializa o Chunker com um modelo de tokenização específico,
        tamanho de chunk e overlap.

        Params:
            model_config (ModelConfig): Configurações do modelo pré-treinado para tokenização.
            config (ChunkerConfig): Configurações de chunking (tamanho do chunk e overlap).
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        self.chunk_size = config.chunk_size
        self.overlap = config.overlap

    def chunk(self, text: str) -> List[str]:
        """
        Divide um texto em trechos (chunks) a partir de tokens, mantendo um overlap entre os trechos.
        Cada trecho é representado por uma lista de tokens contendo até max_tokens palavras.
        Os chunks são extraídos diretamente do texto original.

        Params:
            text (str): Texto a ser dividido em trechos.

        Returns:
            List[str]: Lista de trechos do texto.
        """
        # Return empty list if text is empty or whitespace
        if not text or not text.strip():
            return []

        # Create tokens with offsets
        tokens = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        # Return empty list if no tokens
        if not tokens["input_ids"]:
            return []

        # Extract input IDs and offset mappings
        input_ids = tokens["input_ids"]
        offsets = tokens["offset_mapping"]

        # Set chunks
        chunks = []
        start = 0
        while start < len(input_ids):
            end = start + self.chunk_size
            chunk_offsets = offsets[start:end]
            char_start = chunk_offsets[0][0]
            char_end = chunk_offsets[-1][1]
            chunk = text[char_start:char_end]
            if chunk and chunk.strip():  # filtra chunks vazios/brancos
                chunks.append(chunk)
            start += self.chunk_size - self.overlap

        return chunks

    def chunk_decode(self, text: str) -> List[str]:
        """
        Divide um texto em trechos (chunks) a partir de tokens, mantendo um overlap entre os trechos.
        Cada trecho é representado por uma lista de tokens contendo até max_tokens palavras.
        Os chunks são decodificados de volta para strings.

        Params:
            text (str): Texto a ser dividido em trechos.

        Returns:
            List[str]: Lista de trechos do texto.
        """
        if not text or not text.strip():
            return []

        # Create tokens
        tokens = self.tokenizer(
            text,
            truncation=False,
            return_offsets_mapping=False,
            add_special_tokens=False,
        )["input_ids"]

        # Return empty list if no tokens
        if not tokens:
            return []

        # Set chunks
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            # Set chunk ids
            chunk_ids = tokens[i : i + self.chunk_size]
            if not chunk_ids:
                continue

            # Get chunks
            chunk = self.tokenizer.decode(
                chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            if chunk and chunk.strip():  # filtra chunks vazios/brancos
                chunks.append(chunk)
        return chunks
