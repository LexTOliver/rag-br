"""
Service module for document indexing in the vector index.

This module defines the IndexService class responsible for handling
the logic of vectorizing and indexing documents into a vector index.
"""

from vectorize.vector_index import VectorIndex


class IndexService:
    """
    Service class for handling document indexing in the vector index.

    Attributes:
        vector_index: The vector index instance used for indexing.

    Methods:
        index_document(document: str, metadata: dict) -> dict:
            Indexes a document in the vector index and returns the indexing result.
    """

    def __init__(self, vector_index: VectorIndex):
        """
        Initializes the IndexService with the provided vector index.

        Params:
            vector_index (VectorIndex): The vector index instance to be used for indexing.
        """
        self.vector_index = vector_index

    def index_document(self, document: str, metadata: dict) -> dict:
        """
        Indexes a document in the vector index and returns the indexing result.

        Params:
            document (str): The content of the document to be indexed.
            metadata (dict): Additional metadata associated with the document.
        Returns:
            dict: A dictionary containing the indexing result.
        """
        # TODO: Search if it is needed to preprocess text before indexing

        # Vectorize and index document
        res = self.vector_index.index_document(document, metadata)

        # Parse status and message
        status = "success" if res.status == "indexed" else "failure"
        message = f"Document {res.status}. {res.message or ''}".strip()

        return {
            "doc_id": res.doc_id,
            "status": status,
            "message": message,
            "chunks_indexed": res.chunks_indexed,
        }
