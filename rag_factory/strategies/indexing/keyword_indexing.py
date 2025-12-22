"""Keyword indexing strategy implementation.

This module implements the keyword-based indexing strategy using TF-IDF.
It extracts keywords from document chunks and builds an inverted index
for BM25-style retrieval.
"""

from typing import List, Dict, Any, Set
from sklearn.feature_extraction.text import TfidfVectorizer

from rag_factory.core.indexing_interface import (
    IIndexingStrategy,
    IndexingContext,
    IndexingResult
)
from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency
from rag_factory.factory import register_rag_strategy as register_strategy


@register_strategy("KeywordIndexer")
class KeywordIndexing(IIndexingStrategy):
    """Creates keyword index for BM25/keyword retrieval.
    
    This strategy extracts keywords from existing chunks using TF-IDF
    and builds an inverted index mapping keywords to chunks.
    
    Produces:
        - KEYWORDS: Inverted index for keyword search
        - DATABASE: Data persisted to database
        
    Requires:
        - DATABASE: Service for retrieving chunks and storing index
    """

    def produces(self) -> Set[IndexCapability]:
        """Declare produced capabilities.

        Returns:
            Set containing KEYWORDS and DATABASE capabilities
        """
        return {
            IndexCapability.KEYWORDS,
            IndexCapability.DATABASE
        }

    def requires_services(self) -> Set[ServiceDependency]:
        """Declare required services.

        Returns:
            Set containing DATABASE service dependency
        """
        return {
            ServiceDependency.DATABASE
        }

    async def process(
        self,
        documents: List[Dict[str, Any]],
        context: IndexingContext
    ) -> IndexingResult:
        """Extract keywords and build inverted index.

        Args:
            documents: List of documents to index
            context: Indexing context with database service

        Returns:
            IndexingResult with capabilities and metrics

        Raises:
            ValueError: If no chunks are found for the documents
        """
        # Get chunks from database
        # We assume chunks have already been created by a previous strategy
        # or we need to fetch them based on document IDs
        document_ids = [doc.get("id") for doc in documents if doc.get("id")]

        if not document_ids:
            # If documents don't have IDs yet, we can't link keywords to them
            # But maybe we are supposed to work on the text directly?
            # The story says "Get chunks from database", so we assume they exist.
            # If documents are new, they might not be in DB yet if we run this in parallel?
            # Usually chunking runs first.
            # For now, let's assume we can get chunks for these docs.
            pass

        chunks = await context.database.get_chunks_for_documents(document_ids)

        # If no chunks found, create them from documents
        if not chunks:
            chunk_size = self.config.get('chunk_size', 512)
            overlap = self.config.get('overlap', 50)
            
            chunks = []
            for doc in documents:
                text = doc.get('text', '')
                if not text:
                    continue
                    
                doc_id = doc.get('id', 'unknown')
                doc_chunks = self._chunk_document(text, doc_id, chunk_size, overlap)
                chunks.extend(doc_chunks)
            
            # Store chunks to database
            if chunks:
                await context.database.store_chunks(chunks)

        # Extract keywords using TF-IDF
        max_keywords = self.config.get('max_keywords', 1000)
        ngram_range = self.config.get('ngram_range', (1, 2))

        vectorizer = TfidfVectorizer(
            max_features=max_keywords,
            stop_words='english',
            ngram_range=ngram_range
        )

        texts = [c['text'] for c in chunks]

        # Handle empty texts or too few documents for TF-IDF
        if not texts or all(not t.strip() for t in texts):
             return IndexingResult(
                capabilities=self.produces(),
                metadata={"warning": "No text content to index"},
                document_count=len(documents),
                chunk_count=len(chunks)
            )

        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            # Can happen if vocabulary is empty (e.g. all stop words)
            return IndexingResult(
                capabilities=self.produces(),
                metadata={"warning": "Empty vocabulary after stop word removal"},
                document_count=len(documents),
                chunk_count=len(chunks)
            )

        # Build inverted index
        feature_names = vectorizer.get_feature_names_out()
        inverted_index: Dict[str, List[Dict[str, Any]]] = {}

        # Iterate over chunks (rows in matrix)
        # tfidf_matrix is a sparse matrix
        for chunk_idx, chunk in enumerate(chunks):
            # Get keywords for this chunk
            chunk_vector = tfidf_matrix[chunk_idx]

            # Get non-zero indices
            indices = chunk_vector.nonzero()[1]

            for idx in indices:
                keyword = feature_names[idx]
                score = float(chunk_vector[0, idx])

                if keyword not in inverted_index:
                    inverted_index[keyword] = []

                inverted_index[keyword].append({
                    'chunk_id': chunk.get('chunk_id', chunk.get('id')),
                    'score': score
                })

        # Store inverted index
        await context.database.store_keyword_index(inverted_index)

        return IndexingResult(
            capabilities=self.produces(),
            metadata={
                'total_keywords': len(inverted_index),
                'avg_keywords_per_chunk': len(feature_names) / len(chunks) if chunks else 0,
                'method': 'tfidf',
                'max_keywords': max_keywords,
                'ngram_range': ngram_range
            },
            document_count=len(documents),
            chunk_count=len(chunks)
        )
    
    def _chunk_document(
        self,
        text: str,
        doc_id: str,
        chunk_size: int,
        overlap: int
    ) -> List[Dict[str, Any]]:
        """Simple character-based chunking."""
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunks.append({
                'chunk_id': f"{doc_id}_{chunk_idx}",
                'text': chunk_text,
                'document_id': doc_id,
                'chunk_index': chunk_idx,
                'metadata': {}
            })
            
            if end == len(text):
                break
                
            start += chunk_size - overlap
            chunk_idx += 1
        
        return chunks
