# Story 13.3: Implement Keyword Extraction Indexing

**Story ID:** 13.3
**Epic:** Epic 13 - Core Indexing Strategies Implementation
**Story Points:** 8
**Priority:** Medium
**Dependencies:** Epic 13.1 (Chunking)

---

## User Story

**As a** system
**I want** keyword-based indexing without embeddings
**So that** I can support non-ML retrieval approaches

---

## Detailed Requirements

### Functional Requirements

1.  **Keyword Indexing Strategy**
    *   Implement `KeywordIndexing` class implementing `IIndexingStrategy`.
    *   Extract keywords from chunks using TF-IDF or similar statistical methods.
    *   Build an inverted index mapping keywords to chunks.
    *   Produce `KEYWORDS` and `DATABASE` capabilities.
    *   Operate without requiring an embedding service or LLM.

2.  **Keyword Extraction**
    *   Use `scikit-learn`'s `TfidfVectorizer` (or similar) for extraction.
    *   Support configurable parameters:
        *   `max_keywords`: Maximum number of keywords to track.
        *   `ngram_range`: Support for unigrams, bigrams, etc.
        *   `stop_words`: Language-specific stop word removal.

3.  **Inverted Index Construction**
    *   Map extracted keywords to chunk IDs with relevance scores.
    *   Store the inverted index in the database.

### Non-Functional Requirements

1.  **Performance**
    *   Target: <500ms per 10k words.
    *   Efficient sparse matrix operations.

2.  **Independence**
    *   Must function completely independently of GPU/ML model services.

---

## Acceptance Criteria

### AC1: Strategy Implementation
- [ ] `KeywordIndexing` class exists and implements `IIndexingStrategy`.
- [ ] `produces()` returns `{IndexCapability.KEYWORDS, IndexCapability.DATABASE}`.
- [ ] `requires_services()` returns `{ServiceDependency.DATABASE}` (no embedding service).

### AC2: Keyword Extraction
- [ ] TF-IDF extraction works correctly on text chunks.
- [ ] Stop words are filtered out.
- [ ] N-grams are supported as configured.

### AC3: Inverted Index
- [ ] Inverted index is correctly built (keyword -> [chunk_id, score]).
- [ ] Index is stored in the database.

### AC4: Testing
- [ ] Unit tests for keyword extraction logic.
- [ ] Integration tests verifying database storage.
- [ ] Performance benchmarks meet targets.

---

## Technical Specifications

### Implementation

```python
from rag_factory.core.indexing import IIndexingStrategy, IndexingContext, IndexingResult
from rag_factory.core.capabilities import IndexCapability
from rag_factory.core.dependencies import ServiceDependency
from sklearn.feature_extraction.text import TfidfVectorizer

class KeywordIndexing(IIndexingStrategy):
    """Creates keyword index for BM25/keyword retrieval"""
    
    def produces(self) -> set[IndexCapability]:
        return {
            IndexCapability.KEYWORDS,
            IndexCapability.DATABASE
        }
    
    def requires_services(self) -> set[ServiceDependency]:
        return {
            ServiceDependency.DATABASE  # No embedding or LLM needed!
        }
    
    async def process(
        self,
        documents: list['Document'],
        context: IndexingContext
    ) -> IndexingResult:
        """
        Extract keywords and build inverted index.
        
        Strategy:
        1. Get chunks from database
        2. Extract keywords using TF-IDF
        3. Build inverted index
        4. Store index in database
        """
        # Get chunks
        chunks = await context.database.get_chunks_for_documents(
            [doc.id for doc in documents]
        )
        
        if not chunks:
            raise ValueError("No chunks found. Run chunking strategy first.")
        
        # Extract keywords using TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=self.config.get('max_keywords', 1000),
            stop_words='english',
            ngram_range=(1, 2)  # unigrams and bigrams
        )
        
        texts = [c['text'] for c in chunks]
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Build inverted index
        feature_names = vectorizer.get_feature_names_out()
        inverted_index = {}
        
        for chunk_idx, chunk in enumerate(chunks):
            # Get keywords for this chunk
            chunk_vector = tfidf_matrix[chunk_idx]
            keywords = [
                (feature_names[i], chunk_vector[0, i])
                for i in chunk_vector.nonzero()[1]
            ]
            
            # Add to inverted index
            for keyword, score in keywords:
                if keyword not in inverted_index:
                    inverted_index[keyword] = []
                
                inverted_index[keyword].append({
                    'chunk_id': chunk['id'],
                    'score': float(score)
                })
        
        # Store inverted index
        await context.database.store_keyword_index(inverted_index)
        
        return IndexingResult(
            capabilities=self.produces(),
            metadata={
                'total_keywords': len(inverted_index),
                'avg_keywords_per_chunk': len(feature_names) / len(chunks),
                'method': 'tfidf'
            },
            document_count=len(documents),
            chunk_count=len(chunks)
        )
```

### Technical Dependencies
- scikit-learn (for TF-IDF)
- Database service
