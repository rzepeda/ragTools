# Add these methods to DatabaseContext class

async def store_keyword_index(self, inverted_index: Dict[str, List[Dict[str, Any]]]) -> None:
    """Store keyword inverted index to database."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, self._store_keyword_index_sync, inverted_index)

def _store_keyword_index_sync(self, inverted_index: Dict[str, List[Dict[str, Any]]]) -> None:
    """Synchronous implementation of store_keyword_index."""
    if not inverted_index:
        return
    
    if "inverted_index" not in self.tables:
        logger.warning("No 'inverted_index' table mapping found")
        return
    
    table = self.get_table("inverted_index")
    
    try:
        with self.engine.begin() as conn:
            conn.execute(delete(table))
            for keyword, chunk_list in inverted_index.items():
                for entry in chunk_list:
                    data = {
                        self._map_field("term"): keyword,
                        self._map_field("chunk_id"): entry['chunk_id'],
                        self._map_field("score"): entry.get('score', 1.0)
                    }
                    conn.execute(insert(table).values(**data))
            logger.info(f"Stored keyword index with {len(inverted_index)} terms")
    except Exception as e:
        logger.error(f"Failed to store keyword index: {e}")
        raise

async def search_keyword(self, query_terms: List[str], top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
    """Search using keyword index."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self._search_keyword_sync, query_terms, top_k)

def _search_keyword_sync(self, query_terms: List[str], top_k: int) -> List[Dict[str, Any]]:
    """Synchronous keyword search."""
    try:
        index_table = self.get_table('inverted_index')
        chunks_table = self.get_table('chunks')
        
        with self.engine.connect() as conn:
            matching_chunks = {}
            for term in query_terms:
                query = select(index_table).where(index_table.c.term == term)
                for row in conn.execute(query).fetchall():
                    chunk_id = row.chunk_id
                    matching_chunks[chunk_id] = matching_chunks.get(chunk_id, 0) + 1
            
            if not matching_chunks:
                return []
            
            results = []
            for chunk_id, term_count in matching_chunks.items():
                chunk_id_field = self._map_field('chunk_id')
                text_field = self._map_field('text')
                doc_id_field = self._map_field('document_id')
                
                query = select(chunks_table).where(chunks_table.c[chunk_id_field] == chunk_id)
                chunk_row = conn.execute(query).fetchone()
                
                if chunk_row:
                    results.append({
                        'chunk_id': chunk_id,
                        'text': getattr(chunk_row, text_field, ''),
                        'document_id': getattr(chunk_row, doc_id_field, ''),
                        'score': term_count / len(query_terms),
                        'metadata': {}
                    })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
    except Exception as e:
        logger.error(f"Keyword search failed: {e}")
        return []
