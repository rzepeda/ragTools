-- Initialize RAG Factory database
-- This script runs automatically when the PostgreSQL container starts

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema for RAG data
CREATE SCHEMA IF NOT EXISTS rag;

-- Create documents table
CREATE TABLE IF NOT EXISTS rag.documents (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) UNIQUE NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create chunks table
CREATE TABLE IF NOT EXISTS rag.chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) UNIQUE NOT NULL,
    document_id VARCHAR(255) REFERENCES rag.documents(document_id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    embedding vector(1536),  -- Adjust dimension based on your embedding model
    metadata JSONB DEFAULT '{}',
    token_count INTEGER,
    chunk_index INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON rag.chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON rag.chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON rag.documents USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON rag.chunks USING gin(metadata);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION rag.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for documents table
DROP TRIGGER IF EXISTS update_documents_updated_at ON rag.documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON rag.documents
    FOR EACH ROW
    EXECUTE FUNCTION rag.update_updated_at_column();

-- Insert sample data
INSERT INTO rag.documents (document_id, title, content, metadata) VALUES
    ('sample_001', 'Introduction to RAG', 'Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by retrieving relevant information from a knowledge base.', '{"category": "introduction", "tags": ["RAG", "AI"]}'),
    ('sample_002', 'Vector Databases', 'Vector databases store embeddings and enable efficient similarity search using techniques like approximate nearest neighbor search.', '{"category": "infrastructure", "tags": ["database", "vectors"]}'),
    ('sample_003', 'Chunking Strategies', 'Effective chunking is crucial for RAG systems. Different strategies include fixed-size, semantic, and structural chunking.', '{"category": "techniques", "tags": ["chunking", "preprocessing"]}')
ON CONFLICT (document_id) DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA rag TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA rag TO PUBLIC;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA rag TO PUBLIC;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'RAG Factory database initialized successfully';
END $$;
