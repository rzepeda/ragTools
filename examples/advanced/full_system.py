"""
Advanced RAG system example with multiple strategies and production patterns.

This example demonstrates:
- Multiple chunking and retrieval strategies
- Comprehensive logging and metrics
- Error handling and retry logic
- Configuration management
- Performance optimization
- Production-ready patterns
"""

import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import yaml

from rag_factory import RAGFactory, RAGLogger, MetricsCollector
from rag_factory.strategies.chunking import (
    StructuralChunker,
    FixedSizeChunker,
    SemanticChunker,
    ChunkingConfig,
    ChunkingMethod,
    Chunk
)


@dataclass
class SystemMetrics:
    """Track system-wide metrics."""
    total_documents_processed: int = 0
    total_chunks_created: int = 0
    total_processing_time: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    strategy_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


class AdvancedRAGSystem:
    """Advanced RAG system with multiple strategies and production features."""
    
    def __init__(self, config_path: str):
        """
        Initialize the advanced RAG system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.metrics = SystemMetrics()
        self.metrics_collector = MetricsCollector()
        self.chunkers = {}
        
        self.logger.info("Initializing Advanced RAG System")
        self._initialize_strategies()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging with appropriate handlers and formatters."""
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO"))
        
        # Create logger
        logger = logging.getLogger("AdvancedRAG")
        logger.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        # File handler if configured
        if log_file := log_config.get("file"):
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_strategies(self):
        """Initialize all configured chunking strategies."""
        strategies_config = self.config.get("strategies", {})
        
        for strategy_name, strategy_config in strategies_config.items():
            try:
                self.logger.info(f"Initializing strategy: {strategy_name}")
                chunker = self._create_chunker(strategy_name, strategy_config)
                self.chunkers[strategy_name] = chunker
                self.logger.info(f"✓ Strategy {strategy_name} initialized")
            except Exception as e:
                self.logger.error(f"✗ Failed to initialize {strategy_name}: {e}")
                if strategy_config.get("required", False):
                    raise
    
    def _create_chunker(self, name: str, config: Dict[str, Any]):
        """Create a chunker based on configuration."""
        method = ChunkingMethod(config.get("method", "fixed_size"))
        
        chunking_config = ChunkingConfig(
            method=method,
            target_chunk_size=config.get("target_chunk_size", 256),
            chunk_overlap=config.get("chunk_overlap", 20),
            min_chunk_size=config.get("min_chunk_size", 128),
            max_chunk_size=config.get("max_chunk_size", 1024),
            respect_headers=config.get("respect_headers", True),
            respect_paragraphs=config.get("respect_paragraphs", True)
        )
        
        if method == ChunkingMethod.STRUCTURAL:
            return StructuralChunker(chunking_config)
        elif method == ChunkingMethod.FIXED_SIZE:
            return FixedSizeChunker(chunking_config)
        elif method == ChunkingMethod.SEMANTIC:
            return SemanticChunker(chunking_config, embedding_service=None)
        else:
            raise ValueError(f"Unknown chunking method: {method}")
    
    def process_document(
        self,
        document: str,
        doc_id: str,
        strategy_name: Optional[str] = None,
        retry_count: int = 3
    ) -> Optional[List[Chunk]]:
        """
        Process a document with retry logic and error handling.
        
        Args:
            document: Document text to process
            doc_id: Document identifier
            strategy_name: Specific strategy to use (None for default)
            retry_count: Number of retries on failure
            
        Returns:
            List of chunks or None on failure
        """
        if strategy_name is None:
            strategy_name = self.config.get("default_strategy", "structural")
        
        if strategy_name not in self.chunkers:
            self.logger.error(f"Strategy not found: {strategy_name}")
            return None
        
        chunker = self.chunkers[strategy_name]
        
        for attempt in range(retry_count):
            try:
                self.logger.info(
                    f"Processing document {doc_id} with {strategy_name} "
                    f"(attempt {attempt + 1}/{retry_count})"
                )
                
                start_time = time.time()
                chunks = chunker.chunk_document(document, doc_id)
                elapsed = time.time() - start_time
                
                # Update metrics
                self.metrics.total_documents_processed += 1
                self.metrics.total_chunks_created += len(chunks)
                self.metrics.total_processing_time += elapsed
                
                if strategy_name not in self.metrics.strategy_metrics:
                    self.metrics.strategy_metrics[strategy_name] = {
                        "count": 0,
                        "total_time": 0.0,
                        "total_chunks": 0
                    }
                
                self.metrics.strategy_metrics[strategy_name]["count"] += 1
                self.metrics.strategy_metrics[strategy_name]["total_time"] += elapsed
                self.metrics.strategy_metrics[strategy_name]["total_chunks"] += len(chunks)
                
                self.logger.info(
                    f"✓ Created {len(chunks)} chunks in {elapsed:.3f}s"
                )
                
                return chunks
                
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {e}"
                )
                
                if attempt == retry_count - 1:
                    self.logger.error(
                        f"✗ Failed to process document {doc_id} after {retry_count} attempts"
                    )
                    self.metrics.errors.append({
                        "doc_id": doc_id,
                        "strategy": strategy_name,
                        "error": str(e),
                        "timestamp": time.time()
                    })
                    return None
                
                # Exponential backoff
                time.sleep(2 ** attempt)
        
        return None
    
    def process_batch(
        self,
        documents: List[Dict[str, str]],
        strategy_name: Optional[str] = None
    ) -> Dict[str, List[Chunk]]:
        """
        Process multiple documents in batch.
        
        Args:
            documents: List of document dicts with 'id' and 'text' keys
            strategy_name: Strategy to use for all documents
            
        Returns:
            Dictionary mapping document IDs to chunk lists
        """
        self.logger.info(f"Processing batch of {len(documents)} documents")
        
        results = {}
        start_time = time.time()
        
        for doc in documents:
            doc_id = doc.get("id", "unknown")
            text = doc.get("text", "")
            
            chunks = self.process_document(text, doc_id, strategy_name)
            if chunks:
                results[doc_id] = chunks
        
        elapsed = time.time() - start_time
        self.logger.info(
            f"Batch processing completed in {elapsed:.2f}s "
            f"({len(results)}/{len(documents)} successful)"
        )
        
        return results
    
    def get_metrics_report(self) -> str:
        """Generate a formatted metrics report."""
        report = [
            "\n" + "="*60,
            "SYSTEM METRICS REPORT",
            "="*60,
            f"\nDocuments Processed: {self.metrics.total_documents_processed}",
            f"Total Chunks Created: {self.metrics.total_chunks_created}",
            f"Total Processing Time: {self.metrics.total_processing_time:.2f}s",
            f"Errors: {len(self.metrics.errors)}",
            "\nStrategy Performance:",
            "-"*60
        ]
        
        for strategy, metrics in self.metrics.strategy_metrics.items():
            avg_time = metrics["total_time"] / metrics["count"] if metrics["count"] > 0 else 0
            avg_chunks = metrics["total_chunks"] / metrics["count"] if metrics["count"] > 0 else 0
            
            report.extend([
                f"\n{strategy.upper()}:",
                f"  Documents: {metrics['count']}",
                f"  Total chunks: {metrics['total_chunks']}",
                f"  Avg time: {avg_time:.3f}s",
                f"  Avg chunks/doc: {avg_chunks:.1f}"
            ])
        
        if self.metrics.errors:
            report.extend([
                "\nErrors:",
                "-"*60
            ])
            for error in self.metrics.errors[:5]:  # Show first 5 errors
                report.append(f"  {error['doc_id']}: {error['error']}")
        
        report.append("="*60)
        
        return "\n".join(report)


def main():
    """Run the advanced RAG system example."""
    
    # Initialize system
    config_path = Path(__file__).parent / "config" / "strategies.yaml"
    system = AdvancedRAGSystem(str(config_path))
    
    # Sample documents
    documents = [
        {
            "id": "doc_001",
            "text": """# Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Supervised Learning

In supervised learning, the algorithm learns from labeled training data to make predictions or decisions.

## Unsupervised Learning

Unsupervised learning finds hidden patterns in data without pre-existing labels."""
        },
        {
            "id": "doc_002",
            "text": """# Deep Learning

Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input.

## Neural Networks

Neural networks are computing systems inspired by biological neural networks in animal brains.

## Applications

Deep learning has revolutionized computer vision, natural language processing, and speech recognition."""
        },
        {
            "id": "doc_003",
            "text": """# Natural Language Processing

NLP enables computers to understand, interpret, and generate human language in a valuable way.

## Key Tasks

Common NLP tasks include sentiment analysis, named entity recognition, and machine translation."""
        }
    ]
    
    # Process documents
    print("\n" + "="*60)
    print("PROCESSING DOCUMENTS")
    print("="*60)
    
    results = system.process_batch(documents)
    
    # Display sample results
    print("\n" + "="*60)
    print("SAMPLE RESULTS")
    print("="*60)
    
    for doc_id, chunks in list(results.items())[:2]:  # Show first 2 documents
        print(f"\nDocument: {doc_id}")
        print(f"Chunks: {len(chunks)}")
        if chunks:
            print(f"First chunk: {chunks[0].text[:100]}...")
    
    # Display metrics
    print(system.get_metrics_report())
    
    return 0


if __name__ == "__main__":
    exit(main())
