# Story 5.1: Implement Agentic RAG Strategy

**Story ID:** 5.1
**Epic:** Epic 5 - Agentic & Advanced Retrieval Strategies
**Story Points:** 13
**Priority:** High
**Dependencies:** Epic 4 (basic strategies), Epic 3 (LLM service)

---

## User Story

**As a** system
**I want** agents to choose how to search the knowledge base
**So that** retrieval is flexible based on query type

---

## Detailed Requirements

### Functional Requirements

1. **Tool Definition System**
   - Define semantic search tool (searches embeddings for relevant chunks)
   - Define document reader tool (reads full documents by ID or title)
   - Define metadata search tool (filters by document metadata)
   - Define hybrid search tool (combines semantic + keyword search)
   - Tool descriptions that LLM can understand
   - Tool parameter schemas (required/optional parameters)
   - Tool result formatting

2. **Tool Selection Logic**
   - LLM-based tool selection (agent decides which tools to use)
   - Rule-based fallback for simple queries
   - Support for sequential tool calls (use results from one tool to inform next)
   - Support for parallel tool calls (run multiple searches simultaneously)
   - Tool selection reasoning/explanation logging
   - Confidence scoring for tool selections

3. **Agent Framework Integration**
   - Integration with LangGraph for agent workflows
   - Integration with Anthropic tool use (Claude function calling)
   - Custom simple agent implementation (no external framework required)
   - Agent state management (track conversation, tool results)
   - Agent planning phase (decide strategy before executing)
   - Agent reflection phase (evaluate results quality)

4. **Multi-Step Retrieval**
   - Support for iterative search refinement
   - Use initial results to inform follow-up queries
   - Query decomposition (break complex queries into sub-queries)
   - Result synthesis across multiple tool calls
   - Maximum iteration limits to prevent infinite loops
   - Early stopping when sufficient results found

5. **Tool Result Management**
   - Store intermediate tool results
   - Merge results from multiple tools
   - Deduplicate chunks across tool results
   - Rank and reorder results based on relevance
   - Filter low-quality results
   - Format results for final presentation

6. **Query Analysis**
   - Classify query type (factual, exploratory, specific document, metadata-based)
   - Extract entities and keywords from queries
   - Identify required information types
   - Determine query complexity (simple vs multi-hop)
   - Suggest appropriate tools based on query analysis

### Non-Functional Requirements

1. **Performance**
   - Tool selection decision <500ms
   - Support up to 5 tool calls per query
   - Total retrieval time <5 seconds for complex queries
   - Parallel tool execution where possible

2. **Reliability**
   - Handle tool failures gracefully (skip failed tools)
   - Fallback to semantic search if agent fails
   - Timeout protection (max 30 seconds per query)
   - Prevent infinite loops in multi-step retrieval

3. **Observability**
   - Log all tool selections with reasoning
   - Track tool execution times
   - Monitor tool success/failure rates
   - Provide explanation of retrieval strategy used

4. **Maintainability**
   - Clear tool interface for adding new tools
   - Pluggable agent frameworks
   - Well-documented tool descriptions
   - Configuration-driven tool availability

5. **Cost Efficiency**
   - Minimize LLM calls for tool selection
   - Cache tool selection decisions for similar queries
   - Use cheaper models for tool selection when possible

---

## Acceptance Criteria

### AC1: Tool System
- [ ] Semantic search tool implemented with proper schema
- [ ] Document reader tool implemented
- [ ] Metadata search tool implemented
- [ ] Hybrid search tool implemented
- [ ] Tool descriptions are clear and LLM-understandable
- [ ] Tool parameter validation working
- [ ] Tool result formatting consistent

### AC2: Agent Selection
- [ ] LLM-based agent can select appropriate tools
- [ ] Agent provides reasoning for tool selection
- [ ] Rule-based fallback works for simple queries
- [ ] Agent handles tool selection errors gracefully
- [ ] Agent can select multiple tools for one query
- [ ] Tool selection decision logged

### AC3: Framework Integration
- [ ] Anthropic tool use integration working
- [ ] Custom agent implementation working (no dependencies)
- [ ] LangGraph integration optional and pluggable
- [ ] Agent state persisted across tool calls
- [ ] Agent can plan multi-step strategies

### AC4: Multi-Step Retrieval
- [ ] Agent can make sequential tool calls
- [ ] Agent uses previous results to inform next queries
- [ ] Query decomposition working for complex queries
- [ ] Results merged and deduplicated correctly
- [ ] Maximum iterations enforced (default: 3)
- [ ] Early stopping when results sufficient

### AC5: Query Analysis
- [ ] Query type classification working
- [ ] Entity extraction from queries
- [ ] Tool recommendation based on query analysis
- [ ] Query complexity assessment
- [ ] Analysis results guide tool selection

### AC6: Observability
- [ ] All tool calls logged with timestamps
- [ ] Tool selection reasoning captured
- [ ] Execution metrics tracked (time, tokens, cost)
- [ ] Agent decision trace available for debugging
- [ ] Performance metrics per tool

### AC7: Testing
- [ ] Unit tests for all tools with mocked data
- [ ] Unit tests for agent logic
- [ ] Integration tests with real LLM
- [ ] Integration tests with database
- [ ] Performance benchmarks for complex queries

---

## Technical Specifications

### File Structure
```
rag_factory/
├── strategies/
│   ├── agentic/
│   │   ├── __init__.py
│   │   ├── strategy.py          # Main agentic strategy
│   │   ├── agent.py              # Agent implementation
│   │   ├── tools.py              # Tool definitions
│   │   ├── tool_selector.py     # Tool selection logic
│   │   ├── query_analyzer.py    # Query analysis
│   │   ├── frameworks/
│   │   │   ├── __init__.py
│   │   │   ├── anthropic.py     # Anthropic tool use
│   │   │   ├── langgraph.py     # LangGraph integration (optional)
│   │   │   └── simple.py        # Simple custom agent
│   │   └── config.py            # Agentic strategy config
│
tests/
├── unit/
│   └── strategies/
│       └── agentic/
│           ├── test_strategy.py
│           ├── test_agent.py
│           ├── test_tools.py
│           ├── test_tool_selector.py
│           └── test_query_analyzer.py
│
├── integration/
│   └── strategies/
│       └── test_agentic_integration.py
```

### Dependencies
```python
# requirements.txt additions
anthropic>=0.18.1          # For Claude tool use
langgraph>=0.0.20          # Optional agent framework
langchain-core>=0.1.0      # Optional for LangGraph
pydantic>=2.0.0            # Tool schema validation
```

### Tool Definition
```python
# rag_factory/strategies/agentic/tools.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

class ToolParameter(BaseModel):
    """Tool parameter definition."""
    name: str
    type: str  # "string", "integer", "array", etc.
    description: str
    required: bool = True
    default: Any = None

class ToolResult(BaseModel):
    """Result from tool execution."""
    tool_name: str
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float
    metadata: Dict[str, Any] = {}

class Tool(ABC):
    """Base class for agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> List[ToolParameter]:
        """Tool parameters."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description
                    }
                    for param in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required]
            }
        }


class SemanticSearchTool(Tool):
    """Tool for semantic similarity search."""

    def __init__(self, retrieval_service):
        self.retrieval_service = retrieval_service

    @property
    def name(self) -> str:
        return "semantic_search"

    @property
    def description(self) -> str:
        return """Search for relevant text chunks using semantic similarity.
        Use this when you need to find information based on meaning and context,
        not just exact keyword matches. Good for answering questions like
        'What is...' or 'How does... work?'"""

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="The search query to find relevant chunks",
                required=True
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="Number of results to return (default: 5)",
                required=False,
                default=5
            ),
            ToolParameter(
                name="min_score",
                type="number",
                description="Minimum similarity score (0-1, default: 0.7)",
                required=False,
                default=0.7
            )
        ]

    def execute(self, query: str, top_k: int = 5, min_score: float = 0.7) -> ToolResult:
        """Execute semantic search."""
        import time
        start = time.time()

        try:
            # Perform semantic search
            results = self.retrieval_service.search(
                query=query,
                top_k=top_k,
                min_score=min_score
            )

            execution_time = time.time() - start

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=results,
                execution_time=execution_time,
                metadata={
                    "num_results": len(results),
                    "query": query,
                    "top_k": top_k
                }
            )
        except Exception as e:
            execution_time = time.time() - start
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=[],
                error=str(e),
                execution_time=execution_time
            )


class DocumentReaderTool(Tool):
    """Tool for reading full documents."""

    def __init__(self, document_service):
        self.document_service = document_service

    @property
    def name(self) -> str:
        return "read_document"

    @property
    def description(self) -> str:
        return """Read the full content of a specific document by ID or title.
        Use this when you need to read an entire document or when the user
        asks about a specific named document. Example: 'Show me the
        user_guide.pdf document' or 'What's in document ID 123?'"""

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="document_id",
                type="string",
                description="Document ID or title to read",
                required=True
            )
        ]

    def execute(self, document_id: str) -> ToolResult:
        """Read document by ID."""
        import time
        start = time.time()

        try:
            document = self.document_service.get_document(document_id)
            execution_time = time.time() - start

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=document,
                execution_time=execution_time,
                metadata={"document_id": document_id}
            )
        except Exception as e:
            execution_time = time.time() - start
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time
            )


class MetadataSearchTool(Tool):
    """Tool for searching by metadata filters."""

    def __init__(self, retrieval_service):
        self.retrieval_service = retrieval_service

    @property
    def name(self) -> str:
        return "metadata_search"

    @property
    def description(self) -> str:
        return """Search documents by metadata like author, date, category, tags.
        Use this when the query asks for documents with specific properties.
        Example: 'Find documents from 2024' or 'Show papers by John Doe'"""

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="filters",
                type="object",
                description="Metadata filters as key-value pairs (e.g., {'author': 'John', 'year': 2024})",
                required=True
            )
        ]

    def execute(self, filters: Dict[str, Any]) -> ToolResult:
        """Execute metadata search."""
        import time
        start = time.time()

        try:
            results = self.retrieval_service.search_by_metadata(filters)
            execution_time = time.time() - start

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=results,
                execution_time=execution_time,
                metadata={
                    "num_results": len(results),
                    "filters": filters
                }
            )
        except Exception as e:
            execution_time = time.time() - start
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=[],
                error=str(e),
                execution_time=execution_time
            )
```

### Agent Implementation
```python
# rag_factory/strategies/agentic/agent.py
from typing import List, Dict, Any, Optional
import logging
from .tools import Tool, ToolResult
from rag_factory.services.llm import LLMService
from rag_factory.services.llm.base import Message, MessageRole

logger = logging.getLogger(__name__)

class AgentState:
    """State management for agent."""

    def __init__(self):
        self.query = ""
        self.tool_calls: List[Dict[str, Any]] = []
        self.tool_results: List[ToolResult] = []
        self.iterations = 0
        self.max_iterations = 3
        self.final_results = []

    def add_tool_call(self, tool_name: str, parameters: Dict[str, Any]):
        """Record a tool call."""
        self.tool_calls.append({
            "tool": tool_name,
            "parameters": parameters,
            "iteration": self.iterations
        })

    def add_tool_result(self, result: ToolResult):
        """Record a tool result."""
        self.tool_results.append(result)

    def should_continue(self) -> bool:
        """Check if agent should continue iterating."""
        if self.iterations >= self.max_iterations:
            return False
        if not self.tool_results:
            return True
        # Check if we have sufficient results
        successful_results = [r for r in self.tool_results if r.success]
        return len(successful_results) == 0


class SimpleAgent:
    """
    Simple agentic implementation using Anthropic tool use.
    No external framework dependencies.
    """

    def __init__(self, llm_service: LLMService, tools: List[Tool]):
        self.llm_service = llm_service
        self.tools = {tool.name: tool for tool in tools}
        self.tool_definitions = [tool.to_anthropic_tool() for tool in tools]

    def run(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Run the agent to retrieve information.

        Args:
            query: User query
            max_iterations: Maximum tool call iterations

        Returns:
            Dict with results and execution trace
        """
        state = AgentState()
        state.query = query
        state.max_iterations = max_iterations

        logger.info(f"Agent starting for query: {query}")

        # Planning phase: decide which tools to use
        plan = self._plan_retrieval(query)
        logger.info(f"Agent plan: {plan}")

        # Execution phase: run tools
        while state.should_continue():
            state.iterations += 1

            # Get tool selection from LLM
            tool_calls = self._select_tools(query, state)

            if not tool_calls:
                logger.info("No more tools to call, stopping")
                break

            # Execute tools
            for tool_call in tool_calls:
                result = self._execute_tool(tool_call)
                state.add_tool_result(result)

                if result.success:
                    logger.info(
                        f"Tool {result.tool_name} succeeded in {result.execution_time:.2f}s, "
                        f"returned {len(result.data) if isinstance(result.data, list) else 1} results"
                    )
                else:
                    logger.warning(f"Tool {result.tool_name} failed: {result.error}")

        # Synthesis phase: combine results
        final_results = self._synthesize_results(state)

        return {
            "results": final_results,
            "trace": {
                "query": query,
                "iterations": state.iterations,
                "tool_calls": state.tool_calls,
                "tool_results": [
                    {
                        "tool": r.tool_name,
                        "success": r.success,
                        "execution_time": r.execution_time,
                        "num_results": len(r.data) if isinstance(r.data, list) else 1
                    }
                    for r in state.tool_results
                ],
                "plan": plan
            }
        }

    def _plan_retrieval(self, query: str) -> Dict[str, Any]:
        """Plan retrieval strategy."""
        # Use LLM to analyze query and suggest approach
        prompt = f"""Analyze this query and determine the best retrieval strategy:

Query: {query}

Available tools:
{self._format_tool_descriptions()}

Provide a brief plan for how to retrieve the information. Consider:
- What type of query is this? (factual, exploratory, specific document, metadata-based)
- Which tools would be most appropriate?
- Do you need multiple tools or multiple steps?

Plan:"""

        messages = [Message(role=MessageRole.USER, content=prompt)]
        response = self.llm_service.complete(messages, temperature=0.3, max_tokens=200)

        return {
            "reasoning": response.content,
            "cost": response.cost
        }

    def _select_tools(self, query: str, state: AgentState) -> List[Dict[str, Any]]:
        """Select which tools to call."""
        # Build context with previous results
        context = self._build_context(state)

        # Create prompt for tool selection
        prompt = f"""You are a retrieval agent. Select the appropriate tool(s) to answer this query.

Query: {query}

{context}

Available tools:
{self._format_tool_descriptions()}

Think about:
1. What information do we still need?
2. Which tool(s) would help get that information?
3. What parameters should we use?

Select tool(s) to call."""

        messages = [Message(role=MessageRole.USER, content=prompt)]

        # Call LLM with tool definitions
        # Note: Actual Anthropic tool use integration would go here
        # For now, we'll parse the response
        response = self.llm_service.complete(messages, temperature=0.3, max_tokens=300)

        # Parse tool selections from response
        # In real implementation, this would use Anthropic's tool use format
        tool_calls = self._parse_tool_calls(response.content)

        for call in tool_calls:
            state.add_tool_call(call["tool"], call["parameters"])

        return tool_calls

    def _execute_tool(self, tool_call: Dict[str, Any]) -> ToolResult:
        """Execute a tool."""
        tool_name = tool_call["tool"]
        parameters = tool_call["parameters"]

        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                data=None,
                error=f"Tool {tool_name} not found",
                execution_time=0.0
            )

        tool = self.tools[tool_name]
        return tool.execute(**parameters)

    def _synthesize_results(self, state: AgentState) -> List[Any]:
        """Combine and deduplicate results from all tool calls."""
        all_results = []
        seen_ids = set()

        for result in state.tool_results:
            if not result.success:
                continue

            if isinstance(result.data, list):
                for item in result.data:
                    # Deduplicate by chunk_id or doc_id
                    item_id = item.get("chunk_id") or item.get("doc_id") or str(item)
                    if item_id not in seen_ids:
                        seen_ids.add(item_id)
                        all_results.append(item)
            elif result.data:
                all_results.append(result.data)

        # Rank by relevance if we have scores
        if all_results and "score" in all_results[0]:
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return all_results

    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for prompt."""
        descriptions = []
        for tool in self.tools.values():
            params = ", ".join([
                f"{p.name}({p.type}{'*' if p.required else ''})"
                for p in tool.parameters
            ])
            descriptions.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(descriptions)

    def _build_context(self, state: AgentState) -> str:
        """Build context string from previous results."""
        if not state.tool_results:
            return "This is the first retrieval step."

        context = f"Previous tool calls (iteration {state.iterations}):\n"
        for result in state.tool_results:
            if result.success:
                num_results = len(result.data) if isinstance(result.data, list) else 1
                context += f"- {result.tool_name}: {num_results} results found\n"
            else:
                context += f"- {result.tool_name}: failed ({result.error})\n"

        return context

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from LLM response.
        In real implementation, this would use Anthropic's structured tool use.
        """
        # Simplified parsing for demonstration
        # Real implementation would use Anthropic's tool calling format
        tool_calls = []

        # Example: if response mentions "semantic_search"
        if "semantic_search" in response.lower():
            tool_calls.append({
                "tool": "semantic_search",
                "parameters": {
                    "query": response,  # Would extract actual query
                    "top_k": 5
                }
            })

        return tool_calls
```

### Agentic Strategy
```python
# rag_factory/strategies/agentic/strategy.py
from typing import List, Dict, Any, Optional
import logging
from ..base import RAGStrategy
from .agent import SimpleAgent
from .tools import SemanticSearchTool, DocumentReaderTool, MetadataSearchTool
from rag_factory.services.llm import LLMService

logger = logging.getLogger(__name__)

class AgenticRAGStrategy(RAGStrategy):
    """
    Agentic RAG strategy where an agent selects appropriate tools
    to retrieve information based on query type.
    """

    def __init__(
        self,
        llm_service: LLMService,
        retrieval_service: Any,
        document_service: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)

        # Initialize tools
        self.tools = [
            SemanticSearchTool(retrieval_service),
            DocumentReaderTool(document_service),
            MetadataSearchTool(retrieval_service)
        ]

        # Initialize agent
        self.agent = SimpleAgent(llm_service, self.tools)

        # Configuration
        self.max_iterations = config.get("max_iterations", 3) if config else 3
        self.enable_query_analysis = config.get("enable_query_analysis", True) if config else True
        self.fallback_to_semantic = config.get("fallback_to_semantic", True) if config else True

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant information using agentic approach.

        Args:
            query: User query
            top_k: Maximum results to return
            **kwargs: Additional parameters

        Returns:
            List of retrieved chunks/documents with metadata
        """
        logger.info(f"Agentic RAG strategy retrieving for: {query}")

        try:
            # Run agent
            result = self.agent.run(query, max_iterations=self.max_iterations)

            # Extract results
            chunks = result["results"][:top_k]

            # Add strategy metadata
            for chunk in chunks:
                chunk["strategy"] = "agentic"
                chunk["agent_trace"] = result["trace"]

            logger.info(
                f"Agentic retrieval completed: {len(chunks)} results, "
                f"{result['trace']['iterations']} iterations, "
                f"{len(result['trace']['tool_calls'])} tool calls"
            )

            return chunks

        except Exception as e:
            logger.error(f"Agentic retrieval failed: {e}")

            # Fallback to semantic search if enabled
            if self.fallback_to_semantic:
                logger.info("Falling back to semantic search")
                # Simple semantic search fallback
                return self._fallback_search(query, top_k)

            raise

    def _fallback_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback to simple semantic search."""
        # Use semantic search tool directly
        tool = next(t for t in self.tools if t.name == "semantic_search")
        result = tool.execute(query=query, top_k=top_k)

        if result.success:
            return result.data
        return []

    @property
    def name(self) -> str:
        return "agentic"

    @property
    def description(self) -> str:
        return "Agent-based retrieval with dynamic tool selection"
```

---

## Unit Tests

### Test File Locations
- `tests/unit/strategies/agentic/test_strategy.py`
- `tests/unit/strategies/agentic/test_agent.py`
- `tests/unit/strategies/agentic/test_tools.py`

### Test Cases

#### TC5.1.1: Tool Tests
```python
import pytest
from unittest.mock import Mock
from rag_factory.strategies.agentic.tools import (
    SemanticSearchTool, DocumentReaderTool, MetadataSearchTool, ToolResult
)

@pytest.fixture
def mock_retrieval_service():
    service = Mock()
    service.search.return_value = [
        {"chunk_id": 1, "text": "Result 1", "score": 0.9},
        {"chunk_id": 2, "text": "Result 2", "score": 0.8}
    ]
    service.search_by_metadata.return_value = [
        {"doc_id": "doc1", "title": "Document 1"}
    ]
    return service

@pytest.fixture
def mock_document_service():
    service = Mock()
    service.get_document.return_value = {
        "doc_id": "doc1",
        "title": "Test Document",
        "content": "Document content here"
    }
    return service

def test_semantic_search_tool_definition():
    """Test semantic search tool has correct definition."""
    tool = SemanticSearchTool(Mock())

    assert tool.name == "semantic_search"
    assert len(tool.description) > 0
    assert len(tool.parameters) == 3
    assert tool.parameters[0].name == "query"
    assert tool.parameters[0].required == True

def test_semantic_search_tool_execute(mock_retrieval_service):
    """Test semantic search tool execution."""
    tool = SemanticSearchTool(mock_retrieval_service)

    result = tool.execute(query="test query", top_k=5)

    assert result.success == True
    assert result.tool_name == "semantic_search"
    assert len(result.data) == 2
    assert result.execution_time > 0
    mock_retrieval_service.search.assert_called_once_with(
        query="test query",
        top_k=5,
        min_score=0.7
    )

def test_semantic_search_tool_error_handling(mock_retrieval_service):
    """Test semantic search tool handles errors."""
    mock_retrieval_service.search.side_effect = Exception("Search failed")
    tool = SemanticSearchTool(mock_retrieval_service)

    result = tool.execute(query="test query")

    assert result.success == False
    assert result.error == "Search failed"
    assert result.data == []

def test_document_reader_tool(mock_document_service):
    """Test document reader tool."""
    tool = DocumentReaderTool(mock_document_service)

    result = tool.execute(document_id="doc1")

    assert result.success == True
    assert result.data["doc_id"] == "doc1"
    assert result.metadata["document_id"] == "doc1"

def test_metadata_search_tool(mock_retrieval_service):
    """Test metadata search tool."""
    tool = MetadataSearchTool(mock_retrieval_service)

    filters = {"author": "John Doe", "year": 2024}
    result = tool.execute(filters=filters)

    assert result.success == True
    assert len(result.data) == 1
    assert result.metadata["filters"] == filters

def test_tool_to_anthropic_format():
    """Test tool conversion to Anthropic format."""
    tool = SemanticSearchTool(Mock())

    anthropic_tool = tool.to_anthropic_tool()

    assert anthropic_tool["name"] == "semantic_search"
    assert "description" in anthropic_tool
    assert "input_schema" in anthropic_tool
    assert "properties" in anthropic_tool["input_schema"]
    assert "query" in anthropic_tool["input_schema"]["properties"]
    assert "query" in anthropic_tool["input_schema"]["required"]
```

#### TC5.1.2: Agent Tests
```python
import pytest
from unittest.mock import Mock, patch
from rag_factory.strategies.agentic.agent import SimpleAgent, AgentState
from rag_factory.strategies.agentic.tools import Tool, ToolResult

@pytest.fixture
def mock_llm_service():
    service = Mock()
    response = Mock()
    response.content = "I will use semantic_search to find relevant information"
    response.cost = 0.001
    service.complete.return_value = response
    return service

@pytest.fixture
def mock_tool():
    tool = Mock(spec=Tool)
    tool.name = "test_tool"
    tool.description = "Test tool description"
    tool.parameters = []
    tool.to_anthropic_tool.return_value = {
        "name": "test_tool",
        "description": "Test tool"
    }
    tool.execute.return_value = ToolResult(
        tool_name="test_tool",
        success=True,
        data=[{"chunk_id": 1, "text": "Result"}],
        execution_time=0.1
    )
    return tool

def test_agent_state_initialization():
    """Test agent state initialization."""
    state = AgentState()

    assert state.query == ""
    assert state.iterations == 0
    assert state.max_iterations == 3
    assert len(state.tool_calls) == 0
    assert len(state.tool_results) == 0

def test_agent_state_should_continue():
    """Test agent state continuation logic."""
    state = AgentState()

    # Should continue initially
    assert state.should_continue() == True

    # Should stop after max iterations
    state.iterations = 3
    assert state.should_continue() == False

def test_agent_initialization(mock_llm_service, mock_tool):
    """Test agent initialization."""
    agent = SimpleAgent(mock_llm_service, [mock_tool])

    assert len(agent.tools) == 1
    assert "test_tool" in agent.tools
    assert len(agent.tool_definitions) == 1

def test_agent_run_basic(mock_llm_service, mock_tool):
    """Test basic agent run."""
    agent = SimpleAgent(mock_llm_service, [mock_tool])

    # Mock tool selection to stop after one iteration
    with patch.object(agent, '_select_tools', side_effect=[
        [{"tool": "test_tool", "parameters": {}}],
        []  # Return empty to stop
    ]):
        result = agent.run("test query", max_iterations=2)

    assert "results" in result
    assert "trace" in result
    assert result["trace"]["query"] == "test query"
    assert result["trace"]["iterations"] >= 1

def test_agent_planning_phase(mock_llm_service, mock_tool):
    """Test agent planning phase."""
    agent = SimpleAgent(mock_llm_service, [mock_tool])

    plan = agent._plan_retrieval("test query")

    assert "reasoning" in plan
    assert "cost" in plan
    mock_llm_service.complete.assert_called()

def test_agent_tool_execution(mock_llm_service, mock_tool):
    """Test agent executes tools correctly."""
    agent = SimpleAgent(mock_llm_service, [mock_tool])

    tool_call = {"tool": "test_tool", "parameters": {"query": "test"}}
    result = agent._execute_tool(tool_call)

    assert result.success == True
    assert result.tool_name == "test_tool"
    mock_tool.execute.assert_called_once()

def test_agent_synthesize_results(mock_llm_service, mock_tool):
    """Test result synthesis and deduplication."""
    agent = SimpleAgent(mock_llm_service, [mock_tool])

    state = AgentState()
    state.add_tool_result(ToolResult(
        tool_name="tool1",
        success=True,
        data=[
            {"chunk_id": 1, "text": "Result 1", "score": 0.9},
            {"chunk_id": 2, "text": "Result 2", "score": 0.8}
        ],
        execution_time=0.1
    ))
    state.add_tool_result(ToolResult(
        tool_name="tool2",
        success=True,
        data=[
            {"chunk_id": 2, "text": "Result 2", "score": 0.85},  # Duplicate
            {"chunk_id": 3, "text": "Result 3", "score": 0.7}
        ],
        execution_time=0.1
    ))

    results = agent._synthesize_results(state)

    # Should deduplicate chunk_id 2
    assert len(results) == 3
    chunk_ids = [r["chunk_id"] for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates

    # Should be sorted by score
    assert results[0]["score"] >= results[1]["score"]

def test_agent_max_iterations(mock_llm_service, mock_tool):
    """Test agent respects max iterations."""
    agent = SimpleAgent(mock_llm_service, [mock_tool])

    # Mock to always return tool calls
    with patch.object(agent, '_select_tools', return_value=[
        {"tool": "test_tool", "parameters": {}}
    ]):
        result = agent.run("test query", max_iterations=2)

    assert result["trace"]["iterations"] <= 2

def test_agent_failed_tool_handling(mock_llm_service, mock_tool):
    """Test agent handles failed tools gracefully."""
    mock_tool.execute.return_value = ToolResult(
        tool_name="test_tool",
        success=False,
        data=[],
        error="Tool failed",
        execution_time=0.1
    )

    agent = SimpleAgent(mock_llm_service, [mock_tool])

    with patch.object(agent, '_select_tools', side_effect=[
        [{"tool": "test_tool", "parameters": {}}],
        []
    ]):
        result = agent.run("test query")

    # Should complete without crashing
    assert "results" in result
    assert len(result["results"]) == 0  # No successful results
```

#### TC5.1.3: Strategy Tests
```python
import pytest
from unittest.mock import Mock, patch
from rag_factory.strategies.agentic.strategy import AgenticRAGStrategy

@pytest.fixture
def mock_llm_service():
    return Mock()

@pytest.fixture
def mock_retrieval_service():
    service = Mock()
    service.search.return_value = [
        {"chunk_id": 1, "text": "Fallback result", "score": 0.8}
    ]
    return service

@pytest.fixture
def mock_document_service():
    return Mock()

def test_strategy_initialization(mock_llm_service, mock_retrieval_service, mock_document_service):
    """Test strategy initialization."""
    strategy = AgenticRAGStrategy(
        mock_llm_service,
        mock_retrieval_service,
        mock_document_service
    )

    assert strategy.name == "agentic"
    assert len(strategy.tools) == 3
    assert strategy.agent is not None

def test_strategy_retrieve(mock_llm_service, mock_retrieval_service, mock_document_service):
    """Test strategy retrieve method."""
    strategy = AgenticRAGStrategy(
        mock_llm_service,
        mock_retrieval_service,
        mock_document_service
    )

    # Mock agent run
    mock_agent_result = {
        "results": [
            {"chunk_id": 1, "text": "Result 1"},
            {"chunk_id": 2, "text": "Result 2"}
        ],
        "trace": {
            "iterations": 1,
            "tool_calls": [{"tool": "semantic_search"}]
        }
    }

    with patch.object(strategy.agent, 'run', return_value=mock_agent_result):
        results = strategy.retrieve("test query", top_k=5)

    assert len(results) == 2
    assert results[0]["strategy"] == "agentic"
    assert "agent_trace" in results[0]

def test_strategy_fallback_on_error(mock_llm_service, mock_retrieval_service, mock_document_service):
    """Test strategy falls back to semantic search on error."""
    strategy = AgenticRAGStrategy(
        mock_llm_service,
        mock_retrieval_service,
        mock_document_service,
        config={"fallback_to_semantic": True}
    )

    # Mock agent to raise error
    with patch.object(strategy.agent, 'run', side_effect=Exception("Agent failed")):
        results = strategy.retrieve("test query", top_k=5)

    # Should get fallback results
    assert len(results) == 1
    assert results[0]["text"] == "Fallback result"

def test_strategy_config(mock_llm_service, mock_retrieval_service, mock_document_service):
    """Test strategy respects configuration."""
    config = {
        "max_iterations": 5,
        "enable_query_analysis": False,
        "fallback_to_semantic": False
    }

    strategy = AgenticRAGStrategy(
        mock_llm_service,
        mock_retrieval_service,
        mock_document_service,
        config=config
    )

    assert strategy.max_iterations == 5
    assert strategy.enable_query_analysis == False
    assert strategy.fallback_to_semantic == False
```

---

## Integration Tests

### Test File Location
`tests/integration/strategies/test_agentic_integration.py`

### Test Scenarios

#### IS5.1.1: End-to-End Agentic Retrieval
```python
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="API key not set")
def test_agentic_retrieval_workflow(test_db):
    """Test complete agentic retrieval workflow."""
    from rag_factory.services.llm import LLMService, LLMServiceConfig
    from rag_factory.strategies.agentic.strategy import AgenticRAGStrategy

    # Setup services
    llm_config = LLMServiceConfig(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        provider_config={"api_key": os.getenv("ANTHROPIC_API_KEY")}
    )
    llm_service = LLMService(llm_config)

    # Mock retrieval and document services with test data
    retrieval_service = Mock()
    retrieval_service.search.return_value = [
        {"chunk_id": 1, "text": "Python is a programming language", "score": 0.95},
        {"chunk_id": 2, "text": "Python is used for AI and ML", "score": 0.90}
    ]

    document_service = Mock()

    # Create strategy
    strategy = AgenticRAGStrategy(llm_service, retrieval_service, document_service)

    # Test retrieval
    results = strategy.retrieve("What is Python?", top_k=5)

    assert len(results) > 0
    assert all("chunk_id" in r for r in results)
    assert all("strategy" in r for r in results)
    assert all(r["strategy"] == "agentic" for r in results)

    # Check agent trace
    trace = results[0]["agent_trace"]
    assert "iterations" in trace
    assert "tool_calls" in trace
    assert len(trace["tool_calls"]) > 0

@pytest.mark.integration
def test_multi_step_retrieval(test_db):
    """Test multi-step retrieval with multiple tool calls."""
    # Setup with real services
    # Test that agent can make multiple tool calls
    # Verify results are combined correctly
    pass

@pytest.mark.integration
def test_tool_failure_recovery(test_db):
    """Test agent recovers from tool failures."""
    # Setup with mock that fails for first tool
    # Verify agent tries alternative tools
    # Verify final results are still valid
    pass

@pytest.mark.integration
def test_concurrent_agentic_requests(test_db):
    """Test concurrent agentic retrieval requests."""
    import concurrent.futures

    # Setup strategy
    # Run multiple queries concurrently
    # Verify all complete successfully
    pass
```

---

## Performance Benchmarks

```python
# tests/benchmarks/test_agentic_performance.py

@pytest.mark.benchmark
def test_tool_selection_latency():
    """Test tool selection is <500ms."""
    # Measure time for agent to select tools
    # Assert <500ms
    pass

@pytest.mark.benchmark
def test_agentic_retrieval_latency():
    """Test total retrieval time <5s for complex queries."""
    # Run complex query requiring multiple tools
    # Assert total time <5s
    pass

@pytest.mark.benchmark
def test_parallel_tool_execution():
    """Test tools execute in parallel when possible."""
    # Mock multiple independent tools
    # Verify they run concurrently, not sequentially
    pass
```

---

## Definition of Done

- [ ] All tool classes implemented (semantic, document, metadata)
- [ ] Tool parameter validation working
- [ ] SimpleAgent implementation complete
- [ ] LangGraph integration optional and pluggable
- [ ] Anthropic tool use integration working
- [ ] Multi-step retrieval working
- [ ] Result deduplication implemented
- [ ] Query analysis implemented
- [ ] AgenticRAGStrategy class complete
- [ ] Fallback mechanism working
- [ ] All unit tests pass (>90% coverage)
- [ ] All integration tests pass
- [ ] Performance benchmarks meet requirements
- [ ] Logging and observability complete
- [ ] Configuration system working
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] No linting errors

---

## Setup Instructions

### Installation

```bash
# Install dependencies
pip install anthropic pydantic

# Optional: LangGraph integration
pip install langgraph langchain-core
```

### Configuration

```yaml
# config.yaml
strategies:
  agentic:
    enabled: true
    max_iterations: 3
    enable_query_analysis: true
    fallback_to_semantic: true

    # LLM for agent
    llm:
      provider: anthropic
      model: claude-3-haiku-20240307

    # Available tools
    tools:
      - semantic_search
      - read_document
      - metadata_search
```

### Usage Example

```python
from rag_factory.strategies.agentic import AgenticRAGStrategy
from rag_factory.services.llm import LLMService, LLMServiceConfig

# Setup LLM service
llm_config = LLMServiceConfig(provider="anthropic", model="claude-3-haiku-20240307")
llm_service = LLMService(llm_config)

# Create strategy
strategy = AgenticRAGStrategy(
    llm_service=llm_service,
    retrieval_service=retrieval_service,
    document_service=document_service,
    config={"max_iterations": 3}
)

# Retrieve with agent
results = strategy.retrieve("What are the safety guidelines for using the API?")

# View agent trace
trace = results[0]["agent_trace"]
print(f"Agent used {len(trace['tool_calls'])} tool calls in {trace['iterations']} iterations")
for tool_call in trace['tool_calls']:
    print(f"  - {tool_call['tool']} with {tool_call['parameters']}")
```

---

## Notes for Developers

1. **Tool Design**: Keep tool descriptions clear and specific. The LLM uses these to decide which tool to call.

2. **Agent Frameworks**: Start with SimpleAgent. Only add LangGraph if you need complex workflows.

3. **Cost Management**: Use Haiku for tool selection to minimize costs. Each agent run makes multiple LLM calls.

4. **Iteration Limits**: Set reasonable max_iterations (2-3) to prevent excessive API calls.

5. **Fallback Strategy**: Always enable fallback to semantic search for reliability.

6. **Tool Parameters**: Validate tool parameters strictly to prevent errors during execution.

7. **Observability**: Log all tool selections and executions for debugging.

8. **Testing**: Mock LLM responses in unit tests to avoid API costs and ensure deterministic behavior.

9. **Anthropic Tool Use**: When available, use Anthropic's native tool calling format for better accuracy.

10. **Result Quality**: The agent's tool selection quality depends heavily on good tool descriptions and prompt engineering.
