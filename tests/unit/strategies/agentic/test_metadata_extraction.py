import pytest
from rag_factory.registry.service_registry import ServiceRegistry
from rag_factory.strategies.agentic.workflows import extract_metadata_from_query

@pytest.fixture(scope="module")
def llm_service():
    """Fixture to provide an LLM service instance."""
    # Ensure the .env file is loaded for credentials
    from dotenv import load_dotenv
    load_dotenv()
    
    registry = ServiceRegistry("config/services.yaml")
    # Assuming 'llm_local' is the service name for the LLM in services.yaml
    return registry.get("llm_local")

@pytest.mark.asyncio
async def test_extract_metadata_llm_simple_year(llm_service):
    """Test simple year extraction."""
    query = "documents discovered in 2023"
    schema = {"year": "publication year"}
    expected = {"year": "2023"}
    
    result = await extract_metadata_from_query(query, schema, llm_service)
    # LLM might return int or string
    assert "year" in result
    assert str(result["year"]) == "2023"

@pytest.mark.asyncio
async def test_extract_metadata_llm_multiple_fields(llm_service):
    """Test multiple fields including document_id."""
    query = "NASA report DOC-2020-Mars from 2020 about Mars"
    schema = {
        "document_id": "document identifier",
        "year": "year",
        "author": "organization",
        "topic": "subject"
    }
    expected = {
        "document_id": "DOC-2020-Mars",
        "year": "2020",
        "author": "NASA",
        "topic": "Mars"
    }
    
    result = await extract_metadata_from_query(query, schema, llm_service)
    # The LLM can be inconsistent with extractions, so we check for a subset.
    for key, value in expected.items():
        assert key in result, f"Expected key '{key}' not in result"
        assert str(result[key]).lower() == str(value).lower()

@pytest.mark.asyncio
async def test_extract_metadata_llm_doc_id_variations(llm_service):
    """Test document ID variations."""
    query = "show me report #12345"
    schema = {"document_id": "document reference"}
    
    result = await extract_metadata_from_query(query, schema, llm_service)
    assert "document_id" in result
    # The extracted ID could be "#12345" or "12345"
    assert result["document_id"].strip() in ["#12345", "12345"]

    query = "find document ABC-DEF-2023"
    schema = {"document_id": "document identifier"}
    expected = {"document_id": "ABC-DEF-2023"}
    
    result = await extract_metadata_from_query(query, schema, llm_service)
    assert result == expected

@pytest.mark.asyncio
async def test_extract_metadata_llm_no_metadata(llm_service):
    """Test no metadata found."""
    query = "tell me about the results"
    schema = {"year": "year", "author": "organization", "document_id": "document ID"}
    expected = {}
    
    result = await extract_metadata_from_query(query, schema, llm_service)
    assert result == expected

@pytest.mark.asyncio
async def test_extract_metadata_llm_partial_match(llm_service):
    """Test partial match."""
    query = "the 2019 study"
    schema = {
        "year": "year",
        "author": "author",
        "publisher": "publisher",
        "document_id": "document ID"
    }
    
    result = await extract_metadata_from_query(query, schema, llm_service)
    assert "year" in result
    assert str(result['year']) == "2019"
    assert len(result.keys()) == 1


@pytest.mark.asyncio
async def test_extract_metadata_llm_doc_id_only(llm_service):
    """Test document ID only."""
    query = "retrieve document REF-789"
    schema = {"document_id": "reference number", "year": "year"}
    expected = {"document_id": "REF-789"}
    
    result = await extract_metadata_from_query(query, schema, llm_service)
    assert result == expected
