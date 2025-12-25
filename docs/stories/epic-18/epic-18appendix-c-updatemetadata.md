Metadata Extraction Enhancement Instructions
Overview
Replace the heuristic-based extract_metadata_from_query() function with an LLM-powered extraction system that dynamically matches query content to a provided metadata schema.
Requirements
1. Function Signature
pythonasync def extract_metadata_from_query(
    query: str,
    schema: Dict[str, str],  # field_name -> description/type
    llm_service: ILLMService
) -> Dict[str, Any]:
2. Schema Format
The schema parameter should accept field definitions like:
pythonschema = {
    "document_id": "specific document identifier or reference number",
    "year": "publication or event year (YYYY format)",
    "author": "author, organization, or publisher name",
    "publisher": "publishing organization",
    "document_type": "type of document (report, paper, specification, etc.)",
    "language": "document language code (en, es, etc.)"
}
```

**3. LLM Prompt Construction**

Build a prompt that:
- Clearly explains the extraction task
- Provides the query text
- Lists the schema fields with descriptions
- Requests JSON-only output
- Handles cases where no metadata is found
- Is forgiving of LLM formatting variations
- **Emphasizes document_id extraction from explicit references**

Example prompt template:
```
You are a metadata extraction assistant. Extract structured metadata from the user query based on the provided schema.

USER QUERY:
"{query}"

METADATA SCHEMA:
{formatted_schema_fields}

INSTRUCTIONS:
1. Extract ONLY metadata that is explicitly mentioned or clearly implied in the query
2. Match extracted values to the schema field names exactly
3. Pay special attention to document identifiers, reference numbers, or document IDs (e.g., "DOC-123", "document 456", "report ABC-2023")
4. Return ONLY valid JSON with extracted fields
5. If no metadata matches, return empty JSON: {{}}
6. Do not include explanations, markdown formatting, or additional text

OUTPUT FORMAT:
Return only the JSON object, nothing else.
4. Forgiving JSON Extraction
The response parser must handle various LLM output formats:

Clean JSON: {"year": 2023, "document_id": "DOC-123"}
Markdown wrapped: ```json\n{"year": 2023}\n```
Text with JSON: Here is the metadata: {"year": 2023}
Extra whitespace/newlines
Single quotes instead of double quotes (fix if possible)

Implementation approach:

Try direct json.loads() first
If that fails, use regex to extract JSON-like patterns: \{[^}]+\}
Clean common formatting issues (strip markdown, fix quotes)
If still fails, return empty dict {} rather than raising exception
Log warnings for malformed responses (for debugging)

5. Integration with Existing Services

Use the ILLMService interface from Epic 11 (already in project)
Call await llm_service.complete(prompt, temperature=0.0) for deterministic extraction
Set low temperature (0.0-0.1) for consistent structured output
Handle LLM service errors gracefully (return empty dict on failure)

6. Error Handling
Handle these cases:

LLM service unavailable → return {}
Malformed JSON response → return {} and log warning
Schema validation errors → return {} and log warning
Network timeout → return {} and log error
Empty query → return {} immediately (skip LLM call)

7. Validation
After extraction, validate:

All returned keys exist in the schema
Remove any keys not in schema (LLM hallucination prevention)
Basic type checking where schema provides hints (year should be int/string of digits)
Special handling for document_id: preserve exact format/case as extracted
Log warnings for unexpected keys

8. Example Test Cases
The implementation should handle:
python# Test 1: Simple year extraction
query = "documents discovered in 2023"
schema = {"year": "publication year"}
expected = {"year": "2023"} or {"year": 2023}

# Test 2: Multiple fields including document_id
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

# Test 3: Document ID variations
query = "show me report #12345"
schema = {"document_id": "document reference"}
expected = {"document_id": "12345"} or {"document_id": "#12345"}

query = "find document ABC-DEF-2023"
schema = {"document_id": "document identifier"}
expected = {"document_id": "ABC-DEF-2023"}

# Test 4: No metadata
query = "tell me about the results"
schema = {"year": "year", "author": "organization", "document_id": "document ID"}
expected = {}

# Test 5: Partial match
query = "the 2019 study"
schema = {
    "year": "year",
    "author": "author",
    "publisher": "publisher",
    "document_id": "document ID"
}
expected = {"year": "2019"}  # Only year found, others omitted

# Test 6: Document ID only
query = "retrieve document REF-789"
schema = {"document_id": "reference number", "year": "year"}
expected = {"document_id": "REF-789"}
9. Performance Considerations

Cache LLM responses for identical query+schema combinations (optional optimization)
Consider batching multiple queries if called in rapid succession (future enhancement)
Set reasonable timeout for LLM calls (5-10 seconds)
Log LLM call duration for monitoring

10. Documentation Requirements
Add docstring with:

Clear explanation of schema format
Example usage code showing document_id extraction
Note about async requirement
Error handling behavior
Performance characteristics (LLM call overhead)
Document ID format preservation (maintains exact casing and formatting)

Example docstring snippet:
python"""
Extract metadata from natural language query using LLM-based parsing.

Args:
    query: Natural language query string
    schema: Dictionary mapping field names to descriptions
           Example: {
               "document_id": "document identifier or reference",
               "year": "publication year",
               "author": "author or organization"
           }
    llm_service: LLM service instance for extraction

Returns:
    Dictionary of extracted metadata matching schema fields.
    Empty dict if no metadata found or extraction fails.
    
    Examples:
        query = "show me NASA report DOC-2023-001 from last year"
        schema = {"document_id": "document ID", "author": "organization"}
        result = {"document_id": "DOC-2023-001", "author": "NASA"}

Note:
    - Document IDs preserve exact format/casing as extracted
    - Unmatched schema fields are omitted from result
    - LLM failures return empty dict gracefully
    - Temperature set to 0.0 for deterministic extraction
"""
11. Integration Points
This function will be called from:

Agentic RAG's metadata filter tool (existing code location)
Before vector/keyword search execution
Schema will come from document index metadata fields
Document ID filtering is high-priority for specific document retrieval

Make sure to:

Keep the same function name for backward compatibility
Update any imports needed
Add tests in tests/unit/strategies/retrieval/test_metadata_extraction.py
Include document_id-specific test cases


Implementation Checklist

 Create async function with correct signature
 Build LLM prompt template with schema formatting
 Add explicit document_id handling in prompt instructions
 Implement forgiving JSON extraction with multiple fallback strategies
 Add schema validation and key filtering
 Preserve document_id format exactly as extracted (no normalization)
 Integrate with ILLMService interface
 Add comprehensive error handling
 Write unit tests with mock LLM service
 Add specific test cases for document_id extraction patterns
 Write integration tests with real LLM (optional, gated by RUN_LLM_TESTS)
 Update function docstring with document_id examples
 Log debug information for troubleshooting

Special Considerations for document_id
1. Common Document ID Patterns to Handle:

Alphanumeric with hyphens: DOC-2023-001, REF-ABC-123
Numeric only: 12345, #789
Prefixed: document 456, report ABC
Mixed case: Doc-2023-A, REF_xyz_001

2. Extraction Rules:

Preserve exact casing (don't lowercase/uppercase)
Preserve special characters (hyphens, underscores, hash)
Strip common prefixes only if ambiguous ("document", "report", "ref")
Log when multiple potential IDs found in query

3. Validation:

No length restrictions (IDs vary by system)
Allow alphanumeric + common separators (-, _, #, .)
Flag but don't reject unusual patterns (log for review)

Notes

This enhancement aligns with Epic 11's dependency injection pattern
Uses existing LLM service infrastructure (no new services needed)
Improves on heuristic approach with flexible schema-driven extraction
Maintains backward compatibility by keeping function name and general behavior
Document ID support enables precise document retrieval in agentic workflows