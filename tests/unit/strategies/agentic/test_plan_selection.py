"""Tests for workflow plan selection and LLM response parsing.

This module tests the flexible JSON parsing that handles various LLM response formats.
"""

import pytest
from rag_factory.strategies.agentic.workflows import extract_plan_from_response


class TestPlanSelectionParsing:
    """Test parsing of LLM responses for plan selection."""

    def test_parse_clean_json(self):
        """Test parsing perfect JSON response."""
        response = '{"plan": 3, "reasoning": "User wants full document"}'
        
        result = extract_plan_from_response(response)
        
        assert result["plan"] == 3
        assert "full document" in result["reasoning"].lower()

    def test_parse_json_with_leading_text(self):
        """Test parsing when LLM adds explanation before JSON."""
        response = '''Based on the query, I'll choose workflow 3.
        
        {"plan": 3, "reasoning": "Comprehensive information needed"}
        
        This should work well for this type of query.'''
        
        result = extract_plan_from_response(response)
        
        assert result["plan"] == 3
        assert "comprehensive" in result["reasoning"].lower()

    def test_parse_json_with_trailing_text(self):
        """Test parsing when LLM adds explanation after JSON."""
        response = '''{"plan": 2, "reasoning": "Query mentions metadata"}
        
        I selected plan 2 because the query contains author and date information.'''
        
        result = extract_plan_from_response(response)
        
        assert result["plan"] == 2

    def test_parse_malformed_json_with_plan_field(self):
        """Test fallback when JSON is broken but plan field is present."""
        response = 'I think plan: 2 would be best because it has metadata filters'
        
        result = extract_plan_from_response(response)
        
        assert result["plan"] == 2

    def test_parse_just_number(self):
        """Test when LLM only returns a number."""
        response = "5"
        
        result = extract_plan_from_response(response)
        
        assert result["plan"] == 5

    def test_parse_number_in_sentence(self):
        """Test extracting plan number from natural language."""
        response = "I recommend using workflow 4 for this query."
        
        result = extract_plan_from_response(response)
        
        assert result["plan"] == 4

    def test_parse_completely_invalid_response(self):
        """Test fallback to plan 1 when nothing parseable."""
        response = "I don't understand this query"
        
        result = extract_plan_from_response(response)
        
        assert result["plan"] == 1  # Default fallback
        assert "fallback" in result["reasoning"].lower()

    def test_parse_number_outside_valid_range_high(self):
        """Test handling of plan numbers above valid range."""
        response = '{"plan": 99, "reasoning": "Invalid"}'
        
        result = extract_plan_from_response(response)
        
        assert result["plan"] == 1  # Fallback to default

    def test_parse_number_outside_valid_range_low(self):
        """Test handling of plan numbers below valid range."""
        response = '{"plan": 0, "reasoning": "Invalid"}'
        
        result = extract_plan_from_response(response)
        
        assert result["plan"] == 1  # Fallback to default

    def test_parse_json_with_extra_fields(self):
        """Test parsing JSON with additional fields."""
        response = '''{"plan": 3, "reasoning": "Full document needed", 
                       "confidence": 0.95, "alternative": 1}'''
        
        result = extract_plan_from_response(response)
        
        assert result["plan"] == 3
        assert "reasoning" in result

    def test_parse_multiline_json(self):
        """Test parsing formatted/pretty-printed JSON."""
        response = '''{
            "plan": 6,
            "reasoning": "Document ID mentioned in query"
        }'''
        
        result = extract_plan_from_response(response)
        
        assert result["plan"] == 6

    def test_parse_plan_with_quotes_variations(self):
        """Test parsing with different quote styles."""
        responses = [
            "plan: 3",
            '"plan": 3',
            "'plan': 3",
            "plan = 3"
        ]
        
        for response in responses:
            result = extract_plan_from_response(response)
            assert result["plan"] == 3

    def test_parse_empty_response(self):
        """Test handling of empty response."""
        response = ""
        
        result = extract_plan_from_response(response)
        
        assert result["plan"] == 1  # Default fallback

    def test_parse_none_response(self):
        """Test handling of None response."""
        response = None
        
        result = extract_plan_from_response(response)
        
        assert result["plan"] == 1  # Default fallback


class TestPlanSelectionWithMockLLM:
    """Test plan selection with various mock LLM responses."""

    MOCK_RESPONSES = {
        "factual_query": '{"plan": 1, "reasoning": "Simple factual question"}',
        
        "metadata_query": '''I'll use plan 2 for this.
        {"plan": 2, "reasoning": "Query mentions NASA and date"}''',
        
        "full_document": 'plan: 3\nreasoning: User wants comprehensive info',
        
        "hybrid_search": "4",
        
        "exploratory": '''Based on the query characteristics, 
        I recommend workflow 5 because it's exploratory.
        
        {"plan": 5, "reasoning": "Broad research query"}
        
        This will provide comprehensive results.''',
        
        "direct_document": '{"plan": 6, "reasoning": "Document ID provided"}',
        
        "invalid": "I'm not sure what to do here",
    }

    @pytest.mark.parametrize("response_key,expected_plan", [
        ("factual_query", 1),
        ("metadata_query", 2),
        ("full_document", 3),
        ("hybrid_search", 4),
        ("exploratory", 5),
        ("direct_document", 6),
        ("invalid", 1),  # Fallback
    ])
    def test_mock_llm_responses(self, response_key, expected_plan):
        """Test that all mock LLM responses parse to correct plans."""
        response = self.MOCK_RESPONSES[response_key]
        result = extract_plan_from_response(response)
        assert result["plan"] == expected_plan
