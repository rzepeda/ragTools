"""Tests for workflow definitions and structure.

This module tests that each of the 6 predefined workflows is correctly structured.
"""

import pytest
from rag_factory.strategies.agentic.workflows import WORKFLOWS, WorkflowStep


class TestWorkflowDefinitions:
    """Test that all workflow definitions are valid."""

    def test_all_workflows_exist(self):
        """Test that all 6 workflows are defined."""
        assert len(WORKFLOWS) == 6
        for i in range(1, 7):
            assert i in WORKFLOWS

    def test_workflow_structure(self):
        """Test that each workflow has required fields."""
        required_fields = ["name", "description", "steps"]
        
        for plan_num, workflow in WORKFLOWS.items():
            for field in required_fields:
                assert field in workflow, f"Workflow {plan_num} missing {field}"
            
            # Steps should be a list
            assert isinstance(workflow["steps"], list)
            assert len(workflow["steps"]) > 0

    def test_workflow_steps_structure(self):
        """Test that each step has required fields."""
        for plan_num, workflow in WORKFLOWS.items():
            for step_idx, step in enumerate(workflow["steps"]):
                assert "tool" in step, f"Workflow {plan_num} step {step_idx} missing tool"
                assert "params" in step, f"Workflow {plan_num} step {step_idx} missing params"
                
                # Tool should be a valid tool name
                valid_tools = ["semantic_search", "metadata_search", "hybrid_search", "read_document"]
                assert step["tool"] in valid_tools


class TestWorkflow1SimpleSemanticSearch:
    """Test workflow 1: Simple Semantic Search."""

    def test_workflow_1_definition(self):
        """Test workflow 1 structure."""
        workflow = WORKFLOWS[1]
        
        assert workflow["name"] == "Simple Semantic Search"
        assert len(workflow["steps"]) == 1
        assert workflow["steps"][0]["tool"] == "semantic_search"

    def test_workflow_1_parameters(self):
        """Test workflow 1 has correct parameters."""
        step = WORKFLOWS[1]["steps"][0]
        params = step["params"]
        
        assert "query" in params
        assert "top_k" in params


class TestWorkflow2MetadataFiltered:
    """Test workflow 2: Metadata-Filtered Search."""

    def test_workflow_2_definition(self):
        """Test workflow 2 structure."""
        workflow = WORKFLOWS[2]
        
        assert workflow["name"] == "Metadata-Filtered Search"
        assert len(workflow["steps"]) == 1
        assert workflow["steps"][0]["tool"] == "metadata_search"

    def test_workflow_2_parameters(self):
        """Test workflow 2 has correct parameters."""
        step = WORKFLOWS[2]["steps"][0]
        params = step["params"]
        
        assert "query" in params
        assert "metadata_filter" in params
        assert "top_k" in params


class TestWorkflow3SearchThenRead:
    """Test workflow 3: Search Then Read Document."""

    def test_workflow_3_definition(self):
        """Test workflow 3 structure."""
        workflow = WORKFLOWS[3]
        
        assert workflow["name"] == "Search Then Read Document"
        assert len(workflow["steps"]) == 2
        assert workflow["steps"][0]["tool"] == "semantic_search"
        assert workflow["steps"][1]["tool"] == "read_document"

    def test_workflow_3_chaining(self):
        """Test workflow 3 chains results correctly."""
        step1 = WORKFLOWS[3]["steps"][0]
        step2 = WORKFLOWS[3]["steps"][1]
        
        # First step should search
        assert "query" in step1["params"]
        
        # Second step should use result from first
        assert "document_id" in step2["params"]
        # Should reference step 1 result
        assert "{from_step_1}" in str(step2["params"]["document_id"]) or \
               "{document_id}" in str(step2["params"]["document_id"])


class TestWorkflow4HybridSearch:
    """Test workflow 4: Hybrid Search."""

    def test_workflow_4_definition(self):
        """Test workflow 4 structure."""
        workflow = WORKFLOWS[4]
        
        assert workflow["name"] == "Hybrid Search"
        assert len(workflow["steps"]) == 1
        assert workflow["steps"][0]["tool"] == "hybrid_search"

    def test_workflow_4_parameters(self):
        """Test workflow 4 has correct parameters."""
        step = WORKFLOWS[4]["steps"][0]
        params = step["params"]
        
        assert "query" in params
        assert "metadata_hint" in params or "top_k" in params


class TestWorkflow5MultiStepRefinement:
    """Test workflow 5: Multi-Step Refinement."""

    def test_workflow_5_definition(self):
        """Test workflow 5 structure."""
        workflow = WORKFLOWS[5]
        
        assert workflow["name"] == "Multi-Step Refinement"
        # Should have multiple steps for refinement
        assert len(workflow["steps"]) >= 2

    def test_workflow_5_progression(self):
        """Test workflow 5 progresses from broad to specific."""
        steps = WORKFLOWS[5]["steps"]
        
        # First step should be broad search
        assert steps[0]["tool"] in ["semantic_search", "hybrid_search"]
        
        # Later steps should refine or read documents
        assert any(step["tool"] == "read_document" for step in steps[1:])


class TestWorkflow6DirectDocumentAccess:
    """Test workflow 6: Direct Document Access."""

    def test_workflow_6_definition(self):
        """Test workflow 6 structure."""
        workflow = WORKFLOWS[6]
        
        assert workflow["name"] == "Direct Document Access"
        assert len(workflow["steps"]) == 1
        assert workflow["steps"][0]["tool"] == "read_document"

    def test_workflow_6_parameters(self):
        """Test workflow 6 expects document_id."""
        step = WORKFLOWS[6]["steps"][0]
        params = step["params"]
        
        assert "document_id" in params


class TestWorkflowParameterSubstitution:
    """Test parameter substitution patterns."""

    def test_query_placeholder(self):
        """Test that {query} placeholder is used correctly."""
        for plan_num, workflow in WORKFLOWS.items():
            for step in workflow["steps"]:
                params_str = str(step["params"])
                if "{query}" in params_str:
                    # Should be in a query or similar parameter
                    assert "query" in step["params"]

    def test_step_reference_placeholder(self):
        """Test that {from_step_X} placeholders are used correctly."""
        for plan_num, workflow in WORKFLOWS.items():
            if len(workflow["steps"]) > 1:
                # Multi-step workflows should reference previous steps
                for step_idx, step in enumerate(workflow["steps"][1:], start=1):
                    params_str = str(step["params"])
                    # If referencing previous step, should use from_step_X pattern
                    if "from_step" in params_str:
                        assert f"from_step_{step_idx}" in params_str or \
                               "from_step_1" in params_str  # Most common case
