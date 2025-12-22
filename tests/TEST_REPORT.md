# Test Results Summary Report

**Generated:** 2025-12-20  
**Test Duration:** 131m 16s  
**Total Test Files:** 206

---

## 📊 Overall Statistics

### File-Level Results
| Status | Count | Percentage |
|--------|-------|------------|
| ✅ **Passed** | 186 | 90.3% |
| ❌ **Failed** | 16 | 7.8% |
| ⏭️ **Skipped** | 2 | 1.0% |
| ⏱️ **Timeout** | 2 | 1.0% |

### Test-Level Results
| Status | Count | Percentage |
|--------|-------|------------|
| ✅ **Passed** | 2,145 | 93.6% |
| ❌ **Failed** | 92 | 4.0% |
| ⏭️ **Skipped** | 54 | 2.4% |
| 📊 **Total** | 2,291 | 100% |

---

## 🔍 Errors Grouped by Category


### Database Migration - Multiple Heads (3 files)

🔴 **HIGH PRIORITY** - `tests/integration/database/test_migration_integration.py`

- **Error Type:** Alembic Multiple Heads
- **Description:** Multiple migration branches: finetuned_schema, contextual_schema, hierarchical_schema, context_aware_schema, keyword_schema
- **Recommended Fix:** Run: alembic merge heads
- **Failed Tests:** 8

🔴 **HIGH PRIORITY** - `tests/integration/database/test_migration_validator_integration.py`

- **Error Type:** Alembic Multiple Heads
- **Description:** Multiple migration branches: finetuned_schema, contextual_schema, hierarchical_schema, context_aware_schema, keyword_schema
- **Recommended Fix:** Run: alembic merge heads
- **Failed Tests:** 22

🔴 **HIGH PRIORITY** - `tests/unit/database/test_migrations.py`

- **Error Type:** Alembic Multiple Heads
- **Description:** Multiple migration branches: finetuned_schema, contextual_schema, hierarchical_schema, context_aware_schema, keyword_schema
- **Recommended Fix:** Run: alembic merge heads
- **Failed Tests:** 12


### Neo4j Connection (1 file)

🟡 **MEDIUM PRIORITY** - `tests/integration_real/test_neo4j_real.py`

- **Error Type:** Neo4j Connection
- **Description:** Cannot connect to Neo4j database
- **Recommended Fix:** Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env
- **Failed Tests:** 0


### Import/Module Errors (1 file)

🟡 **MEDIUM PRIORITY** - `tests/unit/gui/test_backend_integration.py`

- **Error Type:** Import Error
- **Description:** Missing module: rag_factory.core.exceptions
- **Recommended Fix:** Fix import path or install rag_factory.core.exceptions
- **Failed Tests:** 0


### GUI/Tkinter Issues (3 files)

🟡 **MEDIUM PRIORITY** - `tests/unit/gui/test_main_window.py`

- **Error Type:** GUI/Tkinter
- **Description:** Tkinter or GUI-related error
- **Recommended Fix:** Check GUI initialization and mocking
- **Failed Tests:** 12

🟡 **MEDIUM PRIORITY** - `tests/unit/gui/test_retrieval_operations.py`

- **Error Type:** GUI/Tkinter
- **Description:** Tkinter or GUI-related error
- **Recommended Fix:** Check GUI initialization and mocking
- **Failed Tests:** 2

🟡 **MEDIUM PRIORITY** - `tests/unit/gui/test_utility_operations.py`

- **Error Type:** GUI/Tkinter
- **Description:** Tkinter or GUI-related error
- **Recommended Fix:** Check GUI initialization and mocking
- **Failed Tests:** 2


### Documentation Errors (2 files)

🟢 **LOW PRIORITY** - `tests/unit/documentation/test_code_examples.py`

- **Error Type:** Documentation Syntax
- **Description:** Code examples have syntax errors
- **Recommended Fix:** Fix code syntax in documentation
- **Failed Tests:** 2

🟢 **LOW PRIORITY** - `tests/unit/documentation/test_links.py`

- **Error Type:** Documentation Links
- **Description:** Broken internal links in documentation
- **Recommended Fix:** Fix documentation link paths
- **Failed Tests:** 4


### Mock/Test Configuration (2 files)

🟡 **MEDIUM PRIORITY** - `tests/test_mock_registry.py`

- **Error Type:** Mock Configuration
- **Description:** Mock object has incorrect configuration
- **Recommended Fix:** Update mock configuration or test setup
- **Failed Tests:** 0

🟡 **MEDIUM PRIORITY** - `tests/unit/services/database/test_migration_validator.py`

- **Error Type:** Mock Configuration
- **Description:** Mock object is not subscriptable
- **Recommended Fix:** Update mock configuration or test setup
- **Failed Tests:** 12


### Type/Attribute Errors (1 file)

🟡 **MEDIUM PRIORITY** - `tests/integration/test_hybrid_search_pair.py`

- **Error Type:** TypeError
- **Description:** object Mock can't be used in 'await' expression
- **Recommended Fix:** Fix type mismatches or attribute access
- **Failed Tests:** 2


### Assertion Failures (1 file)

🟡 **MEDIUM PRIORITY** - `tests/unit/registry/test_service_factory.py`

- **Error Type:** Assertion Failure
- **Description:** assert <rag_factory.services.llm.service.LLMService object at 0x7a6006ec3a10> is <Mock name='OpenAIL
- **Recommended Fix:** Review test expectations
- **Failed Tests:** 6


### Other Errors (2 files)

🟢 **LOW PRIORITY** - `tests/integration/registry/test_registry_integration.py`

- **Error Type:** Other
- **Description:** See test output for details
- **Recommended Fix:** Investigate error in test results
- **Failed Tests:** 2

🟢 **LOW PRIORITY** - `tests/integration/services/test_service_implementations.py`

- **Error Type:** Other
- **Description:** See test output for details
- **Recommended Fix:** Investigate error in test results
- **Failed Tests:** 4


---

## 📋 Error Type Summary

| Category | Count | Description |
|----------|-------|-------------|
| ⚙️ **Configuration Issues** | 4 | Missing API keys, database connections, model paths, or environment setup |
| 🐛 **Requirement/Implementation Issues** | 9 | Code bugs, test logic errors, or API mismatches |
| 📦 **Dependency Issues** | 1 | Missing modules or import errors |
| ❓ **Unknown/Other** | 2 | Unclassified errors requiring investigation |

---

## 🔧 Recommended Actions (Priority Order)

### 🔴 High Priority

1. **tests/integration/database/test_migration_integration.py**
   - **Issue:** Multiple migration branches: finetuned_schema, contextual_schema, hierarchical_schema, context_aware_schema, keyword_schema
   - **Fix:** Run: alembic merge heads

2. **tests/integration/database/test_migration_validator_integration.py**
   - **Issue:** Multiple migration branches: finetuned_schema, contextual_schema, hierarchical_schema, context_aware_schema, keyword_schema
   - **Fix:** Run: alembic merge heads

3. **tests/unit/database/test_migrations.py**
   - **Issue:** Multiple migration branches: finetuned_schema, contextual_schema, hierarchical_schema, context_aware_schema, keyword_schema
   - **Fix:** Run: alembic merge heads

### 🟡 Medium Priority

1. **tests/integration_real/test_neo4j_real.py**
   - **Issue:** Cannot connect to Neo4j database
   - **Fix:** Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env

2. **tests/unit/gui/test_backend_integration.py**
   - **Issue:** Missing module: rag_factory.core.exceptions
   - **Fix:** Fix import path or install rag_factory.core.exceptions

3. **tests/unit/gui/test_main_window.py**
   - **Issue:** Tkinter or GUI-related error
   - **Fix:** Check GUI initialization and mocking

4. **tests/unit/gui/test_retrieval_operations.py**
   - **Issue:** Tkinter or GUI-related error
   - **Fix:** Check GUI initialization and mocking

5. **tests/unit/gui/test_utility_operations.py**
   - **Issue:** Tkinter or GUI-related error
   - **Fix:** Check GUI initialization and mocking

6. **tests/test_mock_registry.py**
   - **Issue:** Mock object has incorrect configuration
   - **Fix:** Update mock configuration or test setup

7. **tests/unit/services/database/test_migration_validator.py**
   - **Issue:** Mock object is not subscriptable
   - **Fix:** Update mock configuration or test setup

8. **tests/integration/test_hybrid_search_pair.py**
   - **Issue:** object Mock can't be used in 'await' expression
   - **Fix:** Fix type mismatches or attribute access

9. **tests/unit/registry/test_service_factory.py**
   - **Issue:** assert <rag_factory.services.llm.service.LLMService object at 0x7a6006ec3a10> is <Mock name='OpenAIL
   - **Fix:** Review test expectations


---

## 📁 Complete List of Failed Test Files

- `tests/integration/database/test_migration_integration.py`
- `tests/integration/database/test_migration_validator_integration.py`
- `tests/integration/registry/test_registry_integration.py`
- `tests/integration/services/test_service_implementations.py`
- `tests/integration/test_hybrid_search_pair.py`
- `tests/integration_real/test_neo4j_real.py`
- `tests/test_mock_registry.py`
- `tests/unit/database/test_migrations.py`
- `tests/unit/documentation/test_code_examples.py`
- `tests/unit/documentation/test_links.py`
- `tests/unit/gui/test_backend_integration.py`
- `tests/unit/gui/test_main_window.py`
- `tests/unit/gui/test_retrieval_operations.py`
- `tests/unit/gui/test_utility_operations.py`
- `tests/unit/registry/test_service_factory.py`
- `tests/unit/services/database/test_migration_validator.py`


---

## ⏱️ Timeout Test Files

- `tests/integration/gui/test_end_to_end_workflow.py`
- `tests/integration/gui/test_gui_launch.py`

---

## 🧪 Individual Failed Tests

<details>
<summary>Click to expand full list of {len(re.findall(r'FAILED', content))} failed tests</summary>

- `tests/integration/database/test_migration_integration.py::TestMigrationIntegration::test_migration_with_existing_data`
- `tests/integration/database/test_migration_integration.py::TestMigrationIntegration::test_migration_with_existing_data - alembic.util.exc.CommandError: Multiple head revisions are present for given argument 'head'; please specify a specific target revision, '<branchname>@head' to narrow to a specific head, or 'heads' for all heads`
- `tests/integration/database/test_migration_integration.py::TestMigrationIntegration::test_pgvector_extension_installed`
- `tests/integration/database/test_migration_integration.py::TestMigrationIntegration::test_real_migration_execution`
- `tests/integration/database/test_migration_integration.py::TestMigrationIntegration::test_real_migration_execution - alembic.util.exc.CommandError: Multiple head revisions are present for given argument 'head'; please specify a specific target revision, '<branchname>@head' to narrow to a specific head, or 'heads' for all heads`
- `tests/integration/database/test_migration_integration.py::TestMigrationIntegration::test_rollback_functionality`
- `tests/integration/database/test_migration_integration.py::TestMigrationIntegration::test_rollback_functionality - alembic.util.exc.CommandError: Multiple head revisions are present for given argument 'head'; please specify a specific target revision, '<branchname>@head' to narrow to a specific head, or 'heads' for all heads`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorEdgeCases::test_validate_empty_requirements`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorEdgeCases::test_validator_with_auto_discovered_config`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorEdgeCases::test_validator_with_auto_discovered_config - alembic.util.exc.CommandError: The script directory has multiple heads (due to branching).Please use get_heads(), or merge the branches using alembic merge.`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_get_all_revisions`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_get_all_revisions - alembic.util.exc.CommandError: The script directory has multiple heads (due to branching).Please use get_heads(), or merge the branches using alembic merge.`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_get_current_revision`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_get_current_revision - alembic.util.exc.CommandError: Multiple head revisions are present for given argument 'head'; please specify a specific target revision, '<branchname>@head' to narrow to a specific head, or 'heads' for all heads`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_is_at_head`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_is_at_head - alembic.util.exc.CommandError: The script directory has multiple heads (due to branching).Please use get_heads(), or merge the branches using alembic merge.`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_multiple_validators_same_database`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_multiple_validators_same_database - alembic.util.exc.CommandError: Multiple head revisions are present for given argument 'head'; please specify a specific target revision, '<branchname>@head' to narrow to a specific head, or 'heads' for all heads`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_after_downgrade`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_after_downgrade - alembic.util.exc.CommandError: Multiple head revisions are present for given argument 'head'; please specify a specific target revision, '<branchname>@head' to narrow to a specific head, or 'heads' for all heads`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_nonexistent_migration`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_nonexistent_migration - alembic.util.exc.CommandError: Multiple head revisions are present for given argument 'head'; please specify a specific target revision, '<branchname>@head' to narrow to a specific head, or 'heads' for all heads`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_or_raise_success`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_or_raise_success - alembic.util.exc.CommandError: Multiple head revisions are present for given argument 'head'; please specify a specific target revision, '<branchname>@head' to narrow to a specific head, or 'heads' for all heads`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_single_migration`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_single_migration - alembic.util.exc.CommandError: Multiple head revisions are present for given argument 'head'; please specify a specific target revision, '<branchname>@head' to narrow to a specific head, or 'heads' for all heads`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_with_all_migrations`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_with_all_migrations - alembic.util.exc.CommandError: Multiple head revisions are present for given argument 'head'; please specify a specific target revision, '<branchname>@head' to narrow to a specific head, or 'heads' for all heads`
- `tests/integration/gui/test_end_to_end_workflow.py::TestStrategyLoading::test_strategy_loading_workflow`
- `tests/integration/gui/test_gui_launch.py::TestGUILaunch::test_gui_window_properties`
- `tests/integration/registry/test_registry_integration.py::TestOpenAIServices::test_openai_llm_instantiation`
- `tests/integration/services/test_service_implementations.py::TestDatabaseServices::test_neo4j_graph_service_basic_functionality`
- `tests/unit/database/test_migrations.py::TestAlembicMigrations::test_get_current_version`
- `tests/unit/database/test_migrations.py::TestAlembicMigrations::test_get_current_version - alembic.util.exc.CommandError: Multiple head revisions are present for given argument 'head'; please specify a specific target revision, '<branchname>@head' to narrow to a specific head, or 'heads' for all heads`
- `tests/unit/database/test_migrations.py::TestAlembicMigrations::test_migration_creates_indexes`
- `tests/unit/database/test_migrations.py::TestAlembicMigrations::test_migration_creates_tables`
- `tests/unit/database/test_migrations.py::TestAlembicMigrations::test_migration_creates_tables - alembic.util.exc.CommandError: Multiple head revisions are present for given argument 'head'; please specify a specific target revision, '<branchname>@head' to narrow to a specific head, or 'heads' for all heads`
- `tests/unit/database/test_migrations.py::TestAlembicMigrations::test_migration_downgrade`
- `tests/unit/database/test_migrations.py::TestAlembicMigrations::test_migration_downgrade - alembic.util.exc.CommandError: Multiple head revisions are present for given argument 'head'; please specify a specific target revision, '<branchname>@head' to narrow to a specific head, or 'heads' for all heads`
- `tests/unit/database/test_migrations.py::TestAlembicMigrations::test_migration_idempotency`
- `tests/unit/database/test_migrations.py::TestAlembicMigrations::test_migration_idempotency - alembic.util.exc.CommandError: Multiple head revisions are present for given argument 'head'; please specify a specific target revision, '<branchname>@head' to narrow to a specific head, or 'heads' for all heads`
- `tests/unit/database/test_migrations.py::TestAlembicMigrations::test_migration_upgrade_to_head`
- `tests/unit/database/test_migrations.py::TestAlembicMigrations::test_migration_upgrade_to_head - alembic.util.exc.CommandError: Multiple head revisions are present for given argument 'head'; please specify a specific target revision, '<branchname>@head' to narrow to a specific head, or 'heads' for all heads`
- `tests/unit/documentation/test_code_examples.py::TestCodeExamples::test_all_code_examples_have_valid_syntax`
- `tests/unit/documentation/test_links.py::TestDocumentationLinks::test_no_broken_internal_links`
- `tests/unit/documentation/test_links.py::TestDocumentationLinks::test_no_todo_links`
- `tests/unit/gui/test_main_window.py::TestButtonStates::test_button_state_with_strategy_no_text`
- `tests/unit/gui/test_main_window.py::TestButtonStates::test_retrieve_button_state_with_query`
- `tests/unit/gui/test_main_window.py::TestMainWindowCreation::test_window_geometry`
- `tests/unit/gui/test_main_window.py::TestMainWindowCreation::test_window_geometry - AssertionError: assert '900x800' in '1200x800+360+53'`
- `tests/unit/gui/test_main_window.py::TestMainWindowCreation::test_window_minimum_size`
- `tests/unit/gui/test_main_window.py::TestMainWindowCreation::test_window_minimum_size - assert 900 == 800`
- `tests/unit/gui/test_main_window.py::TestUtilityMethods::test_show_help_displays_message`
- `tests/unit/gui/test_main_window.py::TestWidgetCreation::test_all_buttons_exist`
- `tests/unit/gui/test_retrieval_operations.py::TestQueryExecution::test_retrieve_button_disabled_during_operation`
- `tests/unit/gui/test_utility_operations.py::TestClearAllData::test_clear_data_error_handling`
- `tests/unit/registry/test_service_factory.py::TestLLMServiceCreation::test_create_llm_service_lm_studio`
- `tests/unit/registry/test_service_factory.py::TestLLMServiceCreation::test_create_llm_service_lm_studio - Failed: DID NOT RAISE <class 'rag_factory.registry.exceptions.ServiceInstantiationError'>`
- `tests/unit/registry/test_service_factory.py::TestLLMServiceCreation::test_create_llm_service_openai`
- `tests/unit/registry/test_service_factory.py::TestLLMServiceCreation::test_create_llm_service_openai - AssertionError: assert <rag_factory.services.llm.service.LLMService object at 0x7a6006ec3a10> is <Mock name='OpenAILLMService()' id='134552847700224'>`
- `tests/unit/registry/test_service_factory.py::TestLLMServiceCreation::test_create_llm_service_with_defaults`
- `tests/unit/services/database/test_migration_validator.py::TestMigrationValidator::test_get_current_revision_success`
- `tests/unit/services/database/test_migration_validator.py::TestMigrationValidator::test_get_current_revision_success - TypeError: 'Mock' object is not subscriptable`
- `tests/unit/services/database/test_migration_validator.py::TestMigrationValidator::test_validate_all_migrations_applied`
- `tests/unit/services/database/test_migration_validator.py::TestMigrationValidator::test_validate_all_migrations_applied - assert False is True`
- `tests/unit/services/database/test_migration_validator.py::TestMigrationValidator::test_validate_missing_migrations`
- `tests/unit/services/database/test_migration_validator.py::TestMigrationValidator::test_validate_or_raise_failure`
- `tests/unit/services/database/test_migration_validator.py::TestMigrationValidator::test_validate_or_raise_success`
- `tests/unit/services/database/test_migration_validator.py::TestMigrationValidatorIntegration::test_full_validation_workflow`

</details>

---

## 📝 Configuration vs Implementation Breakdown

### Tests Failing Due to Configuration Issues
These tests require proper environment setup (`.env` file, database connections, API keys):


**Database Migration - Multiple Heads:**
- `tests/integration/database/test_migration_integration.py`
- `tests/integration/database/test_migration_validator_integration.py`
- `tests/unit/database/test_migrations.py`

**Neo4j Connection:**
- `tests/integration_real/test_neo4j_real.py`


### Tests Failing Due to Implementation/Requirement Issues
These tests require code fixes or test logic updates:


**Import/Module Errors:**
- `tests/unit/gui/test_backend_integration.py`

**GUI/Tkinter Issues:**
- `tests/unit/gui/test_main_window.py`
- `tests/unit/gui/test_retrieval_operations.py`
- `tests/unit/gui/test_utility_operations.py`

**Documentation Errors:**
- `tests/unit/documentation/test_code_examples.py`
- `tests/unit/documentation/test_links.py`

**Mock/Test Configuration:**
- `tests/test_mock_registry.py`
- `tests/unit/services/database/test_migration_validator.py`

**Type/Attribute Errors:**
- `tests/integration/test_hybrid_search_pair.py`

**Assertion Failures:**
- `tests/unit/registry/test_service_factory.py`


---

## 💡 Quick Fixes

### For Database Migration Issues:
```bash
# Merge multiple migration heads
alembic merge heads

# Apply all migrations
alembic upgrade head
```

### For Neo4j Connection Issues:
Add to `.env` file:
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### For Import Errors:
```bash
# Install missing dependencies
pip install -r requirements.txt

# Or install specific missing modules
pip install <module_name>
```

---

## 📈 Test Health Metrics

- **Overall Pass Rate:** 93.6% (2,145/2,291 tests)
- **File Pass Rate:** 90.3% (186/206 files)
- **Critical Issues:** {len(high_items)} (High Priority)
- **Non-Critical Issues:** {len(medium_items)} (Medium Priority)

