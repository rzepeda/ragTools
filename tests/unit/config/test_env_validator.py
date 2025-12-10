"""Unit tests for environment variable validation."""

import os
import pytest
import warnings
from unittest.mock import patch

from rag_factory.database.env_validator import EnvironmentValidator


class TestEnvironmentValidator:
    """Test environment variable validation."""
    
    def test_validate_production_success(self, monkeypatch):
        """Test validation passes with required variables for production."""
        monkeypatch.setenv('DATABASE_URL', 'postgresql://localhost/db')
        # Should not raise
        EnvironmentValidator.validate(mode='production')
    
    def test_validate_development_success(self, monkeypatch):
        """Test validation passes with required variables for development."""
        monkeypatch.setenv('DATABASE_URL', 'postgresql://localhost/db')
        # Should not raise
        EnvironmentValidator.validate(mode='development')
    
    def test_validate_test_success(self, monkeypatch):
        """Test validation passes with required variables for test mode."""
        monkeypatch.setenv('TEST_DATABASE_URL', 'postgresql://localhost/test')
        # Should not raise
        EnvironmentValidator.validate(mode='test')
    
    def test_validate_production_missing(self, monkeypatch):
        """Test validation fails with missing variables for production."""
        monkeypatch.delenv('DATABASE_URL', raising=False)
        
        with pytest.raises(ValueError) as exc_info:
            EnvironmentValidator.validate(mode='production')
        
        assert 'Missing required environment variables' in str(exc_info.value)
        assert 'DATABASE_URL' in str(exc_info.value)
    
    def test_validate_test_missing(self, monkeypatch):
        """Test validation fails with missing test database URL."""
        monkeypatch.delenv('TEST_DATABASE_URL', raising=False)
        monkeypatch.delenv('DATABASE_TEST_URL', raising=False)
        
        with pytest.raises(ValueError) as exc_info:
            EnvironmentValidator.validate(mode='test')
        
        assert 'Missing required environment variables' in str(exc_info.value)
        assert 'TEST_DATABASE_URL' in str(exc_info.value)
    
    def test_validate_test_with_deprecated_var(self, monkeypatch):
        """Test validation passes with deprecated variable name."""
        monkeypatch.delenv('TEST_DATABASE_URL', raising=False)
        monkeypatch.setenv('DATABASE_TEST_URL', 'postgresql://localhost/test')
        
        # Should not raise - deprecated var is accepted
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            EnvironmentValidator.validate(mode='test')
            
            # Should have deprecation warning
            assert len(w) > 0
            assert issubclass(w[0].category, DeprecationWarning)
            assert 'DATABASE_TEST_URL' in str(w[0].message)
    
    def test_deprecated_warning(self, monkeypatch):
        """Test deprecation warning for old variable names."""
        monkeypatch.setenv('DATABASE_TEST_URL', 'postgresql://localhost/test')
        
        with pytest.warns(DeprecationWarning, match="DATABASE_TEST_URL.*deprecated"):
            EnvironmentValidator._check_deprecated()
    
    def test_no_deprecated_warning_when_not_set(self, monkeypatch):
        """Test no warning when deprecated variables are not set."""
        monkeypatch.delenv('DATABASE_TEST_URL', raising=False)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            EnvironmentValidator._check_deprecated()
            
            # Should have no warnings
            assert len(w) == 0
    
    def test_get_database_url_main(self, monkeypatch):
        """Test getting main database URL."""
        monkeypatch.setenv('DATABASE_URL', 'postgresql://localhost/main')
        
        url = EnvironmentValidator.get_database_url(for_tests=False)
        assert url == 'postgresql://localhost/main'
    
    def test_get_database_url_test_new_name(self, monkeypatch):
        """Test getting test database URL with new variable name."""
        monkeypatch.setenv('TEST_DATABASE_URL', 'postgresql://localhost/test')
        monkeypatch.delenv('DATABASE_TEST_URL', raising=False)
        
        url = EnvironmentValidator.get_database_url(for_tests=True)
        assert url == 'postgresql://localhost/test'
    
    def test_get_database_url_backward_compatible(self, monkeypatch):
        """Test backward compatibility for test database URL."""
        # Old variable name should still work
        monkeypatch.delenv('TEST_DATABASE_URL', raising=False)
        monkeypatch.setenv('DATABASE_TEST_URL', 'postgresql://localhost/old')
        
        with pytest.warns(DeprecationWarning, match="DATABASE_TEST_URL"):
            url = EnvironmentValidator.get_database_url(for_tests=True)
        
        assert url == 'postgresql://localhost/old'
    
    def test_get_database_url_prefers_new_name(self, monkeypatch):
        """Test new variable name takes precedence over old name."""
        monkeypatch.setenv('TEST_DATABASE_URL', 'postgresql://localhost/new')
        monkeypatch.setenv('DATABASE_TEST_URL', 'postgresql://localhost/old')
        
        # Should not warn because new name is set
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            url = EnvironmentValidator.get_database_url(for_tests=True)
            
            # Should use new value
            assert url == 'postgresql://localhost/new'
            
            # Should not have deprecation warning (new name is used)
            deprecation_warnings = [warning for warning in w 
                                   if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0
    
    def test_get_database_url_returns_none_when_not_set(self, monkeypatch):
        """Test returns None when database URL is not set."""
        monkeypatch.delenv('DATABASE_URL', raising=False)
        
        url = EnvironmentValidator.get_database_url(for_tests=False)
        assert url is None
    
    def test_get_database_url_test_returns_none_when_not_set(self, monkeypatch):
        """Test returns None when test database URL is not set."""
        monkeypatch.delenv('TEST_DATABASE_URL', raising=False)
        monkeypatch.delenv('DATABASE_TEST_URL', raising=False)
        
        url = EnvironmentValidator.get_database_url(for_tests=True)
        assert url is None
    
    def test_get_all_database_vars(self, monkeypatch):
        """Test getting all database variables."""
        monkeypatch.setenv('DATABASE_URL', 'postgresql://localhost/main')
        monkeypatch.setenv('TEST_DATABASE_URL', 'postgresql://localhost/test')
        monkeypatch.setenv('DB_DATABASE_URL', 'postgresql://localhost/db_main')
        monkeypatch.setenv('DB_TEST_DATABASE_URL', 'postgresql://localhost/db_test')
        
        vars_dict = EnvironmentValidator.get_all_database_vars()
        
        assert vars_dict['DATABASE_URL'] == 'postgresql://localhost/main'
        assert vars_dict['TEST_DATABASE_URL'] == 'postgresql://localhost/test'
        assert vars_dict['DB_DATABASE_URL'] == 'postgresql://localhost/db_main'
        assert vars_dict['DB_TEST_DATABASE_URL'] == 'postgresql://localhost/db_test'
    
    def test_check_required_returns_empty_for_valid_config(self, monkeypatch):
        """Test _check_required returns empty list when all vars are set."""
        monkeypatch.setenv('DATABASE_URL', 'postgresql://localhost/db')
        
        missing = EnvironmentValidator._check_required('production')
        assert missing == []
    
    def test_check_required_returns_missing_vars(self, monkeypatch):
        """Test _check_required returns list of missing variables."""
        monkeypatch.delenv('DATABASE_URL', raising=False)
        
        missing = EnvironmentValidator._check_required('production')
        assert 'DATABASE_URL' in missing
    
    def test_error_message_includes_documentation_reference(self, monkeypatch):
        """Test error message includes reference to documentation."""
        monkeypatch.delenv('DATABASE_URL', raising=False)
        
        with pytest.raises(ValueError) as exc_info:
            EnvironmentValidator.validate(mode='production')
        
        assert 'ENVIRONMENT_VARIABLES.md' in str(exc_info.value)
    
    def test_deprecation_warning_includes_migration_instructions(self, monkeypatch):
        """Test deprecation warning includes migration instructions."""
        monkeypatch.setenv('DATABASE_TEST_URL', 'postgresql://localhost/test')
        
        with pytest.warns(DeprecationWarning) as warning_list:
            EnvironmentValidator._check_deprecated()
        
        warning_message = str(warning_list[0].message)
        assert 'TEST_DATABASE_URL' in warning_message
        assert 'deprecated' in warning_message.lower()
