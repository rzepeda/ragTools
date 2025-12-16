"""Environment variable validation and management for RAG Factory.

This module provides utilities for validating environment variables,
managing deprecated variable names, and ensuring backward compatibility.
"""

import os
import warnings
from typing import Dict, List, Optional


class EnvironmentValidator:
    """Validates and manages environment variables.
    
    This class provides validation for required environment variables,
    deprecation warnings for old variable names, and backward compatibility
    helpers for transitioning to new variable naming standards.
    """
    
    # Required variables for different modes
    REQUIRED_VARS: Dict[str, List[str]] = {
        'production': ['DATABASE_URL'],
        'development': ['DATABASE_URL'],
        'test': ['TEST_DATABASE_URL'],
    }
    
    # Deprecated variables and their replacements
    DEPRECATED_VARS: Dict[str, str] = {
        'DATABASE_TEST_URL': 'TEST_DATABASE_URL',
    }
    
    @classmethod
    def validate(cls, mode: str = 'development') -> None:
        """Validate environment variables for the given mode.
        
        Args:
            mode: Environment mode (production, development, test)
            
        Raises:
            ValueError: If required variables are missing
            
        Example:
            >>> EnvironmentValidator.validate(mode='production')
            >>> EnvironmentValidator.validate(mode='test')
        """
        missing = cls._check_required(mode)
        if missing:
            raise ValueError(
                f"Missing required environment variables for {mode} mode: {', '.join(missing)}\n"
                f"Please set these variables in your .env file or environment.\n"
                f"See docs/database/ENVIRONMENT_VARIABLES.md for more information."
            )
        
        cls._check_deprecated()
    
    @classmethod
    def _check_required(cls, mode: str) -> List[str]:
        """Check for missing required variables.
        
        Args:
            mode: Environment mode to check
            
        Returns:
            List of missing required variable names
        """
        required = cls.REQUIRED_VARS.get(mode, [])
        missing = []
        
        for var in required:
            # Check if variable is set
            if not os.getenv(var):
                # For test database, also check deprecated name
                if var == 'TEST_DATABASE_URL' and os.getenv('DATABASE_TEST_URL'):
                    continue  # Old name is set, will warn in _check_deprecated
                missing.append(var)
        
        return missing
    
    @classmethod
    def _check_deprecated(cls) -> None:
        """Warn about deprecated variables.
        
        Issues DeprecationWarning for any deprecated environment variables
        that are currently set.
        """
        for old_var, new_var in cls.DEPRECATED_VARS.items():
            if os.getenv(old_var):
                warnings.warn(
                    f"Environment variable '{old_var}' is deprecated. "
                    f"Please use '{new_var}' instead. "
                    f"Support for '{old_var}' will be removed in a future version.\n"
                    f"Migration: Rename {old_var}={new_var} in your .env file.",
                    DeprecationWarning,
                    stacklevel=3
                )
    
    @classmethod
    def get_database_url(cls, for_tests: bool = False) -> Optional[str]:
        """Get database URL with backward compatibility.
        
        This method provides backward compatibility by checking both old
        and new variable names, with the new name taking precedence.
        
        Args:
            for_tests: If True, get test database URL; otherwise get main database URL
            
        Returns:
            Database URL or None if not set
            
        Example:
            >>> url = EnvironmentValidator.get_database_url(for_tests=False)
            >>> test_url = EnvironmentValidator.get_database_url(for_tests=True)
        """
        if for_tests:
            # Try new name first, fall back to old name
            url = os.getenv('TEST_DATABASE_URL') or os.getenv('DATABASE_TEST_URL')
            if os.getenv('DATABASE_TEST_URL') and not os.getenv('TEST_DATABASE_URL'):
                warnings.warn(
                    "Using deprecated 'DATABASE_TEST_URL'. Please rename to 'TEST_DATABASE_URL'.\n"
                    "Update your .env file: DATABASE_TEST_URL → TEST_DATABASE_URL",
                    DeprecationWarning,
                    stacklevel=2
                )
            return url
        else:
            return os.getenv('DATABASE_URL')
    
    @classmethod
    def get_all_database_vars(cls) -> Dict[str, Optional[str]]:
        """Get all database-related environment variables.
        
        Returns:
            Dictionary mapping variable names to their values
            
        Example:
            >>> vars = EnvironmentValidator.get_all_database_vars()
            >>> print(vars['DATABASE_URL'])
        """
        return {
            'DATABASE_URL': os.getenv('DATABASE_URL'),
            'TEST_DATABASE_URL': cls.get_database_url(for_tests=True),
            'DB_DATABASE_URL': os.getenv('DB_DATABASE_URL'),
            'DB_TEST_DATABASE_URL': os.getenv('DB_TEST_DATABASE_URL'),
        }
