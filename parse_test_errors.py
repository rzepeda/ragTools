#!/usr/bin/env python3
"""
Parse test results and categorize errors by type
"""
import re
from collections import defaultdict
from pathlib import Path

def parse_test_results(results_file):
    """Parse test results and extract error information"""
    
    with open(results_file, 'r') as f:
        content = f.read()
    
    # Split by test file sections
    sections = re.split(r'\[\d+/\d+\]', content)
    
    errors_by_category = defaultdict(list)
    errors_by_file = defaultdict(list)
    failing_tests = []
    failing_files = set()
    
    current_file = None
    
    for section in sections:
        # Extract file name
        file_match = re.search(r'(tests/[^\s]+\.py)', section)
        if file_match:
            current_file = file_match.group(1)
        
        # Check if section has failures
        if 'FAILED:' in section and current_file:
            failing_files.add(current_file)
            
            # Extract individual test failures
            test_failures = re.findall(r'(tests/[^\s]+\.py)::([\w:]+)\s+FAILED', section)
            for test_file, test_name in test_failures:
                failing_tests.append(f"{test_file}::{test_name}")
            
            # Categorize errors
            if 'ModuleNotFoundError' in section or 'ImportError' in section:
                error_match = re.search(r'(ModuleNotFoundError|ImportError): (.+?)(?:\n|$)', section)
                if error_match:
                    error_msg = error_match.group(0).strip()
                    errors_by_category['Import Errors'].append({
                        'file': current_file,
                        'error': error_msg,
                        'type': 'configuration'
                    })
                    errors_by_file[current_file].append(('Import Error', error_msg))
            
            if 'FileNotFoundError' in section:
                error_match = re.search(r'FileNotFoundError: (.+?)(?:\n|$)', section)
                if error_match:
                    error_msg = error_match.group(0).strip()
                    if 'onnx' in error_msg.lower() or 'model' in error_msg.lower():
                        errors_by_category['Missing Model Files'].append({
                            'file': current_file,
                            'error': error_msg,
                            'type': 'configuration'
                        })
                    else:
                        errors_by_category['File Not Found'].append({
                            'file': current_file,
                            'error': error_msg,
                            'type': 'configuration'
                        })
                    errors_by_file[current_file].append(('File Not Found', error_msg))
            
            if 'psycopg2.OperationalError' in section or 'could not connect to server' in section:
                error_match = re.search(r'psycopg2\.OperationalError: (.+?)(?:\n|$)', section)
                if error_match:
                    error_msg = error_match.group(0).strip()
                    errors_by_category['Database Connection'].append({
                        'file': current_file,
                        'error': error_msg,
                        'type': 'configuration'
                    })
                    errors_by_file[current_file].append(('Database Connection', error_msg))
            
            if 'DependentObjectsStillExist' in section or 'cannot drop extension vector' in section:
                errors_by_category['Database Migration - Vector Extension'].append({
                    'file': current_file,
                    'error': 'cannot drop extension vector because other objects depend on it',
                    'type': 'requirement'
                })
                errors_by_file[current_file].append(('Migration Error', 'Vector extension dependency'))
            
            if 'UndefinedTable' in section and 'relation "chunks" does not exist' in section:
                errors_by_category['Database Migration - Missing Tables'].append({
                    'file': current_file,
                    'error': 'relation "chunks" does not exist',
                    'type': 'requirement'
                })
                errors_by_file[current_file].append(('Migration Error', 'Missing chunks table'))
            
            if 'AttributeError' in section:
                error_match = re.search(r"AttributeError: (.+?)(?:\n|$)", section)
                if error_match:
                    error_msg = error_match.group(0).strip()
                    errors_by_category['Attribute Errors'].append({
                        'file': current_file,
                        'error': error_msg,
                        'type': 'code'
                    })
                    errors_by_file[current_file].append(('Attribute Error', error_msg))
            
            if 'TypeError' in section:
                error_match = re.search(r"TypeError: (.+?)(?:\n|$)", section)
                if error_match:
                    error_msg = error_match.group(0).strip()
                    errors_by_category['Type Errors'].append({
                        'file': current_file,
                        'error': error_msg,
                        'type': 'code'
                    })
                    errors_by_file[current_file].append(('Type Error', error_msg))
            
            if 'AssertionError' in section:
                error_match = re.search(r"AssertionError: (.+?)(?:\n|$)", section)
                if error_match:
                    error_msg = error_match.group(0).strip()
                    errors_by_category['Assertion Errors'].append({
                        'file': current_file,
                        'error': error_msg,
                        'type': 'code'
                    })
                    errors_by_file[current_file].append(('Assertion Error', error_msg))
            
            if 'KeyError' in section:
                error_match = re.search(r"KeyError: (.+?)(?:\n|$)", section)
                if error_match:
                    error_msg = error_match.group(0).strip()
                    errors_by_category['Key Errors'].append({
                        'file': current_file,
                        'error': error_msg,
                        'type': 'code'
                    })
                    errors_by_file[current_file].append(('Key Error', error_msg))
            
            if 'ValueError' in section:
                error_match = re.search(r"ValueError: (.+?)(?:\n|$)", section)
                if error_match:
                    error_msg = error_match.group(0).strip()
                    if 'API' in error_msg or 'key' in error_msg.lower():
                        errors_by_category['API Configuration'].append({
                            'file': current_file,
                            'error': error_msg,
                            'type': 'configuration'
                        })
                    else:
                        errors_by_category['Value Errors'].append({
                            'file': current_file,
                            'error': error_msg,
                            'type': 'code'
                        })
                    errors_by_file[current_file].append(('Value Error', error_msg))
    
    return errors_by_category, errors_by_file, list(failing_files), failing_tests

def main():
    results_file = Path('tests/test_results_by_file.txt')
    
    errors_by_category, errors_by_file, failing_files, failing_tests = parse_test_results(results_file)
    
    # Print summary
    print("=" * 80)
    print("ERROR CATEGORIZATION SUMMARY")
    print("=" * 80)
    print()
    
    # Configuration Issues
    print("## CONFIGURATION ISSUES")
    print()
    config_categories = ['Database Connection', 'API Configuration', 'Missing Model Files', 
                        'File Not Found', 'Import Errors']
    for category in config_categories:
        if category in errors_by_category:
            print(f"### {category} ({len(errors_by_category[category])} occurrences)")
            files = set(e['file'] for e in errors_by_category[category])
            for f in sorted(files):
                print(f"  - {f}")
            print()
    
    # Requirement Issues  
    print("## REQUIREMENT ISSUES")
    print()
    req_categories = ['Database Migration - Vector Extension', 'Database Migration - Missing Tables']
    for category in req_categories:
        if category in errors_by_category:
            print(f"### {category} ({len(errors_by_category[category])} occurrences)")
            files = set(e['file'] for e in errors_by_category[category])
            for f in sorted(files):
                print(f"  - {f}")
            print()
    
    # Code Issues
    print("## CODE ISSUES")
    print()
    code_categories = ['Attribute Errors', 'Type Errors', 'Assertion Errors', 
                      'Key Errors', 'Value Errors']
    for category in code_categories:
        if category in errors_by_category:
            print(f"### {category} ({len(errors_by_category[category])} occurrences)")
            files = set(e['file'] for e in errors_by_category[category])
            for f in sorted(files):
                print(f"  - {f}")
            print()
    
    # Failing files
    print("## FAILING TEST FILES")
    print()
    for f in sorted(failing_files):
        print(f"- {f}")
    print()
    
    # Failing tests
    print("## FAILING TESTS (Individual)")
    print()
    for test in sorted(failing_tests)[:50]:  # Limit to first 50
        print(f"- {test}")
    if len(failing_tests) > 50:
        print(f"... and {len(failing_tests) - 50} more")

if __name__ == '__main__':
    main()
