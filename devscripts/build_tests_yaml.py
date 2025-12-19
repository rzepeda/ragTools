import os
import sys
import ast
import yaml
import glob
import shutil
import argparse
from pathlib import Path

# Configuration
SOURCE_DIR = "rag_factory"  # UPDATED: Pointing to your actual code folder
TEST_DIR = "tests"
MIRROR_DIR = "tests/mirror"
TEMPLATE_PATH = "templates/test_config_scaffold.yaml"

def find_imports(file_path):
    """
    Parses a python file and returns a list of imports.
    Returns: set(['rag_factory.auth.login', 'tests.mocks.db'])
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports

def map_tests_to_source():
    """
    Scans all files in tests/ and builds a reverse map:
    rag_factory/module.py -> { 'tests': [], 'mocks': [] }
    """
    mapping = {}
    
    # scan all python files in tests/
    test_files = glob.glob(f"{TEST_DIR}/**/*.py", recursive=True)
    
    for test_file in test_files:
        if "mirror" in test_file: continue # Skip the registry itself
        
        # simple heuristic: if it's in tests/mocks, it's a mock
        is_mock = "mocks" in test_file
        
        imported_modules = find_imports(test_file)
        
        for module in imported_modules:
            # We only care if it imports something from the source directory
            if module.startswith(f"{SOURCE_DIR}."):
                # Convert module path (rag_factory.auth.login) to file path (rag_factory/auth/login.py)
                # This implies strict package-to-folder mapping
                src_file_guess = module.replace(".", "/") + ".py"
                
                if os.path.exists(src_file_guess):
                    if src_file_guess not in mapping:
                        mapping[src_file_guess] = {"tests": set(), "mocks": set()}
                    
                    if is_mock:
                        mapping[src_file_guess]["mocks"].add(test_file)
                    else:
                        mapping[src_file_guess]["tests"].add(test_file)
                        
    return mapping

def generate_manifests(target_src_file, data, scaffold=False):
    """
    Writes the .auto.yaml and optionally the .config.yaml
    """
    # Calculate mirror path: rag_factory/auth/login.py -> tests/mirror/rag_factory/auth/login
    try:
        relative_path = os.path.relpath(target_src_file, SOURCE_DIR)
    except ValueError:
        # Fallback if path is weird or absolute
        relative_path = os.path.basename(target_src_file)

    # We preserve the SOURCE_DIR name in the mirror path to avoid collisions
    mirror_base = Path(MIRROR_DIR) / SOURCE_DIR / relative_path
    
    # Ensure the directory exists (e.g. tests/mirror/rag_factory/auth/)
    mirror_base.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Write AUTO yaml
    # We strip the .py extension for the yaml filename usually, or keep it. 
    # Current logic: src/file.py -> tests/mirror/src/file.py.auto.yaml
    auto_file = str(mirror_base) + ".auto.yaml"
    
    auto_data = {
        "header": "[AUTO-GENERATED] DO NOT EDIT",
        "target": target_src_file,
        "test_files": sorted(list(data["tests"])),
        "mocks": sorted(list(data["mocks"]))
    }
    
    with open(auto_file, "w") as f:
        yaml.dump(auto_data, f, sort_keys=False)
    print(f"Updated: {auto_file}")

    # 2. Write CONFIG yaml (Scaffold)
    config_file = str(mirror_base) + ".config.yaml"
    if scaffold:
        if not os.path.exists(config_file):
            if os.path.exists(TEMPLATE_PATH):
                shutil.copy(TEMPLATE_PATH, config_file)
                print(f"Scaffolded: {config_file}")
            else:
                # If template is missing, just create a blank placeholder or warning
                print(f"Error: Template not found at {TEMPLATE_PATH}")
        else:
            print(f"Skipped: {config_file} (Already exists)")

def main():
    parser = argparse.ArgumentParser(description="Build Test Manifests")
    parser.add_argument("path", help="Path to source file or directory to scan (e.g., rag_factory/)")
    parser.add_argument("--scaffold", action="store_true", help="Generate config.yaml scaffold if missing")
    args = parser.parse_args()

    # 1. Build the global map of Source -> Tests
    print("Scanning tests dependencies...")
    full_map = map_tests_to_source()
    
    # 2. Filter for the requested target
    targets = []
    if os.path.isdir(args.path):
        # Recursive processing
        for f in glob.glob(f"{args.path}/**/*.py", recursive=True):
            norm_path = str(Path(f))
            targets.append(norm_path)
    else:
        targets = [str(Path(args.path))]

    # 3. Generate
    processed_count = 0
    for target in targets:
        # Normalize paths
        normalized_target = str(Path(target))
        
        # Only process files that are actually inside our SOURCE_DIR
        if not normalized_target.startswith(SOURCE_DIR):
            continue

        if normalized_target in full_map:
            generate_manifests(normalized_target, full_map[normalized_target], args.scaffold)
            processed_count += 1
        elif os.path.exists(normalized_target):
            # Even if no tests found, we generate empty manifests
            generate_manifests(normalized_target, {"tests": [], "mocks": []}, args.scaffold)
            processed_count += 1
            
    print(f"Processed {processed_count} files.")

if __name__ == "__main__":
    main()