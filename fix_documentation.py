#!/usr/bin/env python3
"""
Script to fix common documentation issues.

This script:
1. Comments out broken internal links
2. Wraps async code examples in async functions
3. Fixes common indentation issues in code blocks
"""

import re
from pathlib import Path
from typing import List, Tuple


def fix_broken_links(content: str, md_file: Path, docs_root: Path) -> str:
    """Remove broken internal links, keeping only the text."""
    
    def check_link(match):
        text = match.group(1)
        link = match.group(2)
        
        # Skip external links
        if link.startswith("http"):
            return match.group(0)
        
        # Skip anchors
        if link.startswith("#"):
            return match.group(0)
        
        # Skip file:// links
        if link.startswith("file://"):
            return match.group(0)
        
        # Check if target exists
        target = (md_file.parent / link).resolve()
        
        if not target.exists():
            # Remove link syntax, keep only the text
            # Add a comment before the text to indicate it was a broken link
            return f"{text} <!-- (broken link to: {link}) -->"
        
        return match.group(0)
    
    # Find and fix [text](link) patterns
    link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
    fixed_content = re.sub(link_pattern, check_link, content)
    
    return fixed_content


def fix_code_blocks(content: str) -> str:
    """Fix common code block issues."""
    
    def fix_python_block(match):
        code = match.group(1)
        original_code = code
        
        # Try to compile first to see if it's already valid
        try:
            compile(code, 'test', 'exec')
            return match.group(0)  # Already valid
        except SyntaxError:
            pass  # Continue with fixes
        
        # Check if code has 'await' outside function
        if 'await ' in code and 'async def' not in code and 'def ' not in code:
            # Wrap in async function
            lines = code.split('\n')
            indented_lines = ['    ' + line if line.strip() else line for line in lines]
            wrapped_code = 'async def example():\n' + '\n'.join(indented_lines)
            try:
                compile(wrapped_code, 'test', 'exec')
                return f'```python\n{wrapped_code}\n```'
            except SyntaxError:
                pass  # Try other fixes
        
        # Check for unexpected indent at start (common in copied code)
        lines = code.split('\n')
        if lines and lines[0] and lines[0][0].isspace():
            # Remove leading indentation from all lines
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                min_indent = min(len(line) - len(line.lstrip()) 
                               for line in non_empty_lines)
                fixed_lines = [line[min_indent:] if len(line) > min_indent else line 
                              for line in lines]
                fixed_code = '\n'.join(fixed_lines)
                try:
                    compile(fixed_code, 'test', 'exec')
                    return f'```python\n{fixed_code}\n```'
                except SyntaxError:
                    pass  # Try other fixes
        
        # If code starts with invalid syntax (like a comment or incomplete statement),
        # try wrapping it in a function or commenting it out
        if code.strip().startswith(('"""', "'''", '#', '//')):
            # It's likely documentation or a comment, leave as is
            return match.group(0)
        
        # Check for unterminated strings - comment out the problematic line
        if 'unterminated' in str(compile.__code__):
            # Can't easily fix unterminated strings, leave as is
            return match.group(0)
        
        # If all else fails, return original
        return match.group(0)
    
    # Fix Python code blocks
    python_block_pattern = r'```python\n(.*?)```'
    fixed_content = re.sub(python_block_pattern, fix_python_block, content, flags=re.DOTALL)
    
    return fixed_content


def process_markdown_file(md_file: Path, docs_root: Path) -> Tuple[bool, str]:
    """Process a single markdown file and return if it was modified."""
    
    try:
        with open(md_file, 'r', encoding='utf-8', errors='replace') as f:
            original_content = f.read()
        
        # Apply fixes
        content = original_content
        content = fix_broken_links(content, md_file, docs_root)
        content = fix_code_blocks(content)
        
        # Check if content changed
        if content != original_content:
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, f"Fixed {md_file.relative_to(docs_root)}"
        
        return False, ""
    
    except Exception as e:
        return False, f"Error processing {md_file}: {e}"


def main():
    """Main function to process all markdown files."""
    
    # Get docs root
    script_dir = Path(__file__).parent
    docs_root = script_dir / "docs"
    
    if not docs_root.exists():
        print(f"Error: docs directory not found at {docs_root}")
        return
    
    print(f"Processing markdown files in {docs_root}...")
    print("=" * 60)
    
    modified_files = []
    errors = []
    
    # Process all markdown files
    for md_file in docs_root.rglob("*.md"):
        modified, message = process_markdown_file(md_file, docs_root)
        
        if modified:
            modified_files.append(message)
        elif message:  # Error message
            errors.append(message)
    
    # Print results
    print(f"\nProcessed {len(list(docs_root.rglob('*.md')))} markdown files")
    print(f"Modified: {len(modified_files)} files")
    
    if modified_files:
        print("\nModified files:")
        for msg in modified_files[:20]:  # Show first 20
            print(f"  ✓ {msg}")
        if len(modified_files) > 20:
            print(f"  ... and {len(modified_files) - 20} more")
    
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  ✗ {error}")
    
    print("\n" + "=" * 60)
    print("Documentation fixes complete!")
    print("\nNote: Broken links have been commented out.")
    print("You may want to create the missing files or update the links.")


if __name__ == "__main__":
    main()
