"""Provide notebook helpers for CrystaLLM-pi.

Importing ``__init__`` at the start of a notebook changes the working directory
to the package root so relative paths resolve consistently with the CLIs.
"""
import os
import sys
from pathlib import Path

def setup_notebook_environment():
    """Automatically navigate to package root and set up Python path. Call this function at the start of
any notebook in the notebooks/ folder.
    """
    current_dir = Path.cwd()
    
    # Navigate to package root if we're in notebooks folder
    if current_dir.name == 'notebooks':
        package_root = current_dir.parent
        os.chdir(str(package_root))
        print(f"Navigated to package root")
        # Add to Python path
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
            
    else:
        print(f"Current directory: {current_dir}")

# Auto-run when imported
setup_notebook_environment()