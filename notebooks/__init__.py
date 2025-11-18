"""
Notebook utilities for CrystaLLMv2_PKV.
Import this module at the start of any notebook to automatically navigate to package root.
"""
import os
import sys
from pathlib import Path

def setup_notebook_environment():
    """
    Automatically navigate to package root and set up Python path.
    Call this function at the start of any notebook in the notebooks/ folder.
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