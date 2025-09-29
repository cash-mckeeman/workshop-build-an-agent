"""
Entry point for the refactored Streamlit application.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the main app
from streamlit_app.app import main

if __name__ == "__main__":
    main()