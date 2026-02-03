"""
pytest configuration for DGSF adapter tests.

Handles import path setup for adapter package.
"""

import sys
from pathlib import Path

# Add projects/dgsf to path so 'adapter' is importable as a package
dgsf_root = Path(__file__).parent.parent.parent
if str(dgsf_root) not in sys.path:
    sys.path.insert(0, str(dgsf_root))
