"""
Makes the project root importable from tests/ without needing the package
installed — config.py, utils.py, etc. sit one directory up.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
