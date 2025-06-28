"""
scripts/__init__.py
Safeguards module access.
"""
import os
import sys
import warnings

__all__ = []

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__)).rsplit(os.sep, 1)[0]
CALLER_PATH = os.path.abspath(sys.argv[0])

if not CALLER_PATH.startswith(PROJECT_ROOT):
    warnings.warn("scripts package accessed from outside the project directory. Check for unsafe access.")