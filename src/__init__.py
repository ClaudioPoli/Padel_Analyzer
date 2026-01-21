"""
Padel Analyzer - AI-based tool to analyze padel matches.

This package provides functionality to:
- Load and process padel match videos
- Track players' movements
- Track ball position
- Detect and identify the padel field
"""

__version__ = "0.1.0"
__author__ = "Padel Analyzer Team"

from .analyzer import PadelAnalyzer

__all__ = ["PadelAnalyzer"]
