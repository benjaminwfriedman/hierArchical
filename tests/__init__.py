"""
hierArchical Testing Suite

This package contains comprehensive unit tests for the hierArchical framework.
All tests are designed to be fast, isolated, and independent of heavy external
dependencies through comprehensive mocking.

Structure:
- unit/: Pure unit tests for individual modules
- property/: Property-based tests using Hypothesis
- fixtures/: Test data and geometry samples
- conftest.py: Shared fixtures and configuration

Usage:
    pytest                    # Run all tests
    pytest -m unit           # Run only unit tests
    pytest -m property       # Run only property tests
    pytest -m geometry       # Run only geometry tests
    pytest --cov=hierarchical # Run with coverage
"""

__version__ = "0.1.0"
__author__ = "hierArchical Testing Team"