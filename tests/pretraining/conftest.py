"""Discontinued SSL method tests (BYOL, JEPA, LeWM, VICReg).

These modules still exist in src/speech_decoding/pretraining/ but are
not part of the active v12 direction. Tests are kept for reference
but excluded from default collection.

Run explicitly with: pytest tests/pretraining/ -p no:conftest
"""
collect_ignore_glob = ["test_*.py"]
