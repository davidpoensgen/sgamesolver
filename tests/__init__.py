"""Pytest requires __init__.py file in tests folder and subfolders.

Pytest assumes root folder to be the first folder up in the
hierarchy without __init__.py file.

If tests folder and subfolders do not contain __init__.py file,
they will considered root folders during test execution.

Root folder should be the folder containing the tests folder.
"""
