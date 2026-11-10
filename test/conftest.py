"""Make test/golden importable, once, for every test that needs it.

golden/ holds the two reference configurations and the numbers the old repo
produced for them. It is a fixture directory, not a package, so it is not
installed and has to be put on the path here rather than in four test files.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "golden"))
