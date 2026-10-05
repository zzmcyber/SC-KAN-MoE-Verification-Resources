"""Run the public release checks; every discovered test must execute."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
if result.skipped:
    print('Release checks require zero skipped tests.',file=sys.stderr)
raise SystemExit(0 if result.wasSuccessful() and not result.skipped and result.testsRun>0 else 1)
