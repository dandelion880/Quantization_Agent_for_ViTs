"""Versioned G-only criteria do not rewrite the frozen uploaded v0.1 package."""
from pathlib import Path
import sys, unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.evaluation.rules import Rules
from tools.evaluation.common import QAVIT

class GOnlyRules(unittest.TestCase):
    def test_new_selection_and_old_package_remain_independently_valid(self):
        old=Rules(QAVIT/'configs/evaluation/v0_1')
        new=Rules(QAVIT/'configs/evaluation/gonly_v0_2')
        self.assertNotIn('select',old.selected)
        self.assertIn('select',new.selected)
        self.assertNotEqual(old.version,new.version)
        self.assertEqual([c['id'] for c in new.criteria('select')],['S1','S2','S3','S4'])
        self.assertIn('select',new.schema['properties']['purpose']['enum'])
        self.assertNotIn('select',old.schema['properties']['purpose']['enum'])
