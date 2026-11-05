"""Service conditions reflect real data use, and deterministic fields are host owned."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError
from tools.experiments import normalize_execution, preflight_experiment
from tests.support import validation_config

class ServiceConditions(unittest.TestCase):
    def setUp(self):
        self.brief = validation_config()['research_brief']
        self.brief['hard_constraints']['calibration_samples'] = 32

    def test_queries_do_not_fabricate_calibration(self):
        for tool in ('query_execution_capabilities', 'query_ptq_sites', 'query_structure', 'query_ptq_plugins'):
            with self.subTest(tool=tool):
                value = normalize_execution({'service':'profile', 'tool':tool, 'arguments':{}}, self.brief)
                self.assertNotIn('calibration', value)
                self.assertNotIn('calibration_samples', value['resolved_choices'])
                self.assertEqual(value['quantization_contract'], {})
                self.assertEqual(value['resolved_choices']['training'], False)

    def test_measurement_inherits_fixed_conditions_and_mirrors_actual_arguments(self):
        value = normalize_execution({'service':'profile', 'tool':'profile_structure', 'arguments':{'modules':[]}}, self.brief)
        self.assertEqual(value['arguments']['count'], 32)
        self.assertEqual(value['calibration'], {'count':32,'offset':0,'split':'calibration'})
        self.assertEqual(value['resolved_choices']['calibration_samples'], 32)
        local = normalize_execution({'service':'profile','tool':'profile_structure','arguments':{'modules':[], 'count':1}},self.brief)
        self.assertEqual(local['calibration']['count'],1)

    def test_explicit_sample_mirror_conflict_remains_local(self):
        with self.assertRaisesRegex(ContractError, 'execution.calibration.count'):
            normalize_execution({'service':'profile','tool':'profile_structure','arguments':{'modules':[], 'count':32},
                                 'calibration':{'count':2}},self.brief)

    def test_query_drops_inapplicable_calibration_mirror(self):
        value = normalize_execution({'service':'profile','tool':'query_ptq_sites','arguments':{},
                                     'calibration':{'count':32}},self.brief)
        self.assertNotIn('calibration',value)
        self.assertNotIn('calibration_samples',value['resolved_choices'])

    def test_query_does_not_check_unconsumed_global_constraints(self):
        value = preflight_experiment({'question':'Available sites?', 'measurements':['site catalog'],
            'execution': {'service':'profile','tool':'query_ptq_sites','arguments':{},
                          'resolved_choices':{'calibration_samples':2,'activation_bits':8}}}, self.brief)
        self.assertNotIn('calibration', value['execution'])
        self.assertEqual(value['execution']['quantization_contract'], {})

    def test_weight_distribution_reads_no_calibration_images(self):
        value = normalize_execution({'service':'profile','tool':'profile_distribution',
            'arguments':{'target':'weight'}, 'calibration':{'count':1}},self.brief)
        self.assertNotIn('calibration',value)
        self.assertEqual(value['arguments']['count'],1)
        self.assertEqual(value['quantization_contract'],{})

if __name__ == '__main__': unittest.main()
