"""Exact scope membership semantics and bounded serialization work; CPU only."""
from copy import deepcopy
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, canonical
from core.evidence_scope import contains, validate_claim_scope, VERSION


def previous_contains(measured, stated):
    if isinstance(measured, dict):
        return isinstance(stated, dict) and all(
            key in measured and previous_contains(measured[key], value) for key, value in stated.items())
    if isinstance(stated, list):
        available = measured if isinstance(measured, list) else [measured]
        return bool(stated) and all(any(canonical(value) == canonical(item) for item in available) for value in stated)
    if isinstance(measured, list):
        return any(canonical(stated) == canonical(item) for item in measured)
    return canonical(measured) == canonical(stated)


class ScopeMembership(unittest.TestCase):
    def test_matches_previous_algorithm_for_json_values_duplicates_and_order(self):
        atoms = [None, False, True, 0, 1, 1.0, '1', '', '样本', {},
                 {'a': 1, 'b': [True, None]}, {'b': [True, None], 'a': 1},
                 [1, '1'], [False, {'nested': [1, 2]}]]
        values = [*atoms, [], atoms, list(reversed(atoms)), atoms + atoms]
        values += [[value] for value in atoms]
        values += [[value, value] for value in atoms]
        values += [{'data': {'ids': value}} for value in ([], ['a'], ['b', 'a'], ['a', 'a'])]
        for measured, stated in itertools.product(values, repeat=2):
            with self.subTest(measured=measured, stated=stated):
                self.assertEqual(contains(measured, stated), previous_contains(measured, stated))
        self.assertFalse(contains([], []))
        self.assertFalse(contains([1], []))
        self.assertFalse(contains([1], [True]))
        self.assertFalse(contains([1], [1.0]))
        self.assertTrue(contains(['a', 'b'], ['b', 'a', 'a']))

    def test_serializes_each_list_item_once(self):
        values = ['sample-' + str(index) for index in range(5000)]
        with patch('core.evidence_scope.canonical', wraps=canonical) as serialize:
            self.assertTrue(contains(values, values))
        self.assertEqual(serialize.call_count, 2 * len(values))

    def test_data_order_and_exact_frozen_conditions_remain_required(self):
        measured = {'schema_version': VERSION, 'model': 'fixture', 'measurement_kind': 'full_model',
                    'measurement_source': 'synthetic_computation',
                    'data': {'final': {'sample_ids': ['a', 'b'], 'sample_count': 2}},
                    'quantization': {'x': {'bits': 4}}, 'parameters': {'alpha': 1},
                    'implementation': {'source': 'frozen'}}
        validate_claim_scope(measured, measured)
        for field, value in (('data', {'final': {'sample_ids': ['b', 'a'], 'sample_count': 2}}),
                             ('data', {'final': {'sample_ids': ['a']}}),
                             ('quantization', {}), ('parameters', {}), ('implementation', {})):
            stated = deepcopy(measured)
            stated[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ContractError):
                validate_claim_scope(stated, measured)


if __name__ == '__main__':
    unittest.main()
