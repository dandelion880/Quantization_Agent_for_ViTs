"""Verified field reads avoid whole-card copies without weakening read-session integrity."""
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, make_card
from core.journal import atomic_json
import core.journal as journal_module
from tests.support import hypothesis
from tests.test_infrastructure import TempCase


class JournalReadProjectionRegression(TempCase):
    def setUp(self):
        super().setUp()
        self.store = self.journal()
        payload = {**hypothesis(1, 'H1'), 'diagnostic_values': list(range(20000))}
        self.value = make_card('HypothesisCard', self.store.run_id, payload, author='generation')
        self.ref = self.value['id']
        self.store.commit('fixture/large_card', cards=[self.value])

    def test_default_and_projected_reads_are_equal(self):
        with self.store.read_session():
            full = self.store.card(self.ref)
            self.assertEqual(self.store.card(self.ref, field=('kind',)), full['kind'])
            self.assertEqual(self.store.card(self.ref, field=('payload', 'diagnostic_values')),
                             full['payload']['diagnostic_values'])
            full['payload']['diagnostic_values'][0] = -99
            self.assertEqual(self.store.card(self.ref)['payload']['diagnostic_values'][0], 0)

    def test_mutable_projected_values_are_independent(self):
        with self.store.read_session():
            payload = self.store.card(self.ref, field=('payload',))
            payload['diagnostic_values'][0] = -99
            self.assertEqual(self.store.card(self.ref, field=('payload', 'diagnostic_values'))[0], 0)

    def test_nested_cache_and_new_session_revalidation(self):
        with patch.object(journal_module, 'validate_card', wraps=journal_module.validate_card) as validate:
            with self.store.read_session():
                self.store.card(self.ref, field=('kind',))
                with self.store.read_session():
                    self.store.card(self.ref, field=('kind',))
                self.assertEqual(validate.call_count, 1)
            with self.store.read_session():
                self.store.card(self.ref, field=('kind',))
            self.assertEqual(validate.call_count, 2)

    def test_same_session_tampering_is_rejected(self):
        with self.store.read_session():
            self.store.card(self.ref, field=('kind',))
            changed = deepcopy(self.value)
            changed['payload']['diagnostic_values'][0] = -99
            atomic_json(self.store.card_path(self.ref), changed)
            with self.assertRaises(ContractError):
                self.store.card(self.ref, field=('kind',))

    def test_unknown_reference_is_rejected(self):
        with self.assertRaises(ContractError):
            self.store.card('not-an-existing-card', field=('kind',))
