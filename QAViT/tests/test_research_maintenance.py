"""Append-only corrections and archive visibility at a frozen read boundary."""
from copy import deepcopy
from pathlib import Path
import unittest
from core.cards import ContractError, make_card
from core.memory import ResearchMemory
from tests.test_infrastructure import TempCase
from tests import test_fallback_solutions as fallback
from tests.support import meta


class ResearchMaintenance(TempCase):
    promote = fallback.FallbackTests.promote

    def test_evicted_and_never_top3_results_remain_visible_without_future_leak(self):
        wf = self.workflow(1)
        self.promote(wf, 1, 80, 40)
        boundary = wf.journal.sequence
        for i in range(2, 5):
            self.promote(wf, i, 80, 40 + i)
        self.promote(wf, 5, 60, 20)
        active = {r['id'] for r in wf.solutions.snapshot()}
        self.assertNotIn('model1', active)
        self.assertNotIn('model5', active)
        catalog = {r['id']: r for r in wf.solutions.catalog()}
        for key in ('model1', 'model5'):
            self.assertEqual(catalog[key]['retention'], 'archive')
            self.assertTrue(Path(catalog[key]['configuration_asset']).is_file())
        old = {r['id'] for r in wf.solutions.available(boundary)}
        self.assertIn('model1', old)
        self.assertNotIn('model5', old)
        self.assertEqual(wf.solutions.fallback()['candidate']['model_ref'], 'model4')

    def test_correction_is_current_retrievable_idempotent_and_time_scoped(self):
        wf = self.workflow(1)
        content = meta(1)
        content['knowledge_updates'] = [{'claim_id': 'timing', 'text': 'old incorrect timing',
            'status': 'hypothesis', 'scope': {'model': 'fixture'}, 'evidence_refs': [], 'rationale': 'old'}]
        original = make_card('MetareviewCard', wf.journal.run_id, content, author='recorder')
        wf.journal.commit('fixture/review', cards=[original], events=[wf.memory.apply_metareview(original)])
        cutoff = wf.journal.sequence
        before = wf.journal.card_path(original['id']).read_bytes()
        update = {**content['knowledge_updates'][0], 'text': 'pre-promotion B differs from current B',
                  'rationale': 'explicit maintenance fixture', 'supersedes': original['id']}
        kwargs = dict(reason='timing correction', operator='test; no model called', guidance='use corrected timing', key='fix/timing')
        wf.memory.append_correction([update], **kwargs)
        sequence = wf.journal.sequence
        wf.memory.append_correction([update], **kwargs)
        self.assertEqual(wf.journal.sequence, sequence)
        with self.assertRaises(ContractError):
            wf.memory.append_correction([update], **{**kwargs, 'guidance': 'different text'})
        self.assertEqual(before, wf.journal.card_path(original['id']).read_bytes())
        result = wf.retriever.retrieve('generation', query={'layer': 'L2'})
        self.assertEqual(result['entries'][0]['body']['text'], update['text'])
        self.assertEqual(result['entries'][0]['version'], 2)
        history = wf.retriever.retrieve('generation', explicit_refs=[original['id']])
        self.assertEqual(history['entries'][0]['knowledge_revisions'][0]['text'], update['text'])
        old = wf.retriever.retrieve('generation', explicit_refs=[original['id']], sequence=cutoff)
        self.assertNotIn('knowledge_revisions', old['entries'][0])
        self.assertEqual(ResearchMemory(wf.journal).snapshot(cutoff)['L2']['timing']['version'], 1)
        with self.assertRaises(ContractError):
            wf.memory.append_correction([update], **{**kwargs, 'key': 'fix/stale'})

if __name__ == '__main__':
    unittest.main()
