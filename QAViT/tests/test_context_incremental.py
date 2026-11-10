"""Exact condition sharing and field-only ACLs keep full sources and scientific facts."""
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import fingerprint, canonical, make_card
from core.journal import read_json
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as delivery


class IncrementalContextRegression(TempCase):
    task = delivery.DeliveryRegression.task
    begin = delivery.DeliveryRegression.begin
    tools = delivery.DeliveryRegression.tools
    science = delivery.DeliveryRegression.science

    def sample_card(self, wf):
        cards = self.science(wf, 'sample-sharing', commit=False)
        payload = deepcopy(cards[1]['payload'])
        ids = ['validation/' + str(i).zfill(8) for i in range(128)]
        data = {'search': {'sample_ids': ids, 'sample_count': 128, 'split': 'search', 'full_dataset': False}}
        payload['scope'] = {'data': deepcopy(data), 'limitations': ['subset only'], 'arbitrary_list': ids}
        payload['provenance'] = {'execution_facts': {'data': deepcopy(data)}, 'sample_selection': {'search': ids}}
        payload['observations'] = {'accuracy_delta': -0.02, 'counterevidence': 'held-out accuracy decreased', 'conditions': 'same 128 subset'}
        card = make_card('EvidenceCard', wf.journal.run_id, payload, refs=[cards[0]['id']])
        wf.journal.commit('sample-sharing', cards=[cards[0], card])
        return cards[0], card, ids

    def test_same_samples_once_with_precise_sources_and_order_preserved(self):
        wf = self.workflow(1)
        _, card, ids = self.sample_card(wf)
        seen = {}
        view = wf.present_card(card)
        wf.coalesce_condition_lists(card, view, seen)
        payload = view['payload']
        self.assertEqual(payload['scope']['data']['search']['sample_ids'], ids)
        self.assertEqual(payload['scope']['arbitrary_list'], ids)
        self.assertEqual(payload['observations'], card['payload']['observations'])
        for value in [payload['provenance']['execution_facts']['data']['search']['sample_ids'],
                      payload['provenance']['sample_selection']['search']]:
            self.assertEqual(value['sha256'], fingerprint(ids))
            self.assertEqual(wf.card_field(card, value['read_card']['field']), ids)
            source = value['body_location']['same_value_as']
            self.assertEqual(wf.card_field(view, source['field']), ids)
        changed = deepcopy(card)
        changed['payload']['scope']['data']['search']['sample_ids'].reverse()
        second = wf.present_card(changed)
        wf.coalesce_condition_lists(changed, second, seen)
        self.assertEqual(second['payload']['scope']['data']['search']['sample_ids'], ids[::-1])
        self.assertEqual(wf.journal.card(card['id']), card)

    def test_context_keeps_outcome_conditions_and_raw_read(self):
        wf = self.workflow(1)
        exp, card, ids = self.sample_card(wf)
        task, context = self.begin(wf, self.task(wf, refs=[card['id']]))
        shown = next(c for c in context['memory']['cards'] if c['id'] == card['id'])
        self.assertIn('held-out accuracy decreased', canonical(shown))
        self.assertIn('same 128 subset', canonical(shown))
        self.assertEqual(shown['payload']['scope']['data']['search']['sample_count'], 128)
        source = shown['payload']['provenance']['sample_selection']['search']['read_card']
        read = wf.host_tool({'inputs': {'name': 'read_card', 'parent_task': task['task_id'], 'arguments': source}})
        self.assertEqual(read['text'], canonical(ids))

    def test_acl_header_does_not_copy_observations(self):
        wf = self.workflow(1)
        _, card, _ = self.sample_card(wf)
        with patch.object(wf.journal, 'card', wraps=wf.journal.card) as read:
            header = wf.visibility_card(card['id'])
            self.assertEqual(header, {'id': card['id'], 'kind': 'EvidenceCard', 'refs': card['refs'], 'payload': {}})
            self.assertEqual([c.kwargs['field'] for c in read.call_args_list], [('refs',)])

    def test_independent_acl_header_reuses_one_verified_read(self):
        wf = self.workflow(1)
        _, card, _ = self.sample_card(wf)
        with patch('core.journal.read_json', wraps=read_json) as parse:
            wf.visibility_card(card['id'])
            self.assertEqual(parse.call_count, 1)
            wf.visibility_card(card['id'])
            self.assertEqual(parse.call_count, 2)  # New operation revalidates.
