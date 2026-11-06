"""Citation/ACL control fixtures only: no LLM, GPU or numerical measurements."""
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, make_card
from tests.test_infrastructure import TempCase
from tests.test_serial_debate_memory import SerialDebateMemoryRegression
from tests.support import experiment_payload
from tools.safeguards import write_artifact


SOURCE = 'def fit(x, parameters):\n    return parameters\ndef apply(x, state, parameters):\n    return x\n'


class ExperimentCitationClosure(TempCase):
    debate_task = SerialDebateMemoryRegression.debate_task

    def setUp(self):
        super().setUp()
        guard = patch('runtime.task_runtime.TaskRuntime.launch',
                      side_effect=AssertionError('citation fixture must not launch workers'))
        self.launch = guard.start()
        self.addCleanup(guard.stop)

    def fixture_evidence(self, wf, label='source'):
        # Same explicit test_double contract as existing memory/ACL tests.
        spec = make_card('ExperimentCard', wf.journal.run_id, experiment_payload())
        evidence = make_card('EvidenceCard', wf.journal.run_id, {
            'experiment_ref': spec['id'], 'execution_fingerprint': 'contract-fixture-' + label,
            'status': 'completed', 'observations': {'fixture_only': True},
            'provenance': {}, 'resources': {}, 'artifacts': [],
            'measurement_source': 'test_double', 'execution_id': 'NO_EXECUTION-' + label},
            refs=[spec['id']])
        wf.journal.commit('fixture/evidence/' + label, cards=[spec, evidence])
        return evidence

    @staticmethod
    def l3_draft(ref):
        draft = experiment_payload('L3')
        draft['evidence_refs'] = [ref]
        draft['execution'].update(service='evaluate_artifact', parameters={},
            evaluations=[{'kind': 'classification', 'split': 'search', 'count': 4}])
        return draft

    def plan(self, wf, name, *, parent=None):
        tid = wf.enqueue_role('fixture/plan/' + name, 'plan_l3', refs=[wf.brief['id']],
                              context={'round': 1}, parent_task=parent)
        wf.queue.ready()
        return wf.queue.claim(tid)

    def test_coded_request_and_bound_freeze_preserve_top_level_support(self):
        wf = self.workflow(1)
        evidence = self.fixture_evidence(wf)
        draft = experiment_payload()
        draft['execution'].update(requires_code=True, interface='quantizer_v1')
        draft['evidence_refs'] = [evidence['id']]
        request = wf.broker.request(draft, requester='generation', refs=[wf.brief['id']])
        self.assertEqual(request['payload']['evidence_refs'], [evidence['id']])
        self.assertEqual(request['refs'].count(evidence['id']), 1)
        wf.journal.commit('fixture/coded-request', cards=[request])
        artifact = write_artifact(wf.run_dir, request, {
            'interface': 'quantizer_v1', 'source': SOURCE}, 'coding')
        frozen = wf.broker.bind_execution(request, artifact)
        self.assertEqual(frozen['payload']['request_ref'], request['id'])
        self.assertEqual(frozen['payload']['evidence_refs'], [evidence['id']])
        self.assertIn(request['id'], frozen['refs'])
        self.assertEqual(frozen['refs'].count(evidence['id']), 1)
        wf.journal.commit('fixture/bound-execution', cards=[frozen])
        self.assertEqual(wf.journal.card(frozen['id']), frozen)
        self.launch.assert_not_called()

    def test_builtin_freeze_closes_citations_but_journal_still_rejects_bypass(self):
        wf = self.workflow(1)
        evidence = self.fixture_evidence(wf)
        draft = experiment_payload()
        draft['evidence_refs'] = [evidence['id']]
        frozen = wf.broker.freeze(draft, requester='generation',
                                   refs=[wf.brief['id'], evidence['id']])
        self.assertFalse(frozen['payload']['execution']['requires_code'])
        self.assertEqual(frozen['refs'].count(evidence['id']), 1)
        wf.journal.commit('fixture/builtin-execution', cards=[frozen])
        bypass = make_card('ExperimentCard', wf.journal.run_id, deepcopy(frozen['payload']),
                           refs=[wf.brief['id']])
        sequence = wf.journal.sequence
        with self.assertRaisesRegex(ContractError, '封套'):
            wf.journal.commit('fixture/missing-envelope', cards=[bypass])
        self.assertEqual(wf.journal.sequence, sequence)
        self.assertNotIn(bypass['id'], wf.journal.cards)

    def test_request_and_freeze_reject_malformed_unknown_and_wrong_kind_refs(self):
        wf = self.workflow(1)
        bad = ['not-a-list', [123], [wf.brief['id']], ['EvidenceCard-unknown-control-fixture@1']]
        for refs in bad:
            for method in ('request', 'freeze'):
                with self.subTest(refs=refs, method=method):
                    draft = experiment_payload()
                    draft['evidence_refs'] = refs
                    with self.assertRaises(ContractError):
                        getattr(wf.broker, method)(draft, requester='generation')

    def test_plan_submit_keeps_visible_support_and_rejects_bad_reference_types(self):
        wf = self.workflow(1)
        evidence = self.fixture_evidence(wf)
        task = self.plan(wf, 'visible')
        self.assertNotIn(evidence['id'], task['inputs']['refs'])
        wf.submit(task, self.l3_draft(evidence['id']), 'SCRIPTED_CITATION_FIXTURE')
        result = wf.queue.get(task['task_id'])['result']
        self.assertEqual(result['draft']['evidence_refs'], [evidence['id']])
        request = wf.broker.request(result['draft'], requester='judger', refs=task['inputs']['refs'])
        wf.journal.commit('fixture/plan-handoff', cards=[request])
        self.assertIn(evidence['id'], request['refs'])
        invalid = self.plan(wf, 'invalid')
        for ref in (wf.brief['id'], 'EvidenceCard-unknown-control-fixture@1'):
            with self.subTest(ref=ref), self.assertRaises(ContractError):
                wf.submit(invalid, self.l3_draft(ref), 'SCRIPTED_CITATION_FIXTURE')
            self.assertEqual(wf.queue.get(invalid['task_id'])['state'], 'running')
            self.assertIsNone(wf.queue.get(invalid['task_id']).get('result'))

    def test_plan_submit_cannot_cite_opposite_current_position_evidence(self):
        wf = self.workflow(1)
        own = self.debate_task(wf, side='A')
        other = self.debate_task(wf, side='B')
        evidence = self.fixture_evidence(wf, 'private-opposite')
        producer = wf.queue.enqueue('fixture/opposite/execute', 'experiment', inputs={
            'parent_task': other['task_id'], 'experiment_ref': evidence['payload']['experiment_ref']})
        wf.queue.ready(); wf.queue.claim(producer)
        wf.queue.finish(producer, outputs=[evidence['id']])
        task = self.plan(wf, 'private', parent=own['task_id'])
        self.assertFalse(wf.can_read_card(task, evidence))
        with self.assertRaises(ContractError):
            wf.submit(task, self.l3_draft(evidence['id']), 'SCRIPTED_CITATION_FIXTURE')
        self.assertEqual(wf.queue.get(task['task_id'])['state'], 'running')
        self.assertIsNone(wf.queue.get(task['task_id']).get('result'))
        self.assertEqual(wf.journal.card(evidence['id']), evidence)
        self.launch.assert_not_called()
