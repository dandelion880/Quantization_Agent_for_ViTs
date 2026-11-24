"""Source declaration and researcher recovery, including the H3 legacy spelling."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError
from core.experiment_requests import ExperimentRequests
from runtime.failures import RequestRevisionRequired, RemoteWorkerError, FrozenSourceError, failure_details
from tools.experiments import normalize_source_refs, EXPERIMENT_DRAFT_SCHEMA
from tools.runtime_interface import select_frozen_source
from tests import test_code_repair as repair


class SourceContractTests(unittest.TestCase):
    def test_public_schema_and_legacy_refs_share_canonical_identity(self):
        payload = {'controls': {'evidence_refs': ['e1'], 'solution_refs': ['s1']}, 'evidence_refs': ['e0', 'e1']}
        original = deepcopy(payload)
        normalized = normalize_source_refs(payload)
        self.assertEqual(normalized['evidence_refs'], ['e0', 'e1'])
        self.assertEqual(normalized['solution_refs'], ['s1'])
        self.assertEqual(payload, original)
        self.assertEqual(normalize_source_refs(normalized), normalized)
        self.assertIn('solution_refs', EXPERIMENT_DRAFT_SCHEMA['properties'])
        self.assertIn('evidence_refs', EXPERIMENT_DRAFT_SCHEMA['properties'])
        with self.assertRaises(ContractError): normalize_source_refs({'controls': {'evidence_refs': 'not-list'}})

    def test_model_ref_alias_requires_one_exact_bound_source(self):
        binding = {'evidence_ref': 'e1', 'model_ref': 's1', 'model_dir': '/package', 'assets': []}
        for ref in (None, 'e1', 's1'):
            self.assertEqual(select_frozen_source([binding], ref), binding)
        # Repeated evidence/solution citations of one immutable package aren't ambiguous.
        self.assertEqual(select_frozen_source([binding, {**binding, 'solution_id': 's1'}], 's1')['model_dir'], '/package')
        historical = {**binding, 'source_snapshot': {'sha256': 'historical'}}
        self.assertEqual(select_frozen_source([historical, binding]), historical)
        with self.assertRaises(RequestRevisionRequired): select_frozen_source([binding, {**binding, 'model_dir': '/other'}])
        with self.assertRaises(RequestRevisionRequired): select_frozen_source([], 's1')
        with self.assertRaises(RequestRevisionRequired): select_frozen_source([binding, {**binding, 'model_dir': '/other'}], 's1')
        with self.assertRaises(ContractError): select_frozen_source([binding], {'ref': 's1'})
        with self.assertRaises(RequestRevisionRequired): select_frozen_source([binding], '/package')

    def test_wrapped_request_error_keeps_structured_cause(self):
        try:
            try:
                select_frozen_source([], 'missing')
            except RequestRevisionRequired as exc:
                raise RuntimeError('plugin wrapped load') from exc
        except RuntimeError as exc:
            error = failure_details(exc, {})
        self.assertEqual(error['code'], 'RuntimeError')
        self.assertEqual(error['origin'], 'request')
        self.assertEqual(error['request_cause']['requested_source'], 'missing')
        self.assertEqual(error['request_cause']['available_sources'], [])
        self.assertFalse(error['request_cause']['mechanism_measurement_completed'])
        self.assertTrue(error['request_cause']['partial_execution_possible'])
        self.assertEqual(error['recovery'], 'revise_request')

    def test_provenance_survives_child_receipt_and_does_not_burn_coding(self):
        try:
            select_frozen_source([], 'missing')
        except RequestRevisionRequired as exc:
            error = failure_details(exc, {'approval': {'candidate': {'path': __file__}}})
        remote = RemoteWorkerError(error, worker='classification', receipt='result.json', worker_log='worker.log')
        details = failure_details(remote, {})
        self.assertEqual(details['origin'], 'request')
        self.assertEqual(details['requested_source'], 'missing')
        self.assertFalse(ExperimentRequests.revisable_failure({'error': details}))
        self.assertEqual(failure_details(FrozenSourceError('package hash changed'), {})['origin'], 'host')
        self.assertEqual(failure_details(RuntimeError('CUDA out of memory'), {})['origin'], 'resource')


class SourceJournalTests(repair.TempCase):
    def test_evidence_binding_uses_committed_card_index_and_original_source_era(self):
        import hashlib
        from core.cards import make_card, fingerprint
        from core.journal import Journal, atomic_json, file_sha256, read_json
        from core.solutions import SolutionLibrary
        from tests.support import experiment_payload
        from tools.frozen import bind_sources, _source_files
        wf = self.workflow(1)
        first = wf.broker.freeze(experiment_payload(), requester='generation')
        wf.journal.commit('fixture/first-experiment', cards=[first])
        initial = SolutionLibrary(wf.journal, wf.brief['payload']).source_record(
            wf.journal.cards[first['id']]['sequence'])
        changed = deepcopy(read_json(wf.run_dir / 'research_run.json')['source_snapshot'])
        changed['tools/ptq_agent/__init__.py'] += '\n# fixture source era after first experiment\n'
        later_version = fingerprint({key: hashlib.sha256(value.encode()).hexdigest() for key, value in changed.items()})
        wf.journal.commit('fixture/source-intervention', events=[{'type': 'implementation_intervention',
            'source_snapshot': changed, 'code_identity': {'core_sha256': later_version}}])
        second = wf.broker.freeze(experiment_payload(), requester='generation')
        wf.journal.commit('fixture/second-experiment', cards=[second])
        evidence_refs = []
        for index, experiment in enumerate((first, second)):
            directory = self.root / str(index) / 'model'
            path = directory / 'manifest.json'
            atomic_json(path, {'fixture': 'source binding only; no numerical measurement'})
            evidence = make_card('EvidenceCard', wf.journal.run_id, {
                'experiment_ref': experiment['id'], 'execution_fingerprint': 'fixture', 'status': 'completed',
                'provenance': {'model_ref': 'fixture-model-' + str(index)},
                'observations': {'reload': {'host_verified': True}}, 'resources': {},
                'artifacts': [{'path': str(path), 'sha256': file_sha256(path), 'bytes': path.stat().st_size}],
                'measurement_source': 'test_double', 'execution_id': 'fixture-' + str(index)},
                author='program', refs=[experiment['id']])
            wf.journal.commit('fixture/evidence-' + str(index), cards=[evidence])
            evidence_refs.append(evidence['id'])
        # The production envelope deliberately has no sequence; its journal index does.
        self.assertNotIn('sequence', wf.journal.card(first['id']))
        self.assertIn('sequence', wf.journal.cards[first['id']])
        bindings = bind_sources(wf.journal, evidence_refs)
        self.assertEqual([b['source_version'] for b in bindings], [initial['source_version'], later_version])
        self.assertNotEqual(bindings[0]['source_cache'], bindings[1]['source_cache'])
        self.assertEqual(_source_files(bindings[1]), changed)
        for binding in bindings:
            self.assertTrue((Path(binding['source_cache']) / 'ptq_agent' / '__init__.py').is_file())
        restored = Journal(wf.run_dir, wf.journal.run_id)
        self.assertEqual(bind_sources(restored, evidence_refs), bindings)


class RequestReturnTests(repair.TempCase):
    setup_request = repair.CodeRepairRegression.setup_request
    progress = repair.CodeRepairRegression.progress
    complete = repair.CodeRepairRegression.complete
    review_attempt = repair.CodeRepairRegression.review_attempt

    def test_failed_load_returns_generation_request_without_code_revision_and_restores(self):
        from core.workflow import Workflow
        wf, tid = self.setup_request()
        self.review_attempt(wf, tid, approve=True)
        wf.queue.ready()
        child = wf.queue.claim(wf.tasks()['fixture/request/execute']['task_id'])
        error = failure_details(RequestRevisionRequired('missing source', requested_source='missing'), {})
        wf.queue.fail(child['task_id'], error)
        self.progress(wf, tid)
        request = wf.queue.get(tid)
        self.assertEqual(request['state'], 'completed')
        self.assertEqual(request['result']['reason'], 'request_revision_required')
        self.assertEqual(request['result']['error'], error)
        self.assertNotIn('fixture/request/code/revision1', wf.tasks())
        restored = Workflow(wf.run_dir)
        self.assertEqual(restored.queue.get(tid)['result'], request['result'])
        self.assertEqual(restored.queue.get(child['task_id'])['state'], 'failed')
        self.assertEqual(restored.journal.find_cards('EvidenceCard'), [])


if __name__ == '__main__': unittest.main()
