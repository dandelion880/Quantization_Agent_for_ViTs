"""Real tiny-ViT source loading and adapter equivalence, not task accuracy."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError
from core.journal import file_sha256
from tests.test_infrastructure import TempCase
from tests.test_calibration_search import fixture
from tests.support import validation_config
from tools.experiments import asset, preflight_experiment
from tools.frozen import load_frozen
from tools.frozen_helpers import frozen_helper_service
from tools.experiment_worker import data_view
from ptq_agent.execution.pipeline import QuantizationExecutor
from ptq_agent.execution.propagation import profile_frozen_propagation
from ptq_agent.execution.calibration_search import search_frozen_scales
from ptq_agent.execution.model import forward


class FrozenHelperService(TempCase):
    def setUp(self):
        super().setUp()
        self.backend, self.cfg, self.data, self.plan = fixture()
        executor = QuantizationExecutor(self.backend, self.cfg, self.data)
        model, weights, activations, audit = executor.prepare(self.plan, {})
        self.package = self.root / 'model'
        executor.export(model, self.plan, weights, activations, self.package, audit)
        self.binding = {'model_dir': str(self.package), 'model_ref': 'fixture-model',
            'evidence_ref': 'fixture-evidence', 'candidate': None,
            'assets': [asset(p) for p in self.package.iterdir()]}
        self.brief = validation_config()['research_brief']

    def request(self, tool, args):
        value = preflight_experiment({'question': 'adapter equivalence', 'measurements': ['result'],
            'execution': {'service': 'frozen_helper', 'tool': tool, 'arguments': deepcopy(args)}}, self.brief)
        self.assertFalse(value['execution']['requires_code'])
        value['execution']['frozen_sources'] = [self.binding]
        return {'payload': value}

    def test_propagation_exact_numerics_nonzero_offset_and_frozen_source(self):
        request = self.request('profile_frozen_propagation', {'blocks': ['blocks.0', 'blocks.1'],
            'split': 'search', 'count': None, 'offset': 1,
            'interventions': [{'id': 'fp0', 'kind': 'fp_block', 'block': 'blocks.0'}]})
        request['payload']['execution']['research_quantization_contract'] = {'protocol': 'static_w4a4/1'}
        model, receipt = load_frozen(self.binding, self.backend, self.cfg, self.data)
        spec = request['payload']['execution']
        before = file_sha256(self.package / 'weights.pt')
        expected = profile_frozen_propagation(self.backend, model, data_view(self.data, spec['evaluation']),
            blocks=spec['arguments']['blocks'], split='search', interventions=spec['arguments']['interventions'],
            source={k: v for k, v in receipt.items() if k != 'manifest'})
        actual = frozen_helper_service(request, self.backend, self.cfg, self.data, self.root)
        self.assertEqual(actual['observations']['result'], expected)
        self.assertEqual(actual['sample_selection']['search'], self.data.ids('search')[1:2])
        self.assertTrue(actual['execution_facts']['frozen_loads'][0]['loaded_without_prepare'])
        self.assertEqual(before, file_sha256(self.package / 'weights.pt'))

    def test_scale_selection_exact_numerics_without_source_package_mutation(self):
        candidates = [{'id': str(f), 'scales': {'blocks.0.attn.qkv': {'activation': f}}} for f in (.8, 1., 1.2)]
        request = self.request('search_frozen_scales', {'candidates': candidates, 'count': 2,
            'offset': 1, 'block': 'blocks.0'})
        model, receipt = load_frozen(self.binding, self.backend, self.cfg, self.data)
        expected = search_frozen_scales(self.backend, model,
            data_view(self.data, request['payload']['execution']['calibration']), candidates, block='blocks.0')
        actual = frozen_helper_service(request, self.backend, self.cfg, self.data, self.root)
        report = deepcopy(actual['observations']['result'])
        report.pop('source')
        self.assertEqual(report, expected)
        self.assertEqual(actual['sample_selection']['calibration'], self.data.ids('calibration')[1:3])
        self.assertFalse(actual['observations']['selected_package_exported'])
        # Re-loading returns the original unchanged package, never search state.
        for receipt in self.binding['assets']:
            self.assertEqual(file_sha256(receipt['path']), receipt['sha256'])

    def test_reject_final_search_unknown_arguments_and_conflicting_views(self):
        cases = [('profile_frozen_propagation', {'blocks': ['blocks.0'], 'split': 'final', 'count': 1}),
                 ('search_frozen_scales', {'candidates': [], 'split': 'search', 'count': 1}),
                 ('search_frozen_scales', {'candidates': [], 'ignored_parameter': 3, 'count': 1})]
        for tool, args in cases:
            with self.subTest(tool=tool, args=args), self.assertRaises(ContractError):
                self.request(tool, args)
        request = self.request('profile_frozen_propagation', {'blocks': ['blocks.0'], 'count': 2})
        request['payload']['execution']['calibration']['count'] = 1
        with self.assertRaises(ContractError):
            preflight_experiment(request['payload'], self.brief)

    def test_ambiguous_or_changed_frozen_source_is_not_recalibrated(self):
        from runtime.failures import RequestRevisionRequired, FrozenSourceError
        request = self.request('profile_frozen_propagation', {'blocks': ['blocks.0'], 'count': 1})
        request['payload']['execution']['frozen_sources'].append(dict(self.binding, model_ref='another'))
        with self.assertRaises(RequestRevisionRequired):
            frozen_helper_service(request, self.backend, self.cfg, self.data, self.root)
        request['payload']['execution']['source_ref'] = 'fixture-model'
        frozen_helper_service(request, self.backend, self.cfg, self.data, self.root)
        (self.package / 'unexpected').write_text('modified package fixture')
        with self.assertRaises(FrozenSourceError):
            frozen_helper_service(request, self.backend, self.cfg, self.data, self.root)

    def test_source_card_handles_and_package_selectors_use_distinct_routes(self):
        from types import SimpleNamespace
        from agents.interaction import RoleInteraction
        from tools.runtime_interface import select_frozen_source
        from runtime.failures import RequestRevisionRequired
        evidence, private, handle = 'EvidenceCard-known@1', 'EvidenceCard-private@1', 'C7'
        reads = []

        def visible(task, ref):
            reads.append(ref)
            if ref not in (evidence, handle):
                raise ContractError('not visible')
            return {'id': evidence}

        resolver = SimpleNamespace(journal=SimpleNamespace(cards={evidence: {}, private: {}}),
                                   require_visible=visible)
        binding = dict(self.binding, evidence_ref=evidence, solution_id='catalog-solution')
        for ref in (handle, evidence, 'fixture-model', 'catalog-solution'):
            submitted = {'evidence_refs': [handle], 'execution': {'service': 'frozen_helper', 'source_ref': ref}}
            value = RoleInteraction.resolve_card_refs(resolver, None, submitted)
            expected = evidence if ref in (handle, evidence) else ref
            self.assertEqual(value['execution']['source_ref'], expected)
            self.assertEqual(select_frozen_source([binding], expected), binding)
        self.assertNotIn('fixture-model', reads)
        self.assertNotIn('catalog-solution', reads)
        with self.assertRaises(ContractError):
            RoleInteraction.resolve_card_refs(resolver, None,
                {'execution': {'service': 'frozen_helper', 'source_ref': private}})
        with self.assertRaises(ContractError):
            RoleInteraction.resolve_card_refs(resolver, None,
                {'execution': {'service': 'custom_experiment', 'source_ref': 'fixture-model'}})
        value = RoleInteraction.resolve_card_refs(resolver, None,
            {'execution': {'service': 'frozen_helper', 'source_ref': 'not-cited'}})
        with self.assertRaises(RequestRevisionRequired):
            select_frozen_source([binding], value['execution']['source_ref'])

    def test_worker_evidence_keeps_exact_source_parameters_and_actual_scope(self):
        from core.cards import make_card
        from core.journal import atomic_json
        from tools.experiments import ExperimentBroker
        from tools.experiment_worker import run_experiment
        journal = self.journal()
        brief = make_card('ResearchBrief', journal.run_id, self.brief)
        journal.commit('brief', cards=[brief])
        request = self.request('profile_frozen_propagation', {'blocks': ['blocks.0'],
                               'split': 'search', 'count': 1, 'offset': 1})
        card = make_card('ExperimentCard', journal.run_id, request['payload'])
        # Only process isolation/backend construction are replaced in this CPU
        # interface test; loading, numerical forwards, worker/provenance and
        # Evidence sealing use the real implementations.
        with patch('tools.experiment_worker.backend_and_data', return_value=(self.backend, self.cfg, self.data)), \
             patch('runtime.guards.protect_files', return_value={'test': 'isolation not exercised'}), \
             patch('runtime.guards.install_gradient_guard'):
            result = run_experiment({'experiment': card}, self.root)
        receipt = {'status': 'completed', 'experiment_ref': card['id'],
            'experiment_sha256': card['content_sha256'], 'data': result,
            'resources': {}, 'execution_id': 'actual-tiny-cpu-worker'}
        path = self.root / 'receipt.json'
        atomic_json(path, receipt)
        evidence = ExperimentBroker(journal, brief['id'], None).evidence(card, receipt, path)
        scope = evidence['payload']['scope']
        self.assertEqual(scope['measurement_kind'], 'frozen_propagation')
        self.assertEqual(scope['parameters']['source']['model_ref'], 'fixture-model')
        self.assertEqual(scope['parameters']['arguments']['blocks'], ['blocks.0'])
        self.assertEqual(scope['site'], ['blocks.0'])
        self.assertEqual(scope['data']['search']['sample_ids'], self.data.ids('search')[1:2])
        self.assertEqual(evidence['payload']['measurement_source'], 'synthetic_computation')


if __name__ == '__main__':
    unittest.main()
