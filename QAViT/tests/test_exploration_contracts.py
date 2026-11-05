"""Exploration conditions apply to interventions; observation scope is independent."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError
from core.execution_contract import check_plan
from tools.experiments import preflight_experiment
from tests.support import validation_config, experiment_payload
from tests.test_execution_contracts import plan_fixture

class ExplorationContracts(unittest.TestCase):
    def setUp(self):
        self.brief = validation_config()['research_brief']
        self.brief['hard_constraints'].update(calibration_samples=32)
        self.brief['quantization_contract'] = {
            'required_sites':['blocks.0.mlp.fc1','blocks.0.mlp.fc2'],
            'specs':{'weight':{'bits':4},'activation':{'bits':4}}}

    def test_local_intervention_with_whole_model_observation_keeps_own_conditions(self):
        draft = experiment_payload('L2')
        execution = draft['execution']
        execution['parameters']['bits'] = 3
        execution['resolved_choices'].update(activation_bits=3,weight_bits=32,calibration_samples=1)
        execution['calibration']['count']=1
        execution['evaluations']=[{'kind':'classification','split':'search','count':1}]
        execution['measurement_kind']='full_model'
        result=preflight_experiment(draft,self.brief,executable=True)
        contract=result['execution']['quantization_contract']
        self.assertEqual(contract['specs']['activation']['bits'],3)
        self.assertEqual(contract['specs']['weight']['bits'],32)
        self.assertEqual(contract['required_sites'],['blocks.0.mlp.fc2'])
        self.assertEqual(result['execution']['calibration']['count'],1)

    def test_no_image_custom_new_names_and_parameters_need_no_native_mapping(self):
        draft={'question':'Weight-only statistic', 'measurements':['error'],
            'execution':{'service':'custom_experiment','parameters':{'alpha':.3,'bits':3},
                'resolved_choices':{'weight_bits':3,'weight_method':'novel_signed_grid'},
                'quantization_contract':{'calibrations':{'weight':{'strategy':'novel_weight_observer','alpha':.3}}}}}
        result=preflight_experiment(draft,self.brief,executable=True)['execution']
        self.assertNotIn('calibration',result)
        self.assertNotIn('calibration_samples',result['resolved_choices'])
        self.assertEqual(result['quantization_contract']['methods']['weight'],'novel_signed_grid')
        self.assertEqual(result['parameters'],{'alpha':.3,'bits':3})
        self.assertNotIn('parameter_bindings',result['quantization_contract'])

    def test_request_preflight_checks_only_supplied_structure_before_coding(self):
        from unittest.mock import patch
        minimal = {'question': 'Weight-only statistic', 'measurements': ['error']}
        custom = {**minimal, 'execution': {'service': 'custom_experiment',
            'parameters': {'alpha': .3}, 'quantization_contract': {
                'methods': {'weight': 'new_method'},
                'calibrations': {'weight': {'strategy': 'new_observer', 'parameters': {'alpha': .3}}},
                'specs': {'weight': {'bits': 3}}}}}
        with patch('tools.experiments.freeze_contract', side_effect=AssertionError('premature freeze')), \
             patch('tools.experiments.asset', side_effect=AssertionError('premature asset read')), \
             patch('ptq_agent.execution.sites.site_catalog', side_effect=AssertionError('premature catalog')):
            self.assertEqual(preflight_experiment(minimal, self.brief)['stage'], 'request')
            self.assertEqual(preflight_experiment(custom, self.brief)['execution'],
                             {**custom['execution'], 'requires_code': True, 'interface': 'experiment_v1',
                              'resolved_choices': {}})
            wrong = deepcopy(custom)
            wrong['execution']['quantization_contract']['weight'] = \
                wrong['execution']['quantization_contract'].pop('specs')['weight']
            with self.assertRaisesRegex(ContractError, '未识别.*weight'):
                preflight_experiment(wrong, self.brief)
            self.assertIn('weight', wrong['execution']['quantization_contract'])

    def test_actual_subscription_decoder_accepts_open_choices_and_nullable_views(self):
        import json
        from agents.roles import role_spec, tools_for
        from ptq_agent.schemas import validate, PTQError
        from runtime.subscription import StructuredOutputError
        from tests.test_interface_simplification import decode
        specification = role_spec('generation', 'generate', {}, {})
        tool_schema = next(t['parameters'] for t in tools_for('generate') if t['name'] == 'request_experiment')
        minimal = {'question': 'Weight-only statistic', 'measurements': ['error']}
        custom = {**minimal, 'resources': {'gpu': None}, 'execution': {
            'service': 'custom_experiment', 'backend': deepcopy(self.brief['model']['adapter']),
            'resolved_choices': {'weight_bits': 3}, 'parameters': {'alpha': .3, 'new_option': {'rank': 2}},
            'method_options': {'new_option': [1, 2]},
            'quantization_contract': {'methods': {'weight': 'new_method'},
                'calibrations': {'weight': {'strategy': 'new_observer', 'alpha': .3}}},
            'calibration': {'split': 'calibration', 'count': None, 'sampling_note': {'purpose': 'full view'}},
            'evaluations': [{'kind': 'custom_metric', 'split': 'search', 'count': None, 'metrics': []}]}}
        for draft in (minimal, custom):
            with self.subTest(draft=draft):
                args = {'experiment': draft}
                validate(args, tool_schema)
                response = decode(specification, json.dumps({'name': 'request_experiment',
                    'arguments_json': json.dumps(args)}))
                self.assertEqual(response.tool_calls[0].arguments, args)
                self.assertEqual(preflight_experiment(draft, self.brief)['stage'], 'request')
        changes = [lambda d: d['execution'].update(parameters=[]),
                   lambda d: d['execution']['calibration'].update(count=-1),
                   lambda d: d['resources'].update(gpu='0'),
                   lambda d: d['execution']['evaluations'][0].update(offset=1)]
        for change in changes:
            bad = deepcopy(custom); change(bad)
            with self.subTest(change=change):
                args = {'experiment': bad}
                with self.assertRaises(PTQError):
                    validate(args, tool_schema)
                with self.assertRaises(StructuredOutputError):
                    decode(specification, json.dumps({'name': 'request_experiment',
                        'arguments_json': json.dumps(args)}))

    def test_supplied_invalid_or_conflicting_request_fields_are_repaired_before_binding(self):
        base = {'question': 'Local weight error', 'measurements': ['error'],
                'execution': {'service': 'custom_experiment'}}
        cases = [({'quantization_contract': {'specs': {'weight': {'bits': 3}}},
                   'resolved_choices': {'weight_bits': 4}}, '冲突'),
                 ({'parameters': {'alpha': .3}, 'quantization_contract': {'parameters': {'alpha': .4}}}, '冲突'),
                 ({'parameters': []}, 'parameters'), ({'resolved_choices': []}, 'resolved_choices'),
                 ({'calibration': {'count': 0}}, 'count'), ({'seed': -1}, 'seed'),
                 ({'requires_code': 'yes'}, 'requires_code'),
                 ({'backend': {'kind': 'unauthorized'}}, '授权'),
                 ({'evaluations': [{'kind': 'classification', 'invented_required_field': 1}]}, 'invented_required_field'),
                 ({'required_unknown': []}, '未识别')]
        for fields, message in cases:
            draft = deepcopy(base); draft['execution'].update(fields)
            with self.subTest(fields=fields), self.assertRaisesRegex(ContractError, message):
                preflight_experiment(draft, self.brief)
        runtime = {'cpu_threads': 2, 'allowed_gpus': [0]}
        for resources in ({'cpu_threads': 3}, {'gpu': 1}):
            with self.subTest(resources=resources), self.assertRaisesRegex(ContractError, '资源上限|授权范围'):
                preflight_experiment({**base, 'resources': resources}, self.brief, runtime)
        with self.assertRaisesRegex(ContractError, '实验层级'):
            preflight_experiment({**base, 'level': 'L3', 'execution': {'service': 'tensor_probe'}}, self.brief)

    def test_partial_final_conditions_check_known_conflicts_without_requiring_missing_choices(self):
        base = {'question': 'Final candidate', 'measurements': ['top1'], 'level': 'L3',
                'execution': {'service': 'custom_experiment'}}
        self.assertEqual(preflight_experiment(base, self.brief)['stage'], 'request')
        for fields in ({'quantization_contract': {'specs': {'weight': {'bits': 3}}}},
                       {'calibration': {'count': 3}}):
            with self.subTest(fields=fields), self.assertRaises(ContractError):
                preflight_experiment({**base, 'execution': {**base['execution'], **fields}}, self.brief)
        pending = {**base, 'level': 'L1', 'execution': {'service': 'custom_experiment',
            'resolved_choices': {'target_activation_calibration': {'strategy': 'new_observer'}}}}
        request = preflight_experiment(pending, self.brief)
        self.assertNotIn('target_site', request['execution']['resolved_choices'])
        self.assertEqual(request['stage'], 'request')

    def test_coding_completes_partial_object_arrays_without_rewriting_request(self):
        from unittest.mock import patch
        from tools.experiments import ExperimentBroker
        request = {'id': 'request', 'content_sha256': 'immutable', 'author': 'generation',
            'payload': {'execution': {'service': 'custom_experiment',
                'evaluations': [{'kind': 'classification'}]}}}
        original = deepcopy(request)
        artifact = {'execution_choices': {'evaluations': [
            {'kind': 'classification', 'split': 'search', 'count': 1}]}}
        broker = object.__new__(ExperimentBroker)
        with patch('tools.safeguards.verify_artifact'), \
             patch.object(broker, 'freeze', side_effect=lambda payload, **_: payload):
            bound = broker.bind_execution(request, artifact)
            self.assertEqual(bound['execution']['evaluations'], artifact['execution_choices']['evaluations'])
            self.assertEqual(request, original)
            for rows in ([{'kind': 'another_kind'}], [{'kind': 'classification'}, {'kind': 'extra'}]):
                with self.subTest(rows=rows), self.assertRaisesRegex(ContractError, '冲突'):
                    broker.bind_execution(request, {'execution_choices': {'evaluations': rows}})

    def test_custom_executor_calibration_is_decided_by_actual_implementation(self):
        draft={'question':'Weight-only algorithm','measurements':['score'], 'level':'L2',
               'execution':{'service':'quantize_evaluate','requires_code':True,'parameters':{'alpha':.3},
                            'evaluations':[{'kind':'classification','split':'search','count':1}]}}
        result=preflight_experiment(draft,self.brief,executable=True)['execution']
        self.assertNotIn('calibration',result)
        self.assertEqual(result['parameters'],{'alpha':.3})

    def test_final_candidate_still_rejects_declared_bit_conflict(self):
        draft=experiment_payload('L3')
        draft['execution'].update(service='custom_experiment',requires_code=True,interface='experiment_v1')
        draft['execution']['resolved_choices']['weight_bits']=3
        with self.assertRaisesRegex(ContractError,'weight_bits'):
            preflight_experiment(draft,self.brief,executable=True)

    def test_exploration_still_rejects_its_own_conflicting_scientific_choices(self):
        draft = {'question': 'Local weight error', 'measurements': ['error'],
            'execution': {'service': 'custom_experiment', 'resolved_choices': {'weight_bits': 3},
                'quantization_contract': {'specs': {'weight': {'bits': 4}}}}}
        with self.assertRaisesRegex(ContractError, '显式冲突'):
            preflight_experiment(draft, self.brief, executable=True)

    def test_kind_coverage_waits_for_catalog_and_then_detects_omission(self):
        from unittest.mock import patch
        plan=plan_fixture()
        contract={'required_kinds':['linear']}
        check_plan(plan,{},contract=contract)  # This phase has no backend/catalog.
        catalog={row['module']:{'kind':'linear','protected':False} for row in plan['layers']}
        catalog['missing.layer']={'kind':'linear','protected':False}
        with patch('ptq_agent.execution.sites.site_catalog',return_value=catalog):
            with self.assertRaisesRegex(ContractError,'遗漏'):
                check_plan(plan,{},backend=object(),cfg={},contract=contract)

    def test_custom_plan_new_method_and_observer_are_not_native_capabilities(self):
        plan=plan_fixture()
        plan['layers'][0]['weight_method']='new_source_method'
        plan['layers'][0]['weight_calibration']={'strategy':'new_source_observer','alpha':.3}
        check_plan(plan,{},native=False)
        with self.assertRaisesRegex(ContractError,'Schema'):
            check_plan(plan,{})
        contract={'specs':{'weight':{'bits':3}}}
        with self.assertRaisesRegex(ContractError,'bits'):
            check_plan(plan,{},contract=contract,native=False)
        for row in plan['layers']:
            row['weight'] = None
        with self.assertRaisesRegex(ContractError, '实际执行目标'):
            check_plan(plan, {}, contract=contract, native=False)

    def test_weight_only_required_site_does_not_require_activation_intervention(self):
        plan = plan_fixture()
        plan['layers'] = plan['layers'][:1]
        row = plan['layers'][0]
        row['activation'] = None
        row.pop('activation_method')
        row.pop('activation_calibration', None)
        contract = {'required_sites': [row['module']], 'specs': {'weight': {'bits': 8}}}
        for native in (False, True):
            check_plan(plan, {}, contract=contract, native=native)
            explicit = {**contract, 'required_site_targets': {row['module']: ['weight', 'activation']}}
            with self.assertRaisesRegex(ContractError, '保留 FP'):
                check_plan(plan, {}, contract=explicit, native=native)
        row['weight'] = None
        with self.assertRaisesRegex(ContractError, '量化目标'):
            check_plan(plan, {}, contract={'required_sites': [row['module']]}, native=False)

    def test_custom_executor_can_use_its_own_plan_when_no_structural_conditions_apply(self):
        plan = {'algorithm': 'source_defined', 'basis_rank': 3}
        result = check_plan(plan, {}, contract={'parameters': {'alpha': .3}}, native=False)
        self.assertEqual(result['plan'], plan)
        self.assertEqual(result['covered_sites'], [])
        with self.assertRaisesRegex(ContractError, '可核验'):
            check_plan(plan, {}, contract={'specs': {'weight': {'bits': 3}}}, native=False)
        with self.assertRaisesRegex(ContractError, 'Schema'):
            check_plan(plan, {})
        opaque = {**plan, 'layers': {'custom_groups': [1, 2]}, 'execution': ['source_owned']}
        coverage = check_plan(opaque, {}, contract={'parameters': {'alpha': .3}}, native=False)
        self.assertEqual(coverage['plan'], opaque)
        self.assertEqual(coverage['quantization'], {})

    def test_native_weight_only_plan_has_no_image_calibration_or_unused_parameters(self):
        plan = plan_fixture()
        for row in plan['layers']:
            row['activation'] = None
            row.pop('activation_method')
            row.pop('activation_calibration', None)
        draft = {'question': 'Local weight-only quantization', 'measurements': ['top1'], 'level': 'L2',
            'execution': {'service': 'quantize_evaluate', 'plan': plan,
                'evaluations': [{'kind': 'classification', 'split': 'search', 'count': 1}]}}
        spec = preflight_experiment(draft, self.brief, executable=True)['execution']
        self.assertNotIn('calibration', spec)
        self.assertNotIn('calibration_samples', spec['resolved_choices'])
        from tools.experiment_worker import backend_and_data
        from tools.runtime_interface import ObservedData
        from ptq_agent.execution.model import RTNExecutor
        backend, cfg, data = backend_and_data(spec)
        observed = ObservedData(data)
        executor = RTNExecutor(backend, cfg, observed)
        model, weights, activations, audit = executor.prepare(spec['plan'], {})
        self.assertEqual(set(weights), {row['module'] for row in spec['plan']['layers']})
        self.assertEqual(activations, {})
        self.assertFalse(observed.reads)
        self.assertTrue(all(row['activation_method'] is None for row in spec['plan']['layers']))
        self.assertEqual(executor.audit(model, spec['plan']), audit)
        draft['execution']['parameters'] = {'unimplemented_alpha': .3}
        with self.assertRaisesRegex(ContractError, '不消费'):
            preflight_experiment(draft, self.brief, executable=True)

    def test_no_data_statistics_do_not_claim_calibration_or_batch_protocol(self):
        from tools.result_contract import check_results
        draft = {'level': 'L3', 'execution': {'service': 'custom_experiment'}}
        brief = {'evaluation_protocol': {'batch_size': 8, 'calibration_samples': 32, 'search_samples': 16,
            'required_results': [{'path': 'weight_error', 'schema': {'type': 'number'}}]}}
        data = {'observations': {'weight_error': .01}, 'provenance': {
            'execution_facts': {'measurement_kind': 'weight_statistic', 'data': {}}}}
        check_results(draft, brief, data)
        draft['execution']['calibration'] = {'split': 'calibration', 'count': 32}
        with self.assertRaisesRegex(ContractError, 'calibration_samples'):
            check_results(draft, brief, data)

if __name__=='__main__': unittest.main()
