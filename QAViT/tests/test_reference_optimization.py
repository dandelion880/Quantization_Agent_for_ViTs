"""Existing training contract and isolated CPU custom chain; no PTQ benefit claim."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError
from core.workflow import Workflow
from tests.support import validation_config
from tests.test_infrastructure import TempCase
from tests import test_runtime_gate_cleanup as gates
from tools.experiments import preflight_experiment

SOURCE = '''import torch
def execute(payload, context):
    weight = context['reference_model'].get_submodule('blocks.0.mlp.fc2').weight.detach()
    initial = weight.abs().max().clamp_min(1e-6) / 3
    scale = initial.clone()
    if context['parameters']['reference_optimization']:
        scale.requires_grad_(True)
        optimizer = torch.optim.SGD([scale], lr=.01)
        v = weight / scale
        code = v + (v.round().clamp(-3, 3) - v).detach()
        loss = (code * scale - weight).square().mean()
        loss.backward()
        optimizer.step()
    else:
        # Fresh forward construction from the pre-optimization source, with no oracle input.
        scale = weight.abs().max().clamp_min(1e-6) / 3
    return {'observations': {'initial_scale': float(initial), 'scale': float(scale.detach()),
        'purpose': 'oracle-assisted' if context['parameters']['reference_optimization'] else 'independent_forward'}}
'''


def draft(training):
    return {'level': 'L2', 'question': 'Isolated reference scale optimization' if training else 'Independent forward reconstruction from Q0',
            'measurements': ['scale'], 'controls': {'scope': 'CPU synthetic mechanics only; source model stays frozen'},
            'resources': {'estimated_seconds': 30},
            'execution': {'service': 'custom_experiment', 'requires_code': True,
                          'resolved_choices': {'training': training}}}


class ReferenceOptimization(TempCase):
    begin = gates.CustomWorkerTests.begin
    progress = gates.CustomWorkerTests.progress
    approve = gates.CustomWorkerTests.approve
    execute = gates.CustomWorkerTests.execute

    def test_frozen_prohibition_cannot_be_overridden_by_reference_label(self):
        for constraints in ({'training': False}, {'training_allowed': False}):
            brief = validation_config()['research_brief']
            brief['hard_constraints'] = constraints
            brief['search_space']['notes'] = 'Skill suggests an oracle; this does not change frozen permission'
            before = deepcopy(brief)
            for executable in (False, True):
                with self.assertRaises(ContractError):
                    preflight_experiment(draft(True), brief, executable=executable)
            self.assertEqual(brief, before)

    def test_allowed_reference_and_fresh_forward_rebuild_use_existing_worker(self):
        results = []
        for training in (True, False):
            config = validation_config()
            config['budget'].update(max_rounds=1, min_free_bytes=0)
            config['research_brief']['hard_constraints'] = {'training_allowed': True} if training else {'training': False}
            config['research_brief']['goals'] = {'scope': 'Isolated scale-only research reference permitted; final method reconstructed independently'}
            wf = Workflow(self.root / ('oracle' if training else 'independent'), config)
            tid = wf.enqueue_role('check/generate', 'generate', refs=[wf.brief['id']],
                context={'round': 1, 'slot': 'H3', 'hypothesis_type': 'new_algorithm'})
            researcher, _ = self.begin(wf, wf.queue.get(tid))
            wf.receive_tools(researcher, {'tool_calls': [{'id': 'experiment', 'name': 'request_experiment',
                'arguments': {'experiment': draft(training)}}]}, 'SCRIPTED_CPU')
            wf.queue.ready()
            request = wf.queue.claim(wf.queue.get(tid)['waiting_for'][0])
            wf.progress_request(request)
            request = wf.queue.get(request['task_id'])
            code, _ = self.begin(wf, wf.tasks()[request['key'] + '/code'])
            wf.submit(code, {'source': SOURCE, 'execution_choices': {
                'parameters': {'reference_optimization': training},
                'resolved_choices': {'training': training}}}, 'SCRIPTED_CPU')
            try:
                request = self.approve(wf, self.progress(wf, request))
                completed = self.execute(wf, request)
                self.assertEqual(completed['state'], 'completed', completed.get('error'))
                result = wf.journal.card(completed['outputs'][0])['payload']
                self.assertEqual(result['measurement_source'], 'synthetic_computation')
                self.assertEqual(result['provenance']['training_declared'], training)
                self.assertEqual(result['provenance']['gradient_audit']['optimizer_steps'], int(training))
                self.assertTrue(result['provenance']['model_unchanged'])
                self.assertEqual(result['provenance']['execution_facts']['frozen_loads'], [])
                request = self.progress(wf, request)
                self.assertEqual(request['result']['evidence_ref'], completed['outputs'][0])
                resumed, _ = self.begin(wf, wf.queue.get(tid))
                self.assertIn(completed['outputs'][0], wf.required_refs(resumed))
                results.append(result)
            finally:
                wf.runtime.stop()
        oracle, independent = results
        self.assertEqual(oracle['provenance']['baseline_state_sha256'], independent['provenance']['baseline_state_sha256'])
        self.assertNotEqual(oracle['observations']['scale'], oracle['observations']['initial_scale'])
        self.assertEqual(independent['observations']['scale'], oracle['observations']['initial_scale'])


if __name__ == '__main__':
    unittest.main()
