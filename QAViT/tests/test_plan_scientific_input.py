"""Scientific plans need numeric choices, not repetitive prose or invented citations."""
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.execution_contract import validate_plan_schema
from tests.test_execution_contracts import plan_fixture
from tests.test_infrastructure import TempCase
from tests.support import validation_config, experiment_payload
from tools.experiment_worker import backend_and_data
from tools.experiments import normalize_execution

class PlanScientificInput(TempCase):
    def test_missing_per_site_explanation_and_empty_sources_are_accepted(self):
        plan = plan_fixture()
        for row in plan['layers']:
            row.pop('reason')
            row.pop('evidence')
        validate_plan_schema(plan)
        for row in plan['layers']:
            row.update(reason='', evidence=[])
        validate_plan_schema(plan)
        bad = deepcopy(plan)
        bad['layers'][0]['weight_method'] = 'unimplemented_native_method'
        from core.cards import ContractError
        with self.assertRaises(ContractError):
            validate_plan_schema(bad)

    def test_fp_control_plan_exports_reloads_without_quantization(self):
        import torch
        from ptq_agent.execution.pipeline import QuantizationExecutor
        from ptq_agent.execution.evaluation import model_digest
        execution = normalize_execution(experiment_payload()['execution'], validation_config()['research_brief'])
        backend, cfg, data = backend_and_data(execution)
        plan = plan_fixture()
        plan['layers'] = []
        validate_plan_schema(plan)
        executor = QuantizationExecutor(backend, cfg, data)
        original = model_digest(backend.model)
        model, weights, activations, audit = executor.prepare(plan, {})
        self.assertEqual(model_digest(model), original)
        self.assertEqual(weights, {})
        self.assertEqual(activations, {})
        directory = self.root / 'fp_control'
        executor.export(model, plan, weights, activations, directory, audit)
        restored, manifest = executor.load(directory)
        self.assertEqual(manifest['plan']['layers'], [])
        x, _ = next(data.batches('search'))
        with torch.no_grad():
            self.assertTrue(torch.equal(backend.model(x), restored(x)))
