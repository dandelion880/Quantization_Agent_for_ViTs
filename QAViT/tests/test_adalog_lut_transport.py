"""Frozen AdaLog LUT validation must survive CPU/CUDA transport without mutation."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.paths import QAVIT
from ptq_agent.execution import adalog
from ptq_agent.schemas import PTQError
import torch


class AdaLogLUTTransport(unittest.TestCase):
    def test_real_cuda_tables_validate_on_cpu_without_mutation(self):
        fixture = json.loads((QAVIT / 'tests/data/adalog_cuda_lut_float32.json').read_text())
        for row in fixture['cases']:
            with self.subTest(q=row['base_q']):
                params = adalog.parameters(torch.tensor(1.), 4, row['base_q'])
                params['table1'] = torch.tensor(row['table1'])
                params['table2'] = torch.tensor(row['table2_int32'], dtype=torch.int32).view(torch.float32)
                before = {k: v.clone() for k, v in params.items()}
                x = torch.linspace(0, 1, 257)
                output = adalog.apply(x, {'bits': 4}, params, {'reserve_zero': True})
                adalog.validate_params(params, {'bits': 4})
                for key, value in before.items():
                    self.assertTrue(torch.equal(value, params[key]))
                self.assertTrue(torch.equal(output, adalog.apply(x, {'bits': 4}, params, {'reserve_zero': True})))

    def test_lut_corruption_beyond_roundoff_and_wrong_base_are_rejected(self):
        for change in ('two_ulp', 'integer_table', 'base', 'shape', 'nonfinite'):
            params = adalog.parameters(torch.tensor(1.), 4, 19)
            if change == 'two_ulp':
                value = params['table2'][1]
                params['table2'][1] = torch.nextafter(torch.nextafter(value, value.new_tensor(float('inf'))), value.new_tensor(float('inf')))
            elif change == 'integer_table':
                params['table1'][1] += 1
            elif change == 'base':
                params['base_q'] += 1
            elif change == 'shape':
                params['table2'] = params['table2'].reshape(4, 4)
            else:
                params['table2'][1] = float('nan')
            with self.subTest(change=change), self.assertRaises(PTQError):
                adalog.validate_params(params, {'bits': 4})

    def test_existing_canonical_cpu_params_still_validate(self):
        for dtype in (torch.float32, torch.float64):
            for q in (1, 19, 37, 55, 74, 256):
                with self.subTest(dtype=dtype, q=q):
                    adalog.validate_params(adalog.parameters(torch.tensor(1., dtype=dtype), 4, q), {'bits': 4})

    @unittest.skipUnless(torch.cuda.is_available(), 'requires an explicitly available CUDA device')
    def test_complete_frozen_adalog_package_loads_from_cpu_to_cuda(self):
        from tempfile import TemporaryDirectory
        from tools.runtime_interface import interface_fixture
        from tests.support import experiment_payload
        from ptq_agent.execution.baselines import build_baseline
        from ptq_agent.execution.model import RTNExecutor
        from ptq_agent.execution.pipeline import quantizers
        payload = experiment_payload()
        payload['execution']['backend'] = {'config_path': 'QAViT/configs/deitb_backend.json'}
        backend, config, data = interface_fixture(payload['execution'], {
            'shape': [1, 3, 16, 16],
            'sites': ['patch_embed.proj', 'blocks.0.attn.qkv', 'blocks.0.mlp.fc2'],
            'seed': 3, 'counts': {'calibration': 2, 'search': 1, 'final': 1}})
        backend.identity = lambda: {'fixture': 'adalog_cpu_cuda_transport', 'state': backend.original_state}
        plan = build_baseline(backend, 'adalog')
        for row in plan['layers']:
            if row.get('activation_method') == 'adalog':
                row['options']['log_base_candidates'] = [19]
        executor = RTNExecutor(backend, config, data)
        model, weights, activations, audit = executor.prepare(plan, {})
        with TemporaryDirectory(prefix='adalog-package-') as directory:
            folder = Path(directory) / 'model'
            executor.export(model, plan, weights, activations, folder, audit)
            tensors = torch.load(folder / 'weights.pt', map_location='cpu', weights_only=True)
            backend.device = 'cuda:0'
            backend.model = backend.model.to(backend.device)
            loaded, _ = RTNExecutor(backend, config, data).load(folder)
            tables = {sid: params['table2'] for sid, params in tensors['activations'].items()
                      if 'table2' in params}
            self.assertTrue(tables, 'fixture must exercise serialized AdaLog tables')
            for sid, table in tables.items():
                self.assertTrue(torch.equal(quantizers(loaded)[sid].params()['table2'].cpu(), table))
            inputs, _ = next(data.batches('search'))
            with torch.inference_mode():
                self.assertTrue(torch.isfinite(loaded(inputs.cuda())).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'requires an explicitly available CUDA device')
    def test_real_bidirectional_transport_preserves_parameters_and_inference(self):
        for q in (19, 37, 55, 74):
            for origin, destination in (('cpu', 'cuda:0'), ('cuda:0', 'cpu')):
                with self.subTest(q=q, origin=origin, destination=destination):
                    params = {k: v.to(destination) for k, v in
                              adalog.parameters(torch.tensor(1., device=origin), 4, q).items()}
                    before = {k: v.detach().cpu().clone() for k, v in params.items()}
                    x = torch.linspace(0, 1, 257, device=destination)
                    output = adalog.apply(x, {'bits': 4}, params, {'reserve_zero': True})
                    adalog.validate_params(params, {'bits': 4})
                    self.assertTrue(all(torch.equal(value, params[key].cpu()) for key, value in before.items()))
                    self.assertTrue(torch.equal(output, adalog.apply(x, {'bits': 4}, params, {'reserve_zero': True})))


if __name__ == '__main__':
    unittest.main()
