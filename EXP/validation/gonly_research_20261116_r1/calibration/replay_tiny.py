"""Actual CPU tiny Transformer, synthetic images: calibration mechanics only."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import sys
import platform

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'QAViT'))
from tests.test_calibration_search import fixture
from ptq_agent.execution.pipeline import QuantizationExecutor
from ptq_agent.execution.calibration_search import search_frozen_scales
from ptq_agent.execution.model import forward
import torch


def main():
    backend, config, data, plan = fixture()
    executor = QuantizationExecutor(backend, config, data)
    model, _, _, audit = executor.prepare(plan, {})
    data.labels.update({sid: int(forward(backend, backend.model, x).argmax(-1).item())
                        for x, sid in data.batches('calibration')})
    candidates = [{'id': str(f), 'scales': {'blocks.0.attn.qkv': {'activation': f}}}
                  for f in (.4, .6, .8, 1., 1.2, 1.4)]
    rows = []
    for metric in ('mse', 'cosine', 'teacher_kl', 'task_loss', 'top1_error', 'hessian_diag_ce'):
        rows.append(search_frozen_scales(backend, deepcopy(model), data, candidates, metric=metric))
    for reference in ('fp_model', 'local_fp'):
        for context in ('quantized',) if reference == 'local_fp' else ('quantized', 'float'):
            for tokens in ('all', 'cls', 'patch'):
                rows.append(search_frozen_scales(backend, deepcopy(model), data, candidates,
                            block='blocks.0', reference=reference, context=context, token_scope=tokens))
    files = ['pipeline.py', 'calibration.py', 'calibration_search.py', 'plugin_contracts.py']
    hashes = {f: hashlib.sha256((ROOT/'QAViT/tools/ptq_agent/execution'/f).read_bytes()).hexdigest() for f in files}
    result = {'scope': 'actual PyTorch tiny random Transformer; synthetic images and FP-argmax fixture labels; not scientific task accuracy',
              'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu', 'threads': torch.get_num_threads()},
              'seed': 7, 'calibration_samples': 3, 'image_shape': [1,3,32,32], 'depth': 2,
              'embed_dim': 24, 'num_heads': 3, 'source_sha256': hashes, 'plan': plan, 'audit': audit,
              'reports': rows}
    (Path(__file__).parent/'tiny_numerical.json').write_text(json.dumps(result, indent=2)+'\n')
    for r in rows:
        print(r['metric'], r['unit'], r['context'], r['reference'], r['token_scope'], r['selected_id'], r['selected_score'])


if __name__ == '__main__':
    main()
