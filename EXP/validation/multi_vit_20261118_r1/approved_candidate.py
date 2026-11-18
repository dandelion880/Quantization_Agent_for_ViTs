from copy import deepcopy
from ptq_agent.execution.baselines import build_baseline
from ptq_agent.execution.recipes import spec


def build_plan(backend, cfg, experiment_payload, parameters):
    plan = deepcopy(build_baseline(backend, 'repq', quantize_head=False))
    matches = [row for row in plan['layers']
               if row['module'] == 'blocks.0.mlp.fc2'
               and row['kind'] == 'linear']
    if len(matches) != 1:
        raise ValueError('Expected exactly one linear row for blocks.0.mlp.fc2')

    row = matches[0]
    row['weight'] = spec(4, False, 'per_channel', 0)
    row['activation'] = spec(4, False, 'per_channel', 2)
    row['weight_method'] = 'uniform_affine'
    row['activation_method'] = 'uniform_affine'
    row['weight_calibration'] = {'strategy': 'minmax'}
    row['activation_calibration'] = {'strategy': 'minmax'}
    row['options'] = {}

    plan['layers'] = [row]
    plan['preprocess'] = []
    plan['operators'] = []
    plan['unlisted_modules'] = 'keep_fp'
    return plan
