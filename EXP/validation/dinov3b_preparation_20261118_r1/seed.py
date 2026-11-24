"""A fixed training-free comparison grounded in FQ-ViT log2 P and groupwise RTN W."""
from ptq_agent.execution.baselines import build_baseline

def build(backend):
    plan = build_baseline(backend, 'gptaq')
    plan['execution'].update(order='joint')
    for row in plan['layers']:
        if row['kind'] == 'linear':
            row['weight_method'] = 'rtn'
            row['weight'].update(granularity='per_group', group_size=32, axis=1)
            row['options'] = {}
        if row['kind'] == 'post_softmax':
            row['activation_method'] = 'log2'
            row['activation_calibration'] = {'strategy':'minmax'}
        elif row.get('activation'):
            row['activation_method'] = 'uniform_affine'
            row['activation_calibration'] = {'strategy':'percentile', 'percentiles':[.999]}
    return plan


def partitioned(backend):
    plan = build(backend)
    spec = backend.spec
    assert spec['cls_token_count'] == 1 and spec['special_token_count'] == 4
    for row in plan['layers']:
        kind = row['kind']
        if kind in ('linear','attention_query','attention_key','attention_value'):
            axis = 3 if kind == 'attention_key' else 1 if kind == 'linear' else 2
            row['options']['token_partition'] = {'axis':axis, 'prefix_ends':[spec['cls_token_count'], spec['prefix_token_count']]}
            row['activation_calibration']['percentiles'] = [1., .999, .99]
    return plan
