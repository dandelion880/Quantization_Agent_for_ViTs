"""Complete, editable starting configurations. Adaptations are not paper reproductions."""
from copy import deepcopy
from .recipes import spec
from .sites import site_catalog
from ..schemas import require

NAMES = ('gptaq', 'adalog', 'repq')


def build_baseline(backend, name, *, quantize_head=False):
    require(name in NAMES, 'unknown complete baseline')
    plan = {'schema_version': 3, 'unlisted_modules': 'keep_fp', 'unresolved': [], 'preprocess': [],
        'execution': {'mode': 'reference_qdq', 'compute_dtype': 'FP32', 'activation_calibration': 'static',
                      'order': 'a_then_w' if name == 'gptaq' else 'joint', 'within_stage': 'sequential'}, 'layers': []}
    percentile = {'strategy': 'percentile', 'percentiles': [1.0, .9999, .999, .99]}
    for sid, info in site_catalog(backend).items():
        module, kind = info['name'], info['kind']
        if any(x in module.split('.') for x in ('head', 'head_dist', 'neck', 'decode_head', 'auxiliary_head', 'classifier')):
            if not (quantize_head and module == 'head' and kind == 'linear'):
                continue
        if kind not in ('linear', 'conv2d', 'attention_query', 'attention_key', 'attention_value', 'post_softmax'):
            continue
        weight = spec(4, name == 'gptaq', 'per_channel', 0) if kind in ('linear', 'conv2d') else None
        wm = ('gptaq' if kind == 'linear' else 'uniform_minmax') if name == 'gptaq' else 'uniform_affine'
        a = spec(4, False)
        am, ac, options = 'uniform_affine', deepcopy(percentile), {}
        if name == 'gptaq' and kind != 'post_softmax':
            a.update(granularity='per_channel', axis=1 if kind == 'conv2d' else 2 if kind in ('linear', 'attention_key') else 3)
        if kind == 'post_softmax':
            am = 'adalog' if name == 'adalog' else 'log_sqrt2'
            options['reserve_zero'] = True
            if am == 'adalog':
                ac = {'strategy': 'search', 'metric': 'mse', 'factors': [.5, .75, 1., 1.25]}
                options['log_base_candidates'] = [19, 37, 55, 74]
        if name == 'repq' and info.get('semantic') == 'post_layernorm':
            plan['preprocess'].append({'method': 'repq_ln', 'module': module,
                'norm': info['edge'].removesuffix(':output'), 'observation': 'float', 'calibration': deepcopy(percentile)})
        if name == 'adalog' and info.get('semantic') == 'post_gelu':
            am = 'adalog'
            ac = {'strategy': 'search', 'metric': 'mse', 'factors': [.5, .75, 1., 1.25]}
            options.update(reserve_zero=True, log_base_candidates=[19, 37, 55, 74])
            plan['preprocess'].append({'method': 'adalog_bias', 'module': module, 'norm': None})
        if name == 'gptaq' and kind == 'linear':
            options.update(blocksize=128, damping=.01, actorder=False, gptaq_clip=False)
        row = {'module': module, 'kind': kind, 'weight': weight, 'activation': a,
            'weight_method': wm if weight else None, 'activation_method': am, 'activation_calibration': ac,
            'options': options}
        if weight:
            row['weight_calibration'] = {'strategy': 'minmax'} if name == 'gptaq' else deepcopy(percentile)
        plan['layers'].append(row)
    return plan
