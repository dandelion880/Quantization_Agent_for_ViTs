"""Static W4A4 boundaries, independent of quantizer names and research strategy."""
from core.cards import ContractError

PROTOCOL = 'static_w4a4/1'
DESCRIPTION = ('模型主体 Linear/Conv 主权重 W4、输入 A4；Attention Q/K/V、用于 PV 的 P、'
    '用于 FC2 的 post-GELU 均 A4。每个固定组最多16码字（零计入），推理前冻结。'
    '允许 LN/Softmax/GELU 内部、gamma/beta、bias、CLS/位置编码、残差及累加高精度。'
    'head/neck 可浮点并记录；粒度、对称性、校准、量化器、重参数化和补偿由研究选择。'
    '[-8,7]与[-7,7]均合法；局部诊断按本次范围，不要求全模型覆盖。')
HEADS = {'head', 'head_dist', 'neck', 'decode_head', 'auxiliary_head', 'classifier'}


def task_module(name):
    return bool(set(name.split('.')) & HEADS)


def check_plan(plan, catalog, *, native=True):
    """Check computation boundaries; equivalent graph edges need only one quantizer."""
    from ptq_agent.execution.plugin_contracts import site_id
    rows = {site_id(row): row for row in plan.get('layers', [])}
    if plan.get('execution', {}).get('activation_calibration') != 'static':
        raise ContractError('static W4A4 requires frozen static activation parameters')
    required = {}
    for sid, info in catalog.items():
        if task_module(info['name']):
            continue
        kind = info['kind']
        targets = ('weight', 'activation') if kind in ('linear', 'conv2d') else (
            ('activation',) if kind in ('attention_query', 'attention_key', 'attention_value', 'post_softmax') else ())
        for target in targets:
            row = rows.get(sid, {})
            # A quantizer on the same validated semantic edge covers the input.
            peers = [row]
            if target == 'activation':
                peers += [r for k, r in rows.items() if k in catalog and catalog[k]['edge'] == info['edge']]
            if not any(r.get(target, {}).get('bits') == 4 for r in peers if r.get(target)):
                raise ContractError('static W4A4 missing ' + sid + '/' + target)
            required[sid + '/' + target] = 4
    # The fc2 input edge is the post-GELU computation boundary. No duplicate QDQ.
    edges = {catalog[key.rsplit('/', 1)[0]]['edge'] for key in required if key.endswith('/activation')}
    for sid, row in rows.items():
        if not native or task_module(row['module']) or catalog[sid]['edge'] not in edges:
            continue
        if native and row.get('activation') and row.get('activation_method') in ('log2', 'log_sqrt2', 'adalog'):
            if not row.get('options', {}).get('reserve_zero', False):
                raise ContractError('original logarithmic grid can contain 17 values including zero; select reserve_zero or adapt the quantizer')
    return {'protocol': PROTOCOL, 'required_boundaries': sorted(required),
            'task_modules': {sid: {t: rows.get(sid, {}).get(t) for t in ('weight', 'activation')}
                for sid, info in catalog.items() if task_module(info['name']) and info['kind'] in ('linear', 'conv2d')}}


def native_audit(model, plan, catalog, required):
    """Capacity from frozen native grids, never from sampled unique values.

    Native packing and ordinary activation grids have <= 2**bits values. The
    log underflow case is checked by check_plan; RepQ's fractional mean zero
    adds two saturated endpoints to fifteen interior values and needs checking
    after fitting. Internal high precision operations are outside these edges.
    """
    from ptq_agent.execution.pipeline import quantizers
    from ptq_agent.execution.plugin_contracts import site_id
    rows = {site_id(row): row for row in plan['layers']}
    qmap = quantizers(model)
    capacities = {}
    for key in required['required_boundaries']:
        sid, target = key.rsplit('/', 1)
        count = 16
        if target == 'activation' and plan.get('schema_version') != 2:
            owner = next(k for k, row in rows.items() if row.get('activation') and catalog[k]['edge'] == catalog[sid]['edge'])
            params = qmap[owner].params()
            if 'repq_affine' in params and bool((params['zero'] != params['zero'].round()).any()):
                count = 17
        capacities[key] = {'bits': 4, 'max_codewords': count, 'includes_zero': True}
    return check_audit({'frozen': True, 'boundaries': capacities}, required)


def check_audit(report, required):
    """Custom executors report actual frozen group capacities through their reviewed audit.

    This is a numerical receipt, not a proof from prose or method spelling. Host
    reload and model-state invariance checks remain independently mandatory.
    """
    if not isinstance(report, dict) or report.get('frozen') is not True:
        raise ContractError('static W4A4 audit lacks frozen quantizer state')
    rows = report.get('boundaries', {})
    for key in required['required_boundaries']:
        row = rows.get(key, {})
        n = row.get('max_codewords')
        if (type(n) is not int or not 1 <= n <= 16 or row.get('bits') != 4
                or row.get('includes_zero') is not True):
            raise ContractError('static W4A4 invalid group capacity (zero included): ' + key)
    return {**required, 'frozen': True, 'boundaries': rows}
