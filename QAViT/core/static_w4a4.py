"""Static W4A4 boundaries, independent of quantizer names and research strategy."""
from core.cards import ContractError

from core.ptq_protocol import LEGACY, PRIMARY, EXTENDED

PROTOCOL = PRIMARY
DESCRIPTION = ('模型主体 Linear/Conv 主权重 W4、输入 A4；Attention Q/K/V、用于 PV 的 P、'
    '用于 FC2 的 post-GELU 均 A4。每个固定组最多16码字（零计入），推理前冻结。'
    '允许 LN/Softmax/GELU 内部、gamma/beta、bias、CLS/位置编码、残差及累加高精度。'
    'head/neck 可浮点并记录；粒度、对称性、校准、量化器、重参数化和补偿由研究选择。'
    '[-8,7]与[-7,7]均合法；局部诊断按本次范围，不要求全模型覆盖。')
HEADS = {'head', 'head_dist', 'neck', 'decode_head', 'auxiliary_head', 'classifier'}


def task_module(name):
    return bool(set(name.split('.')) & HEADS)


def check_plan(plan, catalog, *, native=True, protocol=PROTOCOL, extensions=()):
    """Check computation boundaries; equivalent graph edges need only one quantizer."""
    from ptq_agent.execution.plugin_contracts import site_id
    rows = {site_id(row): row for row in plan.get('layers', [])}
    extensions = set(extensions)
    mode = plan.get('execution', {}).get('activation_calibration')
    if mode != 'static' and 'dynamic_quantization' not in extensions:
        raise ContractError('static W4A4 requires frozen static activation parameters')
    required = {}
    exceptions = {}
    if plan.get("postfit") and "additional_computation" not in extensions:
        if any(stage["kind"] in ("qwt", "channel_affine", "bias_compensation") and not (native and stage["kind"]=="channel_affine" and stage.get("settings",{}).get("integer_fold")) for stage in plan["postfit"]):
            raise ContractError("retained output compensation needs an additional_computation profile")
    if any(s["kind"]=="outlier_tokens" for s in plan.get("structure",[])) and "additional_computation" not in extensions:
        raise ContractError("online token selection and coordinate recovery require declared structural cost")
    if any(s["kind"]=="window_branches" for s in plan.get("structure",[])) and not {"mixed_precision","dynamic_quantization"}<=extensions:
        raise ContractError("per-window mixed activations need dynamic and mixed precision dimensions")
    if any(s["kind"]=="calibrated_operator" and s["settings"].get("output_layout")=="instance_tokens" for s in plan.get("structure",[])) and "dynamic_quantization" not in extensions:
        raise ContractError("instance-dependent token scales require a dynamic profile")
    for row in rows.values():
        if row.get("weight_method")=="hc_rows" and not {"weight_only","coverage_exceptions"} <= extensions:
            raise ContractError("whole-row indices with FP32 vector codebooks require explicit weight-only/coverage dimensions")
        name=row.get("activation_method")
        cfg=row.get("options",{}).get("encoding",{})
        if name=="patch_precision" and not {"mixed_precision","dynamic_quantization"}<=extensions:
            raise ContractError("runtime patch widths require both dynamic and mixed-precision dimensions")
        if (name=="domain_clip" and cfg.get("range_mode")=="instance_tokens" or name=="igq_ranges" or name=="shift_log" and cfg.get("shift_mode")=="runtime_min") and "dynamic_quantization" not in extensions:
            raise ContractError("input-dependent group or offset requires dynamic_quantization")
        if name=="adfq_outliers" and not {"additional_computation","coverage_exceptions"} <= extensions:
            raise ContractError("floating outlier values reach the matrix path and require explicit extended coverage and cost")
        if name=="shift_log" and not cfg.get("reserve_underflow") and "coverage_exceptions" not in extensions:
            raise ContractError("additional underflow state exceeds the declared activation bit width")
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
            specs = [r[target] for r in peers if r.get(target)]
            bits = {s['bits'] for s in specs}
            if protocol == EXTENDED and bits != {4}:
                if not bits and ('coverage_exceptions' in extensions or target == 'activation' and 'weight_only' in extensions):
                    exceptions[sid + '/' + target] = 'unquantized'
                    continue
                if len(bits) == 1 and extensions & {'mixed_precision', 'coverage_exceptions', 'weight_only'}:
                    required[sid + '/' + target] = bits.pop()
                    continue
            if 4 not in bits or protocol != LEGACY and bits != {4}:
                raise ContractError('static W4A4 missing ' + sid + '/' + target)
            required[sid + '/' + target] = 4
    # The fc2 input edge is the post-GELU computation boundary. No duplicate QDQ.
    edges = {catalog[key.rsplit('/', 1)[0]]['edge'] for key in required if key.endswith('/activation')}
    for sid, row in rows.items():
        if not native or task_module(row['module']) or catalog[sid]['edge'] not in edges:
            continue
        if native and row.get('activation') and row.get('activation_method') in ('log2', 'log_sqrt2', 'adalog') and 'coverage_exceptions' not in extensions:
            if not row.get('options', {}).get('reserve_zero', False):
                raise ContractError('original logarithmic grid can contain 17 values including zero; select reserve_zero or adapt the quantizer')
    return {'protocol': protocol, 'required_boundaries': sorted(required),
            **({'required_bits': required, 'unquantized_edges': exceptions, 'extension_dimensions': sorted(extensions),
                'activation_calibration': mode} if protocol != LEGACY else {}),
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
        bits = required.get('required_bits', {}).get(key, 4)
        count = 2 ** bits
        if target == 'activation' and plan.get('schema_version') != 2:
            owner = next(k for k, row in rows.items() if row.get('activation') and catalog[k]['edge'] == catalog[sid]['edge'])
            params = qmap[owner].params()
            if 'repq_affine' in params and bool((params['zero'] != params['zero'].round()).any()):
                count = 2 ** bits + 1
        quant_row=rows.get(key.rsplit("/",1)[0],{})
        includes_zero=True
        if key.endswith("/activation"):
            key_site=key.rsplit("/",1)[0]
            actual=next((r for s,r in rows.items() if r.get("activation") and catalog[s]["edge"]==catalog[key_site]["edge"]),None)
            if actual:
                codec=actual["activation_method"]
                options=actual.get("options",{})
                if codec=="patch_precision":
                    capacities[key]={"bits":bits,"max_codewords":2**(bits+2),"includes_zero":True,"runtime_widths":[bits-2,bits,bits+2]}
                    continue
                if codec=="focus_interval":includes_zero=bool(qmap[site_id(actual)].params()["low"]==0)
                if codec in ("log2","log_sqrt2","adalog") and not options.get("reserve_zero"):count=2**bits+1
                if codec=="shift_log" and not options["encoding"]["reserve_underflow"]:count=2**bits+1
                if codec in ("tanq","sulq","bit_window","shift_log"):
                    includes_zero=codec=="bit_window"
                    if codec in ("tanq","sulq","shift_log"):
                        from ptq_agent.execution.encodings import levels
                        table=levels(quantizers(model)[site_id(actual)].params(),actual)
                        includes_zero=bool((table==0).any())
        capacities[key] = {'bits': bits, 'max_codewords': count, 'includes_zero': includes_zero}
        if quant_row.get("weight_method")=="hc_rows" and key.endswith("/weight"):
            record = qmap[sid].weight_record
            capacities[key].update(unit="index_per_weight_row",codebook_dtype="float32",
                                   codebook_shape=list(record["params"]["codebook"].shape),index_bytes=record["data"].numel())
        if key.endswith("/activation") and actual and actual["activation_method"]=="adfq_outliers":
            capacities[key].update(high_precision_path="floating outlier bypass",main_payload_bits=bits,
                                   threshold=actual["options"]["encoding"]["threshold"])
        if key.endswith("/activation") and actual and actual["activation_method"] in ("igq_ranges","two_channel_scales"):
            capacities[key].update(group_assignment="runtime_range" if actual["activation_method"]=="igq_ranges" else "fixed_binary_channel_membership")

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
        bits = required.get('required_bits', {}).get(key, 4)
        zero = row.get('includes_zero')
        if (type(n) is not int or n<1 or (n>2**bits and "coverage_exceptions" not in required.get("extension_dimensions",[]) and not ({"dynamic_quantization","mixed_precision"}<=set(required.get("extension_dimensions",[])) and row.get("runtime_widths")==[bits-2,bits,bits+2] and n==2**(bits+2))) or row.get('bits') != bits
                or type(zero) is not bool or required.get('protocol', LEGACY) == LEGACY and not zero):
            raise ContractError('static W4A4 invalid group capacity (zero included): ' + key)
    return {**required, 'frozen': True, 'boundaries': rows}
