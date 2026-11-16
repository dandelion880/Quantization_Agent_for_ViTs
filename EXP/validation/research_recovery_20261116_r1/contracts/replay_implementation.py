import torch
import random
from itertools import islice
from ptq_agent.execution.model import tensor_digest
from ptq_agent.execution.runtime_state import target_signature
from ptq_agent.schemas import fingerprint

SOLUTION = 'c1c5ac43b4957e8afca2e8c4caca8f3cd1d3cc75eb72e0c8529bf2d00b6a27db'
FROZEN_EVIDENCE = 'EvidenceCard-46fbe7b34ad44eea8f72d04756d69074@1'
SITES = ['blocks.0.attn', 'blocks.5.attn', 'blocks.11.attn']
ARMS = ['native', 'nearest', 'spatial_pair', 'shuffled_pair', 'lambda_one']

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sample_id(value):
    if isinstance(value, (tuple, list)):
        require(len(value) == 1, 'Expected one sample ID per batch')
        return sample_id(value[0])
    if torch.is_tensor(value):
        require(value.numel() == 1, 'Non-scalar sample ID')
        return value.item()
    require(isinstance(value, (str, int, float)), 'Unsupported sample ID representation')
    return value

def signature(model):
    state = {k: tensor_digest(v) for k, v in model.state_dict().items()}
    targets = {}
    for name, q in model.named_modules():
        if hasattr(q, 'layer') and hasattr(q, 'params') and hasattr(q, 'enabled'):
            row = {'activation': target_signature(q, 'activation')}
            if hasattr(q, 'decoded_weight'):
                row['weight'] = target_signature(q, 'weight')
            targets[name] = row
    return fingerprint({'state': state, 'targets': targets,
                        'training': {n: m.training for n, m in model.named_modules()},
                        'requires_grad': {n: p.requires_grad for n, p in model.named_parameters()}})

def pair_indices(order):
    n = len(order) // 2 * 2
    return {'a': order[:n:2], 'b': order[1:n:2], 'unpaired_patch': order[n:]}

def make_pairings(model):
    embed = model.patch_embed
    require(hasattr(embed, 'grid_size'), 'Adapter has no actual patch grid_size')
    grid = tuple(int(v) for v in embed.grid_size)
    require(len(grid) == 2 and min(grid) > 0, 'Invalid actual patch grid')
    require(getattr(model, 'dist_token', None) is None, 'Distilled model is not authorized')
    if hasattr(model, 'num_prefix_tokens'):
        require(model.num_prefix_tokens == 1, 'Expected exactly one CLS prefix token')
    if hasattr(embed, 'num_patches'):
        require(int(embed.num_patches) == grid[0] * grid[1], 'Patch grid metadata mismatch')
    order = []
    for r in range(grid[0]):
        columns = range(grid[1]) if r % 2 == 0 else range(grid[1] - 1, -1, -1)
        order.extend(1 + r * grid[1] + c for c in columns)
    shuffled = list(order)
    random.Random(42).shuffle(shuffled)
    return grid, {'spatial': pair_indices(order), 'shuffled': pair_indices(shuffled)}

def actual_codebook(q):
    require(q.enabled and q.execution['activation_calibration'] == 'static', 'P quantizer is not frozen/static/enabled')
    spec, options, p = q.layer['activation'], q.layer.get('options', {}), q.params()
    require(spec['bits'] == 4 and q.layer['activation_method'] == 'adalog', 'Frozen P implementation is not the supported actual AdaLog contract')
    require(bool(options.get('reserve_zero', False)), 'Frozen P quantizer does not reserve zero')
    require(set(p) == {'scale', 'base_q', 'shift', 'table1', 'table2'}, 'Unexpected frozen P parameter contract')
    require(p['scale'].numel() == 1 and p['base_q'].numel() == 1 and p['shift'].numel() == 1, 'Expected actual scalar P LUT group')
    require(float(p['shift'].item()) == 0.0, 'P shift must be zero')
    require(p['table1'].numel() == 16 and p['table2'].numel() == 16, 'Frozen P LUT must have 16 entries')
    require(p['scale'].dtype == torch.float32 and p['table1'].dtype == torch.float32 and p['table2'].dtype == torch.float32, 'Expected actual FP32 P decoding')
    # Decode stored LUT terms in the actual decoder multiplication order.
    # Index 15 is reserved for zero; no idealized base is substituted.
    decoded = 2.0 ** (-p['table1'][:15]) * p['table2'][:15] * p['scale']
    c = torch.unique(torch.cat((decoded.reshape(-1), decoded.new_zeros(1))), sorted=True)
    require(0 < c.numel() <= 16 and bool(torch.isfinite(c).all()) and bool((c >= 0).all()) and bool((c == 0).any()), 'Illegal actual decoded P codebook')
    return c.detach().clone(), {k: v.detach().cpu().tolist() for k, v in p.items()}

def nearest(p, c):
    # argmin returns the first sorted index on exact ties.
    return c[(p.unsqueeze(-1) - c).square().argmin(-1)]

def encode_pair(p, c, pairs, lambdas, initial):
    out = initial.clone()
    a, b = pairs['a'], pairs['b']
    require(len(a) > 0, 'No patch pair')
    ca = c.repeat_interleave(c.numel())
    cb = c.repeat(c.numel())
    for h in range(p.shape[1]):
        lam = lambdas[h].to(device=p.device, dtype=torch.float32)
        for start in range(0, p.shape[2], 16):
            stop = min(start + 16, p.shape[2])
            pa = p[0, h, start:stop, a]
            pb = p[0, h, start:stop, b]
            ea, eb = ca - pa.unsqueeze(-1), cb - pb.unsqueeze(-1)
            loss = (ea + eb).square() + lam * (ea - eb).square()
            index = loss.argmin(-1)
            out[0, h, start:stop, a] = ca[index]
            out[0, h, start:stop, b] = cb[index]
    require(bool(torch.isin(out, c).all()), 'Joint encoding left the frozen codebook')
    return out

def value_statistics(v, pairs):
    u = (v[0, :, pairs['a'], :] + v[0, :, pairs['b'], :]) / 2
    w = (v[0, :, pairs['a'], :] - v[0, :, pairs['b'], :]) / 2
    return u, w, u.square().sum((-1, -2)), w.square().sum((-1, -2))

def add_record(store, key, row):
    target = store.setdefault(key, {})
    for name, value in row.items():
        if torch.is_tensor(value):
            value = float(value.item())
        target[name] = target.get(name, 0.0) + value

def total(x):
    # Model, encoding, matmul and pair contributions are FP32.
    # Scalar-energy reductions use FP64 to reduce aggregation roundoff.
    return float(x.double().sum().item())

def sq(x):
    return total(x.square())

def control_diagnostics(p, n, one, c, pairs, site, sid, split):
    mask = n != one
    result = {'sample_id': sid, 'site': site, 'split': split,
              'different_scalars': int(mask.sum().item()),
              'max_abs_difference': float((n - one).abs().max().item()),
              'different_pairs': 0, 'examples': [], 'examples_limit': 20}
    if not bool(mask.any()):
        return result
    a, b = pairs['a'], pairs['b']
    different = (n[0, :, :, a] != one[0, :, :, a]) | (n[0, :, :, b] != one[0, :, :, b])
    result['different_pairs'] = int(different.sum().item())
    for h, query, pair in different.nonzero()[:20].tolist():
        ka, kb = a[pair], b[pair]
        pp = torch.stack((p[0, h, query, ka], p[0, h, query, kb]))
        nn = torch.stack((n[0, h, query, ka], n[0, h, query, kb]))
        oo = torch.stack((one[0, h, query, ka], one[0, h, query, kb]))
        def objective(z, dtype):
            e = z.to(dtype) - pp.to(dtype)
            return float(((e[0] + e[1]).square() + (e[0] - e[1]).square()).item())
        ds = (pp.double().unsqueeze(-1) - c.double()).square()
        ordered = ds.sort(-1).values
        result['examples'].append({'head': h, 'query': query, 'keys': [ka, kb],
            'P': pp.tolist(), 'nearest': nn.tolist(), 'lambda_one': oo.tolist(),
            'nearest_objective_fp32': objective(nn, torch.float32),
            'joint_objective_fp32': objective(oo, torch.float32),
            'nearest_objective_fp64': objective(nn, torch.float64),
            'joint_objective_fp64': objective(oo, torch.float64),
            'scalar_best_second_gaps_fp64': (ordered[:, 1] - ordered[:, 0]).tolist()})
    return result

def measure(capture, books, pairings, lambdas, split, sid, accumulated, controls):
    image = {}
    for site in SITES:
        p, v, native = capture[site]['P'], capture[site]['Vhat'], capture[site]['native']
        c = books[site]
        nn = nearest(p, c)
        arms = {'native': native, 'nearest': nn}
        arms['spatial_pair'] = encode_pair(p, c, pairings['spatial'], lambdas[site], nn)
        arms['shuffled_pair'] = encode_pair(p, c, pairings['shuffled'], lambdas[site], nn)
        arms['lambda_one'] = encode_pair(p, c, pairings['spatial'], torch.ones_like(lambdas[site]), nn)
        controls.append(control_diagnostics(p, nn, arms['lambda_one'], c, pairings['spatial'], site, sid, split))
        o0 = p @ v
        heads, queries = p.shape[1:3]
        image[site] = {a: {'pv_error_numerator': 0.0, 'o0_energy': 0.0} for a in ARMS}
        value_parts = {}
        for pairing, pairs in pairings.items():
            u, w, uu, dd = value_statistics(v, pairs)
            value_parts[pairing] = (u, w)
            for h in range(heads):
                add_record(accumulated, (split, site, 'value', str(h), pairing),
                           {'U': uu[h], 'D': dd[h], 'pair_count': len(pairs['a']), 'images': 1})
        for arm, phat in arms.items():
            require(bool(torch.isin(phat, c).all()), 'Arm contains a value outside C')
            e = phat - p
            delta = e @ v
            actual_delta = phat @ v - o0
            for h in range(heads):
                for token_type, sl in [('CLS', slice(0, 1)), ('patch', slice(1, queries))]:
                    eh = e[0, h, sl]
                    dh = delta[0, h, sl]
                    oh = o0[0, h, sl]
                    rowsum = eh.sum(-1)
                    row = {'images': 1, 'query_count': eh.shape[0], 'probability_elements': eh.numel(),
                           'p_squared_error': sq(eh), 'o0_energy': sq(oh),
                           'pv_error_numerator': sq(dh),
                           'direct_matmul_pv_error_numerator': sq(actual_delta[0, h, sl]),
                           'delta_vs_direct_roundoff_squared': sq(dh - actual_delta[0, h, sl]),
                           'row_sum_error_sum': total(rowsum), 'row_sum_error_squared': sq(rowsum)}
                    add_record(accumulated, (split, site, arm, str(h), token_type), row)
                    image[site][arm]['pv_error_numerator'] += row['pv_error_numerator']
                    image[site][arm]['o0_energy'] += row['o0_energy']
                    for pairing, pairs in pairings.items():
                        u, w = value_parts[pairing]
                        ea, eb = eh[:, pairs['a']], eh[:, pairs['b']]
                        contributions = ((ea + eb).unsqueeze(-1) * u[h].unsqueeze(0)
                                         + (ea - eb).unsqueeze(-1) * w[h].unsqueeze(0))
                        pair_sum = contributions.sum(1)
                        leftovers = [0] + pairs['unpaired_patch']
                        rest = eh[:, leftovers] @ v[0, h, leftovers, :]
                        reconstructed = pair_sum + rest
                        pair_energy = sq(contributions)
                        summed_energy = sq(pair_sum)
                        rest_energy = sq(rest)
                        full_energy = sq(dh)
                        decomp = {'images': 1, 'query_count': eh.shape[0],
                            'sum_pair_squared_norms': pair_energy,
                            'squared_norm_sum_pairs': summed_energy,
                            'cross_pair_difference': summed_energy - pair_energy,
                            'CLS_and_unpaired_squared_norm': rest_energy,
                            'pair_remainder_cross_difference': full_energy - summed_energy - rest_energy,
                            'total_pv_error_numerator': full_energy,
                            'decomposition_reconstruction_squared': sq(reconstructed - dh),
                            'o0_energy': sq(oh)}
                        add_record(accumulated, (split, site, arm, str(h), token_type, pairing), decomp)
        for arm in ARMS:
            row = image[site][arm]
            den = row['o0_energy']
            row['pooled_nmse'] = row['pv_error_numerator'] / den if den > 0 else None
    return image

def finalize_records(accumulated):
    expanded = dict(accumulated)
    for key, row in accumulated.items():
        aggregate_key = key[:3] + ('all_heads',) + key[4:]
        add_record(expanded, aggregate_key, row)
    rows = []
    for key, raw in expanded.items():
        row = dict(raw)
        row.update(split=key[0], site=key[1], arm_or_statistic=key[2], head=key[3])
        if key[2] == 'value':
            row['pairing'] = key[4]
        else:
            row['query_type'] = key[4]
            if len(key) == 6:
                row['pairing'] = key[5]
            den = row.get('o0_energy', 0.0)
            numerator = row.get('pv_error_numerator', row.get('total_pv_error_numerator'))
            row['pooled_nmse'] = numerator / den if den > 0 else None
            if 'row_sum_error_sum' in row:
                count = row['query_count']
                row['row_sum_error_bias'] = row['row_sum_error_sum'] / count
                row['row_sum_error_mse'] = row['row_sum_error_squared'] / count
        rows.append(row)
    return rows

def paired_differences(images):
    comparisons = [('spatial_pair', 'nearest'), ('shuffled_pair', 'nearest'),
                   ('spatial_pair', 'shuffled_pair'), ('native', 'nearest')]
    results = []
    for site in SITES:
        for left, right in comparisons:
            values = []
            for image in images:
                l = image['sites'][site][left]['pooled_nmse']
                r = image['sites'][site][right]['pooled_nmse']
                values.append({'sample_id': image['sample_id'], 'difference': l - r if l is not None and r is not None else None})
            finite = [v['difference'] for v in values if v['difference'] is not None]
            results.append({'site': site, 'comparison': left + '-' + right,
                'per_image': values, 'improved_images': sum(v < 0 for v in finite),
                'worsened_images': sum(v > 0 for v in finite), 'tied_images': sum(v == 0 for v in finite),
                'undefined_images': len(values) - len(finite),
                'mean_paired_difference_descriptive': sum(finite) / len(finite) if finite else None})
    return results

def execute(experiment_payload, context):
    backend, data = context['backend'], context['data']
    reference = context['reference_model']
    reference_before = signature(reference)
    model, receipt = context['load_frozen'](FROZEN_EVIDENCE)
    require(model is not reference and model is not backend.model, 'Frozen loader returned the protected FP model')
    require(not model.training and all(not p.requires_grad for p in model.parameters()), 'Loaded base is not frozen eval')
    require(all(not m.training for m in model.modules()), 'Loaded base has a training submodule')
    before = signature(model)
    grid, pairings = make_pairings(model)
    expected_tokens = 1 + grid[0] * grid[1]
    books, book_records, quantizers, operators = {}, {}, {}, {}
    for site in SITES:
        op = model.get_submodule(site)._ptq_probability
        require(hasattr(op, 'quant_a') and hasattr(op, 'quant_b'), 'Actual PV P/Value quantizers missing at ' + site)
        require(op.quant_b.enabled and op.quant_b.layer['activation']['bits'] == 4, 'Actual Value is not enabled A4')
        require(op.quant_b.execution['activation_calibration'] == 'static', 'Value is not static')
        c, params = actual_codebook(op.quant_a)
        books[site] = c
        book_records[site] = {'sorted_unique_decoded_C': c.cpu().tolist(), 'capacity': c.numel(),
            'frozen_parameters': params, 'P_layer': op.quant_a.layer, 'Value_layer': op.quant_b.layer,
            'group': 'actual scalar per-tensor P group shared by all heads',
            'P_signature': target_signature(op.quant_a, 'activation'),
            'Value_signature': target_signature(op.quant_b, 'activation')}
        quantizers[site] = (op.quant_a, op.quant_b)
        operators[site] = op
    calibration_ids = [sample_id(x) for x in list(data.ids('calibration'))[:2]]
    search_ids = [sample_id(x) for x in list(data.ids('search'))[:1]]
    require(len(calibration_ids) == 2 and len(set(calibration_ids)) == 2, 'Need exactly 2 unique calibration samples')
    require(len(search_ids) == 1 and len(set(search_ids)) == 1, 'Need exactly 1 unique search samples')
    handles, pending, capture = [], {}, {}
    shape_records = {}
    path_checks = {site: {'calls': 0, 'max_abs_PV_discrepancy': 0.0} for site in SITES}
    def save_input(site):
        def hook(m, args):
            require(site not in pending, 'PV called again before previous capture completed')
            require(len(args) == 2, 'Unexpected actual PV signature')
            pending[site] = {'P': args[0].detach().clone()}
        return hook
    def save_quantized(site, field):
        def hook(m, args, output):
            require(site in pending and field not in pending[site], 'Unexpected PV quantizer call order')
            pending[site][field] = output.detach().clone()
        return hook
    def finish(site):
        def hook(m, args, output):
            row = pending.pop(site)
            require(set(row) == {'P', 'native', 'Vhat'}, 'Incomplete actual PV capture')
            p, n, v = row['P'], row['native'], row['Vhat']
            require(p.ndim == 4 and v.ndim == 4 and p.shape[0] == 1, 'Expected actual P[B,H,Nq,Nk]/V[B,H,Nk,D]')
            require(p.shape == n.shape and p.shape[:2] == v.shape[:2] and p.shape[-1] == v.shape[-2], 'Actual PV shapes inconsistent')
            require(p.shape[-1] == expected_tokens and p.shape[-2] == expected_tokens, 'Actual token layout disagrees with adapter patch grid')
            require(all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in (p, n, v, output)), 'Actual capture not finite FP32')
            require(bool(torch.isin(n, books[site]).all()), 'Actual native P output disagrees with decoded frozen LUT')
            reconstructed = n @ v
            discrepancy = float((reconstructed - output).abs().max().item())
            require(torch.equal(reconstructed, output), 'Captured P/V do not reproduce the actual native PV path')
            path_checks[site]['calls'] += 1
            path_checks[site]['max_abs_PV_discrepancy'] = max(path_checks[site]['max_abs_PV_discrepancy'], discrepancy)
            shapes = {'P': list(p.shape), 'Vhat': list(v.shape), 'native_PV': list(output.shape)}
            if site in shape_records:
                require(shape_records[site] == shapes, 'Actual capture shape changed')
            shape_records[site] = shapes
            require(site not in capture, 'Multiple attention calls per image are unsupported')
            capture[site] = row
        return hook
    def run_sample(x):
        require(not pending and not capture, 'Stale capture')
        require(x.shape[0] == 1 and x.dtype == torch.float32, 'Expected FP32 batch_size=1 image')
        backend.forward_model(model, x)
        require(not pending and set(capture) == set(SITES), 'Model forward did not capture all requested sites')
        result = dict(capture)
        capture.clear()
        return result
    def selected(split, expected):
        count = 0
        for index, (x, sid) in enumerate(islice(data.batches(split), len(expected))):
            sid = sample_id(sid)
            require(sid == expected[index], 'Actual data order/identity mismatch for ' + split)
            count += 1
            yield x, sid
        require(count == len(expected), 'Insufficient actual batches for ' + split)
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for site in SITES:
            op = operators[site]
            handles.append(op.register_forward_pre_hook(save_input(site)))
            handles.append(op.quant_a.register_forward_hook(save_quantized(site, 'native')))
            handles.append(op.quant_b.register_forward_hook(save_quantized(site, 'Vhat')))
            handles.append(op.register_forward_hook(finish(site)))
        fit_stats = {}
        with torch.inference_mode():
            for x, sid in selected('calibration', calibration_ids):
                rows = run_sample(x)
                for site in SITES:
                    _, _, u, d = value_statistics(rows[site]['Vhat'], pairings['spatial'])
                    if site not in fit_stats:
                        fit_stats[site] = {'U': torch.zeros_like(u, dtype=torch.float64), 'D': torch.zeros_like(d, dtype=torch.float64)}
                    fit_stats[site]['U'] += u.double()
                    fit_stats[site]['D'] += d.double()
            lambdas, fit_records = {}, {}
            for site, stats in fit_stats.items():
                U, D = stats['U'], stats['D']
                ratio = torch.where(U > 0, D / torch.where(U > 0, U, torch.ones_like(U)), torch.ones_like(U))
                lam = ratio.clamp(1 / 64, 64).float()
                lambdas[site] = lam.detach().clone()
                fit_records[site] = [{'head': h, 'U': float(U[h].item()), 'D': float(D[h].item()),
                    'U_zero_fallback': bool(U[h] == 0), 'unclipped_D_over_U_or_fallback': float(ratio[h].item()),
                    'lambda': float(lam[h].item()), 'clipped_low': bool(ratio[h] < 1 / 64),
                    'clipped_high': bool(ratio[h] > 64), 'calibration_images': 2,
                    'patch_pairs_per_image': len(pairings['spatial']['a'])} for h in range(len(lam))]
            lambda_signature = fingerprint({s: tensor_digest(v) for s, v in lambdas.items()})
            accumulated, controls, calibration_images, search_images = {}, [], [], []
            for split, ids, images in [('calibration', calibration_ids, calibration_images), ('search', search_ids, search_images)]:
                for x, sid in selected(split, ids):
                    rows = run_sample(x)
                    result = measure(rows, books, pairings, lambdas, split, sid, accumulated, controls)
                    images.append({'sample_id': sid, 'sites': result})
            require(lambda_signature == fingerprint({s: tensor_digest(v) for s, v in lambdas.items()}), 'Frozen lambda changed during evaluation')
    finally:
        for handle in handles:
            handle.remove()
        pending.clear()
        capture.clear()
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
    after = signature(model)
    reference_after = signature(reference)
    require(before == after, 'Frozen base state, codebooks or quantization options changed')
    require(reference_before == reference_after, 'Protected FP model changed')
    backend.reset()
    return {'observations': {
        'status': 'completed', 'scope': 'offline local conditional PV comparison only; no task evaluation or whole-network intervention',
        'solution_id': SOLUTION, 'frozen_evidence_ref': FROZEN_EVIDENCE,
        'frozen_load_receipt': receipt, 'backend_identity': backend.identity(),
        'base_signature_before': before, 'base_signature_after': after,
        'reference_signature_before': reference_before, 'reference_signature_after': reference_after,
        'source_models_unchanged': True, 'actual_patch_grid': list(grid), 'actual_shapes': shape_records,
        'path_checks': path_checks, 'codebooks': book_records, 'pairings': pairings,
        'lambda_fit': fit_records, 'frozen_lambda_signature': lambda_signature,
        'sample_protocol': {'calibration_ids': calibration_ids, 'search_ids': search_ids,
            'calibration_search_intersection': [x for x in calibration_ids if x in set(search_ids)],
            'calibration_passes': 2, 'search_passes': 1,
            'fit_only': 'first calibration pass, spatial Value statistics; no search fitting',
            'search_not_claimed_previously_unused': True},
        'metrics': finalize_records(accumulated),
        'calibration_per_image': calibration_images, 'search_per_image': search_images,
        'search_paired_comparisons': paired_differences(search_images),
        'lambda_one_control': {'per_image_site': controls,
            'total_different_scalars': sum(r['different_scalars'] for r in controls),
            'total_different_pairs': sum(r['different_pairs'] for r in controls),
            'interpretation': 'Exact FP32 enumeration; discrepancies are retained, with bounded exact locations and FP64 objective/gap diagnostics.'},
        'metric_definitions': {
            'pv_error_numerator': 'sum ||(Phat-P)@Vhat||^2, conditional on actual frozen Vhat',
            'direct_matmul_pv_error_numerator': 'sum ||Phat@Vhat-P@Vhat||^2; finite precision comparison retained',
            'pooled_nmse': 'summed PV error numerator / summed conditional O0 energy, never mean head NMSE',
            'all_heads': 'sums of raw head sufficient statistics; images field counts head-image contributions',
            'p_squared_error': 'sum scalar probability reconstruction squared error',
            'row_sum_error_bias': 'sum of signed probability row-sum errors / query count',
            'row_sum_error_mse': 'sum squared probability row-sum errors / query count',
            'cross_pair_difference': '||sum_pair d_pair||^2 - sum_pair ||d_pair||^2',
            'pair_remainder_cross_difference': 'total PV error energy - summed-pair energy - CLS/unpaired energy'},
        'limitations': ['One frozen model and one fixed sample selection, not independent repeats.',
            'The local reference bypasses only the current P quantizer; it is not full FP attention.',
            'No probability normalization, residual correction, extra model QDQ, or chained local replacements.',
            'No whole-network KL, classification accuracy, ADE performance, or speedup is measured.',
            'Encoding and matmul are FP32 with TF32 disabled; scalar energy aggregation uses FP64.',
            'Pairwise objective is not the exact total PV objective; cross terms are explicitly reported.']
        }}
