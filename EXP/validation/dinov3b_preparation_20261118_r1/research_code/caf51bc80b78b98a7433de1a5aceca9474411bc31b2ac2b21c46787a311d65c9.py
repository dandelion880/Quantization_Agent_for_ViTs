import copy
import hashlib
import json
import math
import torch
from ptq_agent.execution.sites import install_attention

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def model_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(clone_tree(x) for x in value)
    if isinstance(value, list):
        return [clone_tree(x) for x in value]
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    return value

def boundary(value):
    if torch.is_tensor(value):
        return value
    require(isinstance(value, (tuple, list)) and len(value) == 1, 'Expected single-image block list boundary')
    return boundary(value[0])

def error(reference, candidate):
    require(reference.shape == candidate.shape, 'Metric shape mismatch')
    a, b = reference.double(), candidate.double()
    require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), 'Nonfinite measurement')
    sse = float((b-a).square().sum().item())
    energy = float(a.square().sum().item())
    count = a.numel()
    return {'sse': sse, 'reference_energy': energy, 'elements': count, 'mse': sse/count, 'nmse': sse/max(energy, 1e-12)}

def aggregate(rows):
    if not rows:
        return None
    if 'sse' in rows[0] and 'reference_energy' in rows[0]:
        s = sum(r['sse'] for r in rows)
        d = sum(r['reference_energy'] for r in rows)
        n = sum(r['elements'] for r in rows)
        return {'sse': s, 'reference_energy': d, 'elements': n, 'mse': s/n, 'nmse': s/max(d, 1e-12)}
    if isinstance(rows[0], dict):
        return {k: aggregate([r[k] for r in rows]) for k in rows[0]}
    return sum(rows)/len(rows)

def flatten_metrics(value, prefix=''):
    result = {}
    for k, v in value.items():
        path = prefix + '/' + k
        if isinstance(v, dict):
            result.update(flatten_metrics(v, path))
        elif isinstance(v, (int, float)) and (k in ('mse', 'nmse', 'teacher_kl') or 'row_sum' in path):
            result[path] = float(v)
    return result

def groups(n):
    return {'all': (0,n), 'cls': (0,1), 'storage': (1,5), 'patch': (5,n)}

def local_metrics(p, v, qp):
    n = p.shape[-1]
    e = qp-p
    pv = p@v
    ev = e@v
    result = {'probability': {}, 'pv': {}, 'row_sum': {}}
    for name, (a,b) in groups(n).items():
        pp, qq = p[:,:,a:b,:], qp[:,:,a:b,:]
        row = {'all_keys': error(pp, qq)}
        for key, (c,d) in groups(n).items():
            if key != 'all':
                row[key+'_keys'] = error(pp[:,:,:,c:d], qq[:,:,:,c:d])
        result['probability'][name] = row
        result['pv'][name] = error(pv[:,:,a:b,:], pv[:,:,a:b,:]+ev[:,:,a:b,:])
        delta = qq.double().sum(-1)-pp.double().sum(-1)
        result['row_sum'][name] = {'signed_bias': float(delta.mean()), 'mae': float(delta.abs().mean()), 'mse': float(delta.square().mean()), 'fp_mean': float(pp.double().sum(-1).mean()), 'quantized_mean': float(qq.double().sum(-1).mean())}
    return result

class FrozenProbability(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.state_text = canonical(state)
        self.state_hash = digest(state)
        self.members = [tuple(g) for g in state['members']]
        self.books = copy.deepcopy(state['books'])
        self.n = state['n']

    def forward(self, p):
        require(p.ndim == 4 and p.shape[-1] == self.n, 'Frozen probability shape mismatch')
        out = torch.empty_like(p)
        for members, book in zip(self.members, self.books):
            index = torch.tensor(members, device=p.device, dtype=torch.long)
            x = p.index_select(-1, index)
            y = encode(x, book)
            out.index_copy_(-1, index, y)
        return out

def encode(p, book):
    require(p.dtype == torch.float32, 'Probability must be FP32')
    ratio = (p/p.new_tensor(book['s'])).clamp(1e-15, 1.0)
    codes = torch.round(-torch.log2(ratio)*37.0/book['q']).long()
    table = p.new_tensor(book['lut'])
    y = table[codes.clamp(0,15)]
    return torch.where((p > 0) & (codes < 15), y, torch.zeros_like(y))

def make_books(maximum, qs, factors):
    books = []
    for q in qs:
        for factor in factors:
            scale = maximum*factor
            lut = []
            for c in range(15):
                integer, remainder = divmod(c*q,37)
                lut.append(scale*(2.0**(-integer))*round(30.0*2.0**(-remainder/37.0))/30.0)
            lut.append(0.0)
            lut = torch.tensor(lut, dtype=torch.float32).tolist()
            require(scale > 0 and len(set(lut)) <= 16, 'Invalid four-bit codebook')
            books.append({'id': len(books), 'q': q, 's': scale, 'scale_factor': factor, 'lut': lut, 'unique_reconstruction_values': len(set(lut))})
    return books

def capture(backend, model, image, adapted=False, calls=False):
    saved, handles = {}, []
    def block_hook(name):
        def hook(module, args, kwargs, output):
            require(name not in saved, 'Repeated block call')
            saved[name] = boundary(output).detach().clone()
            if calls and name == 'blocks.5':
                saved['block_call'] = clone_tree((args, kwargs))
                saved['block_input'] = boundary(args[0]).detach().clone()
        return hook
    def norm_hook(module, args, output):
        require('normalized' not in saved, 'Repeated final norm call')
        t = boundary(output)
        require(t.ndim == 3 and t.shape[1] == 201, 'Unexpected normalized-token shape')
        saved['normalized'] = t.detach().clone()
    def head_hook(module, args):
        require('head_input' not in saved, 'Repeated head call')
        saved['head_input'] = args[0].detach().clone()
    def pv_hook(module, args):
        require('p' not in saved, 'Repeated probability edge call')
        saved['p'], saved['v'] = args[0].detach().clone(), args[1].detach().clone()
    try:
        for name in ('blocks.5','blocks.8','blocks.11'):
            handles.append(model.get_submodule(name).register_forward_hook(block_hook(name), with_kwargs=True))
        handles.append(model.get_submodule('norm').register_forward_hook(norm_hook))
        handles.append(model.get_submodule('head').register_forward_pre_hook(head_hook))
        if adapted:
            handles.append(model.get_submodule('blocks.5.attn')._ptq_probability.register_forward_pre_hook(pv_hook))
        logits = backend.forward_model(model, image)
        require(torch.is_tensor(logits) and tuple(logits.shape) == (1,1000), 'Expected bound 1000-class logits')
        require('head_input' in saved and 'normalized' in saved, 'Missing bound normalized-CLS head edge')
        require(torch.equal(saved['head_input'], saved['normalized'][:,0]), 'Classifier does not consume final normalized CLS')
        saved['cls'] = saved['normalized'][:,0].clone()
        saved['logits'] = logits.detach().clone()
        if adapted:
            require(tuple(saved['p'].shape) == (1,12,201,201), 'Unexpected P shape')
            require(tuple(saved['v'].shape) == (1,12,201,64), 'Unexpected V shape')
            require(bool((saved['p'] >= 0).all()), 'Negative probability')
        return saved
    finally:
        for h in handles:
            h.remove()

def check_replay(reference, adapted):
    record = {}
    for key in ('blocks.5','blocks.8','blocks.11','cls','logits'):
        a, b = reference[key], adapted[key]
        record[key] = {'max_abs': float((a-b).abs().max()), 'error': error(a,b)}
        require(torch.allclose(a,b,atol=1e-4,rtol=1e-4), 'Unintervened attention adaptation mismatch at '+key)
    return record

def candidate_statistics(records, members, books, device):
    g, k = len(members), len(books)
    gram = torch.zeros((g*k,g*k), dtype=torch.float64, device=device)
    probability = torch.zeros((g,k), dtype=torch.float64, device=device)
    p_count, pv_count = 0, 0
    for row in records:
        p, v = row['p'].to(device), row['v'].to(device)
        residuals = []
        p_count += p.numel()
        pv_count += p.shape[0]*p.shape[1]*p.shape[2]*v.shape[-1]
        for gi, member in enumerate(members):
            idx = torch.tensor(member, device=device, dtype=torch.long)
            pg, vg = p.index_select(-1,idx), v.index_select(-2,idx)
            for ci, book in enumerate(books):
                e = encode(pg,book)-pg
                probability[gi,ci] += e.double().square().sum()
                residuals.append((e@vg).flatten())
        z = torch.stack(residuals).double()
        gram += z@z.T
    return {'gram': gram.cpu(), 'probability': probability.cpu(), 'p_count': p_count, 'pv_count': pv_count}

def objective_score(stats, choices, objective):
    if objective == 'probability_mse':
        return sum(float(stats['probability'][g,c]) for g,c in enumerate(choices))/stats['p_count']
    k = stats['probability'].shape[1]
    idx = torch.tensor([g*k+c for g,c in enumerate(choices)],dtype=torch.long)
    return float(stats['gram'].index_select(0,idx).index_select(1,idx).sum())/stats['pv_count']

def optimize(stats, initial, objective, sweeps):
    current = list(initial)
    start = objective_score(stats,current,objective)
    trajectory = []
    for sweep in range(sweeps):
        for group in range(len(current)):
            scores = []
            for ci in range(stats['probability'].shape[1]):
                trial = list(current)
                trial[group] = ci
                scores.append(objective_score(stats,trial,objective))
            best = min(scores)
            old = current[group]
            selected = old if scores[old] == best else scores.index(best)
            current[group] = selected
            trajectory.append({'sweep': sweep, 'group': group, 'before_candidate': old, 'candidate_scores': scores, 'selected_candidate': selected, 'choices_after': list(current), 'selected_score': scores[selected]})
    return current, {'initial_choices': list(initial), 'initial_objective': start, 'trajectory': trajectory, 'final_choices': current, 'final_objective': objective_score(stats,current,objective), 'coordinate_candidate_scores': len(trajectory)*stats['probability'].shape[1]}

def downstream_metrics(fp, q):
    result = {'block5_output': error(fp['blocks.5'],q['blocks.5']), 'block8_patch': error(fp['blocks.8'][:,5:],q['blocks.8'][:,5:]), 'block11_patch': error(fp['blocks.11'][:,5:],q['blocks.11'][:,5:]), 'x_norm_clstoken': error(fp['cls'],q['cls'])}
    a, b = fp['logits'].double(), q['logits'].double()
    result['teacher_kl'] = float((a.softmax(-1)*(a.log_softmax(-1)-b.log_softmax(-1))).sum(-1).mean())
    return result

def selected_batches(data, split, count):
    iterator = iter(data.batches(split))
    for _ in range(count):
        try:
            image, sid = next(iterator)
        except StopIteration:
            raise RuntimeError('Insufficient '+split+' samples')
        require(torch.is_tensor(image) and image.shape[0] == 1 and image.dtype == torch.float32, 'Expected FP32 batch size one')
        yield image, sid

def execute(experiment_payload, context):
    backend, data = context['backend'], context['data']
    reference = context['reference_model']
    execution = experiment_payload['execution']
    params = execution['parameters']
    require(execution.get('training') is False, 'Forward-only experiment required')
    require(execution['calibration'] == {'count':32,'offset':0,'split':'calibration'}, 'Calibration protocol mismatch')
    require(params['block'] == 'blocks.5' and params['site'] == 'blocks.5.attn::post_softmax', 'Site mismatch')
    require(params['coordinate_sweeps'] == 2 and params['search_count'] == 64 and params['search_offset'] == 0, 'Search protocol mismatch')
    require(params['log_base_q_candidates'] == [19,37,55,74] and params['scale_factors'] == [0.0625,0.125,0.25,0.5,1.0] and params['nonsemantic_group_seed'] == 42, 'Candidate protocol mismatch')
    require(not reference.training and all(not m.training for m in reference.modules()), 'Reference must be eval')
    require(all(not p.requires_grad for p in reference.parameters()), 'Reference must be frozen')
    require(all(t.dtype == torch.float32 for t in list(reference.parameters())+list(reference.buffers()) if t.is_floating_point()), 'Expected FP32 reference')
    source_hash = model_hash(reference)
    source_flags = [(n,m.training) for n,m in reference.named_modules()]
    reference_grads = [p.requires_grad for p in reference.parameters()]
    model = copy.deepcopy(reference).eval().requires_grad_(False)
    device = next(model.parameters()).device
    candidate_weight_hash = model_hash(model)
    install_attention(model,['blocks.5.attn'])
    edge = model.get_submodule('blocks.5.attn')._ptq_probability
    require(not hasattr(edge,'quant_a') and not hasattr(edge,'quant_b'), 'Unexpected existing quantization')
    calibration_ids = list(data.ids('calibration'))[:32]
    search_ids = list(data.ids('search'))[:64]
    require(len(calibration_ids) == 32 and len(search_ids) == 64, 'Insufficient bound IDs')
    require(len(set(calibration_ids)) == 32 and len(set(search_ids)) == 64, 'Duplicate IDs')
    records, replay, shapes = [], {'calibration': [], 'search': []}, {}
    states, quantizers, arms = {}, {}, {}
    reconstruction_checks = {}
    maximum = 0.0
    try:
        with torch.inference_mode():
            for index,(image,sid) in enumerate(selected_batches(data,'calibration',32)):
                require(sid == calibration_ids[index], 'Calibration order mismatch')
                fp = capture(backend,reference,image)
                adapted = capture(backend,model,image,adapted=True,calls=True)
                checks = check_replay(fp,adapted)
                args, kwargs = adapted['block_call']
                block_output = boundary(model.get_submodule('blocks.5')(*args,**kwargs))
                checks['block_call_replay'] = {'max_abs': float((block_output-adapted['blocks.5']).abs().max()), 'error': error(adapted['blocks.5'],block_output)}
                require(torch.equal(block_output,adapted['blocks.5']), 'Exact block list/RoPE replay failed')
                replay['calibration'].append({'sample_id':sid,'checks':checks})
                maximum = max(maximum,float(adapted['p'].max()))
                records.append({'sample_id':sid,'p':adapted['p'].cpu(),'v':adapted['v'].cpu()})
                shapes = {'p':list(adapted['p'].shape),'v':list(adapted['v'].shape),'block_input':list(adapted['block_input'].shape),'block_output':list(adapted['blocks.5'].shape)}
            require(maximum > 0 and math.isfinite(maximum), 'Invalid calibration maximum')
            books = make_books(maximum,params['log_base_q_candidates'],params['scale_factors'])
            semantic = [list(range(0,1)),list(range(1,5)),list(range(5,201))]
            generator = torch.Generator(device='cpu').manual_seed(42)
            permutation = torch.randperm(201,generator=generator).tolist()
            nonsemantic = [permutation[:1],permutation[1:5],permutation[5:]]
            members_by_name = {'shared':[list(range(201))], 'semantic':semantic, 'nonsemantic':nonsemantic}
            sem_stats = candidate_statistics(records,semantic,books,device)
            non_stats = candidate_statistics(records,nonsemantic,books,device)
            shared_traces = {}
            for objective in ('probability_mse','pv_mse'):
                shared_scores = [objective_score(sem_stats,[ci]*3,objective) for ci in range(20)]
                initial = shared_scores.index(min(shared_scores))
                shared_traces[objective] = {'candidate_scores': shared_scores, 'selected_candidate':initial, 'candidate_score_count':20}
                for grouping in ('shared','semantic','nonsemantic'):
                    name = grouping+'_'+objective
                    if grouping == 'shared':
                        choices = [initial]
                        trace = {'initial_choices':choices,'initial_objective':shared_scores[initial],'final_choices':choices,'final_objective':shared_scores[initial],'trajectory':[],'coordinate_candidate_scores':0}
                    else:
                        stats = sem_stats if grouping == 'semantic' else non_stats
                        choices,trace = optimize(stats,[initial]*3,objective,2)
                    state = {'n':201,'grouping':grouping,'objective':objective,'members':members_by_name[grouping],'candidate_ids':choices,'books':[copy.deepcopy(books[c]) for c in choices],'rounding':'ties_to_even','row_renormalization':False}
                    restored = json.loads(canonical(state))
                    require(restored == state, 'State serialization changed values')
                    original_quantizer = FrozenProbability(state).to(device)
                    restored_quantizer = FrozenProbability(restored).to(device)
                    for row in records:
                        p = row['p'].to(device)
                        require(torch.equal(original_quantizer(p),restored_quantizer(p)), 'Reconstructed quantizer output mismatch')
                    states[name] = restored
                    quantizers[name] = restored_quantizer
                    reconstruction_checks[name] = {'json_round_trip_equal':True,'members_lut_equal':True,'exact_output_equal_on_calibration':True,'samples':32,'frozen_hash_before':digest(restored)}
                    selected_edges = [{'q_at_grid_boundary':b['q'] in (19,74),'scale_at_grid_boundary':b['scale_factor'] in (0.0625,1.0)} for b in state['books']]
                    arms[name] = {'grouping':grouping,'objective':objective,'fit':trace,'selected_boundary_flags':selected_edges,'calibration_per_image':[],'search_per_image':[]}
            del sem_stats, non_stats
            for row in records:
                p, v = row['p'].to(device),row['v'].to(device)
                for name,quantizer in quantizers.items():
                    arms[name]['calibration_per_image'].append({'sample_id':row['sample_id'],'metrics':local_metrics(p,v,quantizer(p))})
            overlap = set(calibration_ids).intersection(search_ids)
            for index,(image,sid) in enumerate(selected_batches(data,'search',64)):
                require(sid == search_ids[index], 'Search order mismatch')
                fp = capture(backend,reference,image)
                adapted = capture(backend,model,image,adapted=True)
                replay['search'].append({'sample_id':sid,'checks':check_replay(fp,adapted)})
                for name,quantizer in quantizers.items():
                    local = local_metrics(adapted['p'],adapted['v'],quantizer(adapted['p']))
                    edge.add_module('quant_a',quantizer)
                    try:
                        actual = capture(backend,model,image,adapted=True)
                    finally:
                        delattr(edge,'quant_a')
                    require(torch.equal(actual['p'],adapted['p']) and torch.equal(actual['v'],adapted['v']), 'Paired P/V inputs changed')
                    metrics = {'local':local,'downstream':downstream_metrics(fp,actual)}
                    arms[name]['search_per_image'].append({'sample_id':sid,'overlaps_calibration':sid in overlap,'metrics':metrics})
            for name,arm in arms.items():
                arm['calibration_aggregate'] = aggregate([r['metrics'] for r in arm['calibration_per_image']])
                rows = arm['search_per_image']
                arm['search_aggregate'] = aggregate([r['metrics'] for r in rows])
                arm['search_overlap_aggregate'] = aggregate([r['metrics'] for r in rows if r['overlaps_calibration']])
                arm['search_nonoverlap_aggregate'] = aggregate([r['metrics'] for r in rows if not r['overlaps_calibration']])
                reconstruction_checks[name]['frozen_hash_after'] = quantizers[name].state_hash
                require(quantizers[name].state_hash == digest(states[name]) and quantizers[name].state_text == canonical(states[name]), 'Frozen quantizer state changed')
            comparisons = {}
            left = 'semantic_pv_mse'
            for right in ('nonsemantic_pv_mse','semantic_probability_mse','shared_pv_mse'):
                differences = []
                for a,b in zip(arms[left]['search_per_image'],arms[right]['search_per_image']):
                    require(a['sample_id'] == b['sample_id'], 'Unpaired comparison')
                    fa,fb = flatten_metrics(a['metrics']),flatten_metrics(b['metrics'])
                    differences.append({'sample_id':a['sample_id'],'overlaps_calibration':a['overlaps_calibration'],'left_minus_right':{k:fa[k]-fb[k] for k in fa}})
                def summarize(subset):
                    if not subset:
                        return None
                    return {k:{'mean_left_minus_right':sum(r['left_minus_right'][k] for r in subset)/len(subset),'left_lower':sum(r['left_minus_right'][k] < 0 for r in subset),'equal':sum(r['left_minus_right'][k] == 0 for r in subset),'left_higher':sum(r['left_minus_right'][k] > 0 for r in subset)} for k in subset[0]['left_minus_right']}
                comparisons[left+'_vs_'+right] = {'left':left,'right':right,'per_image':differences,'all':summarize(differences),'overlap':summarize([r for r in differences if r['overlaps_calibration']]),'nonoverlap':summarize([r for r in differences if not r['overlaps_calibration']]),'interpretation':'Lower is favorable only for error/KL measures; signed row-sum bias is descriptive.'}
            require(model_hash(model) == candidate_weight_hash, 'Non-target model parameters/buffers changed')
            identity = backend.identity()
            result = {'status':'completed','scope':'FP32 model with only blocks.5 post-softmax P locally quantized; not full W4A4','identity':identity,'source_state_sha256':source_hash,'head_state_sha256':model_hash(reference.get_submodule('head')),'shapes':shapes,'calibration_maximum_M':maximum,'candidate_order':'q outer, scale_factor inner; list order resolves non-current ties','candidate_codebooks':books,'group_members':members_by_name,'nonsemantic_permutation':permutation,'shared_search':shared_traces,'states':states,'state_hashes':{k:digest(v) for k,v in states.items()},'reconstruction_checks':reconstruction_checks,'replay_checks':replay,'replay_tolerance':{'atol':1e-4,'rtol':1e-4,'block_call_replay':'bitwise equality'},'sample_ids':{'calibration':calibration_ids,'search':search_ids,'intersection':[s for s in search_ids if s in overlap]},'arms':arms,'paired_comparisons':comparisons,'numerics':{'codec':'reserve_zero 15 positive entries plus zero; no row normalization','round':'ties_to_even','encoding_dtype':'FP32','gram_accumulation_dtype':'FP64','fit_pv':'(Q(P)-P) @ V; global Gram sum includes cross-group terms','local_pv':'EV relative to FP PV energy','nmse_denominator_floor':1e-12,'aggregation':'MSE/NMSE use pooled SSE, reference energy and element counts; other scalars are image means','search_budget':'20 shared scores per objective; each three-group arm 2*3*20 coordinate scores; shared initialization reused','state_reload':'local JSON codebook/member reconstruction only, not complete model export'},'limitations':['Single predetermined nonsemantic partition; no general semantic superiority established by this experiment alone.','Search observations are descriptive and not independent experimental repetitions.','Patch-feature errors are not ADE mIoU; classification KL is not accuracy.','Only calibration determines parameters; no ADE/final data consumed.']}
        return {'observations':result}
    finally:
        if hasattr(edge,'quant_a'):
            delattr(edge,'quant_a')
        require(model_hash(reference) == source_hash, 'Protected FP source state changed')
        require(source_flags == [(n,m.training) for n,m in reference.named_modules()], 'Protected FP mode changed')
        require(reference_grads == [p.requires_grad for p in reference.parameters()], 'Protected FP gradient flags changed')
        backend.reset()
