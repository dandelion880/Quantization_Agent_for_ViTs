import copy
import hashlib
import itertools
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
        return {k: clone_tree(v) for k,v in value.items()}
    return value

def boundary(value):
    if torch.is_tensor(value):
        return value
    require(isinstance(value,(tuple,list)) and len(value)==1, 'Expected single-image block list boundary')
    return boundary(value[0])

def error_record(sse, energy, count):
    return {'sse':sse,'reference_energy':energy,'elements':count,'mse':sse/count,'nmse':sse/max(energy,1e-12)}

def error(reference, candidate):
    require(reference.shape==candidate.shape,'Metric shape mismatch')
    a,b = reference.double(),candidate.double()
    require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'Nonfinite measurement')
    return error_record(float((b-a).square().sum()),float(a.square().sum()),a.numel())

def aggregate(rows):
    if not rows:
        return None
    if isinstance(rows[0],dict):
        if 'sse' in rows[0] and 'reference_energy' in rows[0]:
            return error_record(sum(r['sse'] for r in rows),sum(r['reference_energy'] for r in rows),sum(r['elements'] for r in rows))
        return {k:aggregate([r[k] for r in rows]) for k in rows[0]}
    return sum(rows)/len(rows)

def flatten_metrics(value,prefix=''):
    result={}
    for k,v in value.items():
        path=prefix+'/'+k
        if isinstance(v,dict):
            result.update(flatten_metrics(v,path))
        elif isinstance(v,(int,float)) and (k in ('mse','nmse','teacher_kl') or 'row_sum' in path):
            result[path]=float(v)
    return result

def groups(n):
    return {'all':(0,n),'cls':(0,1),'storage':(1,5),'patch':(5,n)}

def local_metrics(p,v,qp):
    pv=p@v
    ev=(qp-p)@v
    result={'probability':{},'pv':{},'row_sum':{}}
    for name,(a,b) in groups(p.shape[-1]).items():
        pp,qq=p[:,:,a:b,:],qp[:,:,a:b,:]
        row={'all_keys':error(pp,qq)}
        for key,(c,d) in groups(p.shape[-1]).items():
            if key!='all':
                row[key+'_keys']=error(pp[:,:,:,c:d],qq[:,:,:,c:d])
        result['probability'][name]=row
        e=ev[:,:,a:b,:].double()
        ref=pv[:,:,a:b,:].double()
        result['pv'][name]=error_record(float(e.square().sum()),float(ref.square().sum()),ref.numel())
        delta=qq.double().sum(-1)-pp.double().sum(-1)
        result['row_sum'][name]={'signed_bias':float(delta.mean()),'mae':float(delta.abs().mean()),'mse':float(delta.square().mean()),'fp_mean':float(pp.double().sum(-1).mean()),'quantized_mean':float(qq.double().sum(-1).mean())}
    return result

class FrozenProbability(torch.nn.Module):
    def __init__(self,state):
        super().__init__()
        self.state_text=canonical(state)
        self.state_hash=digest(state)
        self.members=[tuple(g) for g in state['members']]
        self.books=copy.deepcopy(state['books'])
        self.n=state['n']
        require(sorted(x for g in self.members for x in g)==list(range(self.n)),'Invalid partition')
    def forward(self,p):
        require(p.ndim==4 and p.shape[-1]==self.n,'Frozen probability shape mismatch')
        out=torch.empty_like(p)
        for members,book in zip(self.members,self.books):
            index=torch.tensor(members,device=p.device,dtype=torch.long)
            out.index_copy_(-1,index,encode(p.index_select(-1,index),book))
        return out
    def verify(self,state):
        actual={'members':[list(g) for g in self.members],'books':self.books,'n':self.n}
        require(actual=={k:state[k] for k in actual},'Frozen numerical state changed')
        require(self.state_text==canonical(state) and self.state_hash==digest(state),'Frozen state metadata changed')

def encode(p,book):
    require(p.dtype==torch.float32,'Probability must be FP32')
    ratio=(p/p.new_tensor(book['s'])).clamp(1e-15,1.0)
    codes=torch.round(-torch.log2(ratio)*37.0/book['q']).long()
    table=p.new_tensor(book['lut'])
    y=table[codes.clamp(0,15)]
    return torch.where((p>0)&(codes<15),y,torch.zeros_like(y))

def make_books(maximum,qs,factors):
    books=[]
    for q in qs:
        for factor in factors:
            scale=maximum*factor
            lut=[]
            for c in range(15):
                integer,remainder=divmod(c*q,37)
                lut.append(scale*(2.0**(-integer))*round(30.0*2.0**(-remainder/37.0))/30.0)
            lut.append(0.0)
            lut=torch.tensor(lut,dtype=torch.float32).tolist()
            require(scale>0 and len(set(lut))<=16,'Invalid four-bit codebook')
            books.append({'id':len(books),'q':q,'s':scale,'scale_factor':factor,'lut':lut,'unique_reconstruction_values':len(set(lut))})
    return books

def capture(backend,model,image,adapted=False,calls=False):
    saved,handles={},[]
    def block_hook(name):
        def hook(module,args,kwargs,output):
            require(name not in saved,'Repeated block call')
            saved[name]=boundary(output).detach().clone()
            if calls and name=='blocks.5':
                saved['block_call']=clone_tree((args,kwargs))
                saved['block_input']=boundary(args[0]).detach().clone()
        return hook
    def norm_hook(module,args,output):
        require('normalized' not in saved,'Repeated final norm call')
        t=boundary(output)
        require(t.ndim==3 and t.shape[1]==201,'Unexpected normalized-token shape')
        saved['normalized']=t.detach().clone()
    def head_hook(module,args):
        require('head_input' not in saved,'Repeated head call')
        saved['head_input']=args[0].detach().clone()
    def pv_hook(module,args):
        require('p' not in saved,'Repeated probability edge call')
        saved['p'],saved['v']=args[0].detach().clone(),args[1].detach().clone()
    try:
        for name in ('blocks.5','blocks.8','blocks.11'):
            handles.append(model.get_submodule(name).register_forward_hook(block_hook(name),with_kwargs=True))
        handles.append(model.get_submodule('norm').register_forward_hook(norm_hook))
        handles.append(model.get_submodule('head').register_forward_pre_hook(head_hook))
        if adapted:
            handles.append(model.get_submodule('blocks.5.attn')._ptq_probability.register_forward_pre_hook(pv_hook))
        logits=backend.forward_model(model,image)
        require(torch.is_tensor(logits) and tuple(logits.shape)==(1,1000),'Expected bound 1000-class logits')
        require(torch.equal(saved['head_input'],saved['normalized'][:,0]),'Classifier must consume normalized CLS')
        saved['cls']=saved['normalized'][:,0].clone()
        saved['logits']=logits.detach().clone()
        if adapted:
            require(tuple(saved['p'].shape)==(1,12,201,201),'Unexpected P shape')
            require(tuple(saved['v'].shape)==(1,12,201,64),'Unexpected V shape')
            require(bool(torch.isfinite(saved['p']).all() and (saved['p']>=0).all()),'Invalid probabilities')
        return saved
    finally:
        for h in handles:
            h.remove()

def check_replay(reference,adapted):
    record={}
    for key in ('blocks.5','blocks.8','blocks.11','cls','logits'):
        a,b=reference[key],adapted[key]
        record[key]={'max_abs':float((a-b).abs().max()),'error':error(a,b)}
        require(torch.allclose(a,b,atol=1e-4,rtol=1e-4),'Unintervened adaptation mismatch at '+key)
    return record

def candidate_statistics(records,members,books,device):
    g,k=len(members),len(books)
    gram=torch.zeros((g*k,g*k),dtype=torch.float64,device=device)
    probability=torch.zeros((g,k),dtype=torch.float64,device=device)
    p_count,pv_count=0,0
    for row in records:
        p,v=row['p'].to(device),row['v'].to(device)
        residuals=[]
        p_count+=p.numel()
        pv_count+=p.shape[0]*p.shape[1]*p.shape[2]*v.shape[-1]
        for gi,member in enumerate(members):
            idx=torch.tensor(member,device=device,dtype=torch.long)
            pg,vg=p.index_select(-1,idx),v.index_select(-2,idx)
            for ci,book in enumerate(books):
                e=encode(pg,book)-pg
                probability[gi,ci]+=e.double().square().sum()
                residuals.append((e@vg).flatten())
        z=torch.stack(residuals).double()
        gram+=z@z.T
    return {'gram':gram.cpu(),'probability':probability.cpu(),'p_count':p_count,'pv_count':pv_count}

def objective_score(stats,choices,objective):
    if objective=='probability_mse':
        return sum(float(stats['probability'][g,c]) for g,c in enumerate(choices))/stats['p_count']
    k=stats['probability'].shape[1]
    idx=torch.tensor([g*k+c for g,c in enumerate(choices)],dtype=torch.long)
    return float(stats['gram'].index_select(0,idx).index_select(1,idx).sum())/stats['pv_count']

def optimize(stats,initial,objective,sweeps):
    current=list(initial)
    start=objective_score(stats,current,objective)
    trajectory=[]
    for sweep in range(sweeps):
        for group in range(len(current)):
            scores=[]
            for ci in range(stats['probability'].shape[1]):
                trial=list(current)
                trial[group]=ci
                scores.append(objective_score(stats,trial,objective))
            best=min(scores)
            old=current[group]
            selected=old if scores[old]==best else scores.index(best)
            current[group]=selected
            trajectory.append({'sweep':sweep,'group':group,'before_candidate':old,'candidate_scores':scores,'selected_candidate':selected,'choices_after':list(current),'selected_score':scores[selected]})
    return current,{'initial_choices':list(initial),'initial_objective':start,'trajectory':trajectory,'final_choices':list(current),'final_objective':objective_score(stats,current,objective),'coordinate_candidate_scores':len(trajectory)*stats['probability'].shape[1]}

def exhaustive(stats,objective):
    g,k=stats['probability'].shape
    scores=[]
    best=None
    selected=None
    for choices in itertools.product(range(k),repeat=g):
        score=objective_score(stats,choices,objective)
        require(math.isfinite(score),'Nonfinite candidate score')
        scores.append(score)
        if best is None or score<best:
            best,selected=score,list(choices)
    return selected,{'method':'finite_grid_exhaustive','ordering':'lexicographic candidate tuples; ties first','tuple_shape':[k]*g,'all_scores':scores,'candidate_score_count':len(scores),'final_choices':selected,'final_objective':best}

def downstream_metrics(fp,q):
    result={'block5_output':error(fp['blocks.5'],q['blocks.5']),'block8_patch':error(fp['blocks.8'][:,5:],q['blocks.8'][:,5:]),'block11_patch':error(fp['blocks.11'][:,5:],q['blocks.11'][:,5:]),'x_norm_clstoken':error(fp['cls'],q['cls'])}
    a,b=fp['logits'].double(),q['logits'].double()
    result['teacher_kl']=float((a.softmax(-1)*(a.log_softmax(-1)-b.log_softmax(-1))).sum(-1).mean())
    return result

def selected_batches(data,split,count):
    iterator=iter(data.batches(split))
    for _ in range(count):
        try:
            image,sid=next(iterator)
        except StopIteration:
            raise RuntimeError('Insufficient '+split+' samples')
        require(torch.is_tensor(image) and image.shape[0]==1 and image.dtype==torch.float32,'Expected FP32 batch size one')
        yield image,sid

def comparison(arms,left,right):
    differences=[]
    for a,b in zip(arms[left]['search_per_image'],arms[right]['search_per_image']):
        require(a['sample_id']==b['sample_id'],'Unpaired comparison')
        fa,fb=flatten_metrics(a['metrics']),flatten_metrics(b['metrics'])
        differences.append({'sample_id':a['sample_id'],'overlaps_calibration':a['overlaps_calibration'],'left_minus_right':{k:fa[k]-fb[k] for k in fa}})
    def summarize(rows):
        if not rows:
            return None
        return {k:{'mean_left_minus_right':sum(r['left_minus_right'][k] for r in rows)/len(rows),'left_lower':sum(r['left_minus_right'][k]<0 for r in rows),'equal':sum(r['left_minus_right'][k]==0 for r in rows),'left_higher':sum(r['left_minus_right'][k]>0 for r in rows)} for k in rows[0]['left_minus_right']}
    return {'left':left,'right':right,'per_image':differences,'all':summarize(differences),'overlap':summarize([r for r in differences if r['overlaps_calibration']]),'nonoverlap':summarize([r for r in differences if not r['overlaps_calibration']]),'interpretation':'Lower favorable for errors/KL only; signed row bias descriptive.'}

def execute(experiment_payload,context):
    backend,data=context['backend'],context['data']
    reference=context['reference_model']
    execution=experiment_payload['execution']
    params=execution['parameters']
    resolved=execution.get('resolved_choices',{})
    require(isinstance(resolved,dict),'Invalid resolved choices')
    declarations=[m['training'] for m in (execution,resolved) if 'training' in m]
    require(bool(declarations) and all(x is False for x in declarations),'Explicit consistent training=false required')
    require(execution['calibration']=={'count':32,'offset':0,'split':'calibration'},'Calibration protocol mismatch')
    require(params['block']=='blocks.5' and params['site']=='blocks.5.attn::post_softmax','Site mismatch')
    require(params['coordinate_sweeps']==2 and params['search_count']==64 and params['search_offset']==0,'Search protocol mismatch')
    require(params['log_base_q_candidates']==[19,37,55,74] and params['scale_factors']==[0.0625,0.125,0.25,0.5,1.0] and params['nonsemantic_group_seeds']==[42,43,44,45,46],'Candidate protocol mismatch')
    require(all(not m.training for m in reference.modules()),'Reference must be eval')
    require(all(not p.requires_grad for p in reference.parameters()),'Reference must be frozen')
    require(all(t.dtype==torch.float32 for t in list(reference.parameters())+list(reference.buffers()) if t.is_floating_point()),'Expected FP32 source')
    source_hash=model_hash(reference)
    source_flags=[(n,m.training) for n,m in reference.named_modules()]
    grad_flags=[p.requires_grad for p in reference.parameters()]
    head_hash=model_hash(reference.get_submodule('head'))
    model=copy.deepcopy(reference).eval().requires_grad_(False)
    device=next(model.parameters()).device
    candidate_hash=model_hash(model)
    install_attention(model,['blocks.5.attn'])
    edge=model.get_submodule('blocks.5.attn')._ptq_probability
    require(not hasattr(edge,'quant_a') and not hasattr(edge,'quant_b'),'Unexpected existing quantization')
    calibration_ids=list(data.ids('calibration'))[:32]
    search_ids=list(data.ids('search'))[:64]
    require(len(calibration_ids)==len(set(calibration_ids))==32 and len(search_ids)==len(set(search_ids))==64,'Invalid sample identities')
    expected={'calibration':'746d2f3a9e48bd2dd3f2d4f7eba9fc97b220891bac238f80914e572ded639a25','search':'8c4d7b2166b81be1ace62d273bf5e511b18a35218a2ed932500a881f5c396b17'}
    identity_checks={s:{'actual_ordered_sha256':digest(ids),'C8_ordered_sha256':expected[s],'matches':digest(ids)==expected[s]} for s,ids in [('calibration',calibration_ids),('search',search_ids)]}
    require(all(x['matches'] for x in identity_checks.values()),'C8 sample identity mismatch')
    records=[]
    replay={'calibration':[],'search':[]}
    arms,states,quantizers,originals,checks={},{},{},{},{}
    maximum=0.0
    try:
        with torch.inference_mode():
            for i,(image,sid) in enumerate(selected_batches(data,'calibration',32)):
                require(sid==calibration_ids[i],'Calibration order mismatch')
                fp=capture(backend,reference,image)
                adapted=capture(backend,model,image,True,True)
                check=check_replay(fp,adapted)
                args,kwargs=adapted['block_call']
                out=boundary(model.get_submodule('blocks.5')(*args,**kwargs))
                check['block_call_replay']={'max_abs':float((out-adapted['blocks.5']).abs().max()),'error':error(adapted['blocks.5'],out)}
                require(torch.equal(out,adapted['blocks.5']),'Exact block/RoPE replay failed')
                replay['calibration'].append({'sample_id':sid,'checks':check})
                maximum=max(maximum,float(adapted['p'].max()))
                records.append({'sample_id':sid,'p':adapted['p'].cpu(),'v':adapted['v'].cpu()})
                shapes={'p':list(adapted['p'].shape),'v':list(adapted['v'].shape),'block_input':list(adapted['block_input'].shape),'block_output':list(adapted['blocks.5'].shape)}
            require(maximum==0.9985849857330322,'C8 calibration maximum mismatch')
            books=make_books(maximum,params['log_base_q_candidates'],params['scale_factors'])
            members={'shared':[list(range(201))],'prefix':[list(range(5)),list(range(5,201))],'semantic':[[0],list(range(1,5)),list(range(5,201))]}
            permutations={}
            for seed in params['nonsemantic_group_seeds']:
                p=torch.randperm(201,generator=torch.Generator(device='cpu').manual_seed(seed)).tolist()
                name='nonsemantic_'+str(seed)
                permutations[name]=p
                members[name]=[p[:1],p[1:5],p[5:]]
            stats={name:candidate_statistics(records,m,books,device) for name,m in members.items()}
            fits={}
            for objective in ('pv_mse','probability_mse'):
                c,t=exhaustive(stats['shared'],objective)
                fits['shared_'+objective]=('shared',objective,c,t)
            shared_initial=fits['shared_pv_mse'][2][0]
            for name in members:
                if name=='shared':
                    continue
                c,t=exhaustive(stats[name],'pv_mse')
                fits[name+'_pv_exact']=(name,'pv_mse',c,t)
                if len(members[name])==3:
                    cc,ct=optimize(stats[name],[shared_initial]*3,'pv_mse',2)
                    ct['exhaustive_objective']=t['final_objective']
                    ct['coordinate_minus_exhaustive']=ct['final_objective']-t['final_objective']
                    fits[name+'_pv_coordinate']=(name,'pv_mse',cc,ct)
            aliases={}
            mapping_owners={}
            for name,(grouping,objective,choices,trace) in fits.items():
                state={'n':201,'grouping':grouping,'objective':objective,'members':members[grouping],'candidate_ids':choices,'books':[copy.deepcopy(books[c]) for c in choices],'rounding':'ties_to_even','row_renormalization':False}
                restored=json.loads(canonical(state))
                require(restored==state,'State serialization changed values')
                originals[name]=FrozenProbability(state).to(device)
                quantizers[name]=FrozenProbability(restored).to(device)
                states[name]=restored
                mapping=[None]*201
                for member,book in zip(state['members'],state['books']):
                    for j in member:
                        mapping[j]={'q':book['q'],'s':book['s'],'lut':book['lut']}
                mapping_hash=digest(mapping)
                owner=mapping_owners.setdefault(mapping_hash,name)
                aliases[name]=owner
                arms[name]={'grouping':grouping,'objective':objective,'fit':trace,'selected_candidate_ids':choices,'mapping_sha256':mapping_hash,'measurement_owner':owner,'deduplication_reason':None if owner==name else 'Identical per-key encoder q/s and LUT, not an independent measurement','selected_boundary_flags':[{'q_at_grid_boundary':books[c]['q'] in (19,74),'scale_at_grid_boundary':books[c]['scale_factor'] in (0.0625,1.0)} for c in choices],'calibration_per_image':[],'search_per_image':[]}
                checks[name]={'json_round_trip_equal':True,'frozen_hash_before':digest(restored),'calibration_equal_samples':0,'search_equal_samples':0}
            for row in records:
                p,v=row['p'].to(device),row['v'].to(device)
                cached={}
                for name,q in quantizers.items():
                    qp=q(p)
                    require(torch.equal(qp,originals[name](p)),'Reconstructed calibration output mismatch')
                    checks[name]['calibration_equal_samples']+=1
                    owner=aliases[name]
                    if owner not in cached:
                        cached[owner]=local_metrics(p,v,qp)
                    else:
                        require(torch.equal(qp,quantizers[owner](p)),'Deduplicated output mismatch')
                    arms[name]['calibration_per_image'].append({'sample_id':row['sample_id'],'metrics':copy.deepcopy(cached[owner])})
            audits={}
            for name,(grouping,objective,choices,trace) in fits.items():
                direct=arms[name]['calibration_per_image']
                sse=sum(r['metrics']['pv']['all']['sse'] for r in direct)
                count=sum(r['metrics']['pv']['all']['elements'] for r in direct)
                actual=sse/count
                predicted=objective_score(stats[grouping],choices,'pv_mse')
                difference=abs(predicted-actual)
                tolerance=1e-9+1e-5*abs(actual)
                audits[name]={'gram_pv_objective':predicted,'direct_reconstructed_pv_objective':actual,'direct_sse':sse,'elements':count,'absolute_difference':difference,'tolerance':tolerance,'passed':difference<=tolerance}
                if objective=='probability_mse':
                    pooled=aggregate([r['metrics']['probability']['all']['all_keys'] for r in direct])
                    audits[name]['direct_probability_objective']=pooled['mse']
            audit_ok=all(a['passed'] for a in audits.values())
            overlap=set(calibration_ids).intersection(search_ids)
            for i,(image,sid) in enumerate(selected_batches(data,'search',64)):
                require(sid==search_ids[i],'Search order mismatch')
                fp=capture(backend,reference,image)
                adapted=capture(backend,model,image,True)
                replay['search'].append({'sample_id':sid,'checks':check_replay(fp,adapted)})
                cache={}
                for name,q in quantizers.items():
                    q.verify(states[name])
                    qp=q(adapted['p'])
                    require(torch.equal(qp,originals[name](adapted['p'])),'Reconstructed search output mismatch')
                    checks[name]['search_equal_samples']+=1
                    owner=aliases[name]
                    if owner not in cache:
                        local=local_metrics(adapted['p'],adapted['v'],qp)
                        edge.add_module('quant_a',q)
                        try:
                            actual=capture(backend,model,image,True)
                        finally:
                            delattr(edge,'quant_a')
                        require(torch.equal(actual['p'],adapted['p']) and torch.equal(actual['v'],adapted['v']),'Paired P/V inputs changed')
                        cache[owner]={'local':local,'downstream':downstream_metrics(fp,actual)}
                    else:
                        require(torch.equal(qp,quantizers[owner](adapted['p'])),'Search dedup mismatch')
                    arms[name]['search_per_image'].append({'sample_id':sid,'overlaps_calibration':sid in overlap,'metrics':copy.deepcopy(cache[owner])})
            summaries={}
            for name,arm in arms.items():
                rows=arm['search_per_image']
                arm['calibration_aggregate']=aggregate([r['metrics'] for r in arm['calibration_per_image']])
                arm['search_aggregate']=aggregate([r['metrics'] for r in rows])
                arm['search_overlap_aggregate']=aggregate([r['metrics'] for r in rows if r['overlaps_calibration']])
                arm['search_nonoverlap_aggregate']=aggregate([r['metrics'] for r in rows if not r['overlaps_calibration']])
                quantizers[name].verify(states[name])
                checks[name]['frozen_hash_after']=quantizers[name].state_hash
                summaries[name]={'selected_candidate_ids':arm['selected_candidate_ids'],'measurement_owner':aliases[name],'gram_fit_objective':arm['fit']['final_objective'],'score_audit':audits[name],'calibration':arm['calibration_aggregate'],'search':arm['search_aggregate'],'search_overlap':arm['search_overlap_aggregate'],'search_nonoverlap':arm['search_nonoverlap_aggregate']}
            pairs=[('prefix_pv_exact','semantic_pv_exact'),('shared_pv_mse','shared_probability_mse')]
            for grouping in members:
                if grouping!='shared':
                    pairs.append((grouping+'_pv_exact','shared_pv_mse'))
                if len(members[grouping])==3:
                    pairs.append((grouping+'_pv_exact',grouping+'_pv_coordinate'))
                if grouping.startswith('nonsemantic_'):
                    pairs.append(('semantic_pv_exact',grouping+'_pv_exact'))
            comparisons={a+'_vs_'+b:comparison(arms,a,b) for a,b in pairs}
            a=arms['prefix_pv_exact']['search_aggregate']
            b=arms['semantic_pv_exact']['search_aggregate']
            screening={}
            for label,x,y in [('pv_nmse',a['local']['pv']['all']['nmse'],b['local']['pv']['all']['nmse']),('teacher_kl',a['downstream']['teacher_kl'],b['downstream']['teacher_kl']),('block11_patch_nmse',a['downstream']['block11_patch']['nmse'],b['downstream']['block11_patch']['nmse'])]:
                screening[label]={'prefix':x,'semantic':y,'prefix_minus_semantic':x-y,'passes_one_percent_rule':x<=y+0.01*abs(y)}
            require(model_hash(model)==candidate_hash,'Non-target model state changed')
            return {'observations':{'status':'completed','scope':'Only blocks.5 PV probability input quantized; other model and heads FP32; not complete W4A4','identity':backend.identity(),'source_state_sha256':source_hash,'head_state_sha256':head_hash,'shapes':shapes,'calibration_maximum_M':maximum,'C8_maximum_exact_match':True,'C8_sample_identity_checks':identity_checks,'candidate_codebooks':books,'candidate_order':'q outer, scale factor inner','group_members':members,'nonsemantic_permutations':permutations,'states':states,'state_hashes':{n:digest(s) for n,s in states.items()},'reconstruction_checks':checks,'score_audits':audits,'all_score_audits_passed':audit_ok,'optimality_scope':'Finite-grid Gram optimum only; never continuous/task optimum. Direct FP32 consistency passed for all selected solutions.' if audit_ok else 'Finite-grid Gram minimizers only. Direct FP32 consistency failed; no actual-operator optimality claim; inspect score_audits before promotion.','sample_ids':{'calibration':calibration_ids,'search':search_ids,'intersection':[s for s in search_ids if s in overlap]},'replay_checks':replay,'replay_tolerance':{'atol':1e-4,'rtol':1e-4,'block_call':'bitwise equality'},'arm_summaries':summaries,'arms':arms,'paired_comparisons':comparisons,'simplification_screen':{'same_per_key_codec_mapping':arms['prefix_pv_exact']['mapping_sha256']==arms['semantic_pv_exact']['mapping_sha256'],'metrics':screening,'all_three_within_one_percent':all(v['passes_one_percent_rule'] for v in screening.values()),'cls_storage':'See arm_summaries local pv cls/storage and paired_comparisons; screening is exploratory, not equivalence.'},'numerics':{'round':'ties_to_even','encoding_and_error_contributions':'FP32','Gram_and_SSE_accumulation':'FP64','cross_group_terms':True,'nmse_denominator_floor':1e-12,'aggregation':'pooled SSE/energy/count for errors; image mean for KL and row statistics','exhaustive_cost':'shared 20 per objective; prefix 400; semantic and each of five random partitions 8000; coordinate 120 per three-group partition','state_reload':'local JSON reconstruction only; no complete model export','trace_storage':'all_scores indexed by lexicographic tuples; full traces and per-image rows in host result asset'},'limitations':['C8 calibration and search samples reused; not independent validation.','Five random partitions are algorithm controls within one execution, not independent data/execution replicates.','NMSE and teacher KL are not Top-1 or ADE mIoU.','Only calibration used to fit; no labels, ADE or final consumed.','Selected-state score audit does not establish direct-operator optimality for all unselected combinations.']}}
    finally:
        if hasattr(edge,'quant_a'):
            delattr(edge,'quant_a')
        require(model_hash(reference)==source_hash,'Protected FP source changed')
        require(model_hash(reference.get_submodule('head'))==head_hash,'Protected head changed')
        require(source_flags==[(n,m.training) for n,m in reference.named_modules()],'Protected FP mode changed')
        require(grad_flags==[p.requires_grad for p in reference.parameters()],'Protected gradient flags changed')
        backend.reset()
