import copy
import hashlib
import itertools
import json
import math
import torch
from ptq_agent.execution.baselines import build_baseline
from ptq_agent.execution.model import RTNExecutor
from ptq_agent.execution.sites import install_attention

def require(x,m):
    if not x: raise RuntimeError(m)
def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()
def model_hash(model):
    h=hashlib.sha256()
    for n,t in sorted(model.state_dict().items()):
        t=t.detach().cpu().contiguous()
        h.update(n.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode()); h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def clone_tree(x):
    if torch.is_tensor(x): return x.detach().clone()
    if isinstance(x,tuple): return tuple(clone_tree(v) for v in x)
    if isinstance(x,list): return [clone_tree(v) for v in x]
    if isinstance(x,dict): return {k:clone_tree(v) for k,v in x.items()}
    return x
def boundary(x):
    if torch.is_tensor(x): return x
    require(isinstance(x,(list,tuple)) and len(x)==1,'Single-image list boundary required')
    return boundary(x[0])
def error_record(s,e,n):
    return dict(sse=s,reference_energy=e,elements=n,mse=s/n,nmse=s/max(e,1e-12))
def error(a,b):
    require(a.shape==b.shape,'Metric shape mismatch')
    a,b=a.double(),b.double()
    require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'Nonfinite metric')
    return error_record(float((a-b).square().sum()),float(a.square().sum()),a.numel())
def residual_error(r,e):
    return error_record(float(e.double().square().sum()),float(r.double().square().sum()),r.numel())
def aggregate(rows):
    if not rows: return None
    if isinstance(rows[0],dict):
        if 'sse' in rows[0] and 'reference_energy' in rows[0]:
            return error_record(sum(r['sse'] for r in rows),sum(r['reference_energy'] for r in rows),sum(r['elements'] for r in rows))
        return {k:aggregate([r[k] for r in rows]) for k in rows[0]}
    return sum(rows)/len(rows)
def groups(n):
    return {'all':(0,n),'cls':(0,1),'storage':(1,5),'patch':(5,n)}
def local_metrics(p,v,qp):
    pv=p@v; ev=(qp-p)@v
    result={'probability':{},'pv':{},'row_sum':{}}
    for name,(a,b) in groups(p.shape[-1]).items():
        pp,qq=p[:,:,a:b,:],qp[:,:,a:b,:]
        row={'all_keys':error(pp,qq)}
        for key,(c,d) in groups(p.shape[-1]).items():
            if key!='all': row[key+'_keys']=error(pp[:,:,:,c:d],qq[:,:,:,c:d])
        result['probability'][name]=row
        result['pv'][name]=residual_error(pv[:,:,a:b,:],ev[:,:,a:b,:])
        delta=qq.double().sum(-1)-pp.double().sum(-1)
        result['row_sum'][name]={'signed_bias':float(delta.mean()),'mae':float(delta.abs().mean()),'mse':float(delta.square().mean()),'fp_mean':float(pp.double().sum(-1).mean()),'quantized_mean':float(qq.double().sum(-1).mean())}
    return result
def encode(p,book):
    require(p.dtype==torch.float32,'FP32 probability required')
    ratio=(p/p.new_tensor(book['s'])).clamp(1e-15,1.0)
    codes=torch.round(-torch.log2(ratio)*37.0/book['q']).long()
    y=p.new_tensor(book['lut'])[codes.clamp(0,15)]
    return torch.where((p>0)&(codes<15),y,torch.zeros_like(y))
def make_books(maximum,qs,factors):
    books=[]
    for q in qs:
        for factor in factors:
            scale=maximum*factor; lut=[]
            for c in range(15):
                integer,remainder=divmod(c*q,37)
                lut.append(scale*(2.0**(-integer))*round(30.0*2.0**(-remainder/37.0))/30.0)
            lut.append(0.0); lut=torch.tensor(lut,dtype=torch.float32).tolist()
            require(scale>0 and len(set(lut))<=16,'Invalid four-bit codebook')
            books.append({'id':len(books),'q':q,'s':scale,'scale_factor':factor,'lut':lut,'unique_reconstruction_values':len(set(lut))})
    return books
class FrozenProbability(torch.nn.Module):
    def __init__(self,state):
        super().__init__(); self.state_text=canonical(state); self.state_hash=digest(state)
        self.members=[tuple(g) for g in state['members']]; self.books=copy.deepcopy(state['books']); self.n=state['n']
        require(sorted(i for g in self.members for i in g)==list(range(self.n)),'Invalid partition')
    def forward(self,p):
        require(p.ndim==4 and p.shape[-1]==self.n,'Probability shape mismatch')
        out=torch.empty_like(p)
        for member,book in zip(self.members,self.books):
            ix=torch.tensor(member,device=p.device,dtype=torch.long)
            out.index_copy_(-1,ix,encode(p.index_select(-1,ix),book))
        return out
    def verify(self,state):
        require(self.members==[tuple(g) for g in state['members']] and self.books==state['books'] and self.n==state['n'],'Numerical state changed')
        require(self.state_text==canonical(state) and self.state_hash==digest(state),'Metadata changed')
def candidate_statistics(records,members,books,device):
    g,k=len(members),len(books)
    gram=torch.zeros((g*k,g*k),dtype=torch.float64,device=device); probability=torch.zeros((g,k),dtype=torch.float64,device=device)
    pc,vc=0,0
    for row in records:
        p,v=row['p'].to(device),row['v'].to(device); residuals=[]
        pc+=p.numel(); vc+=p.shape[0]*p.shape[1]*p.shape[2]*v.shape[-1]
        for gi,member in enumerate(members):
            ix=torch.tensor(member,device=device,dtype=torch.long); pg,vg=p.index_select(-1,ix),v.index_select(-2,ix)
            for ci,book in enumerate(books):
                e=encode(pg,book)-pg; probability[gi,ci]+=e.double().square().sum(); residuals.append((e@vg).flatten())
        z=torch.stack(residuals).double(); gram+=z@z.T
    return {'gram':gram.cpu(),'probability':probability.cpu(),'p_count':pc,'pv_count':vc}
def objective_score(stats,choices,objective):
    if objective=='probability_mse': return sum(float(stats['probability'][g,c]) for g,c in enumerate(choices))/stats['p_count']
    k=stats['probability'].shape[1]; ix=torch.tensor([g*k+c for g,c in enumerate(choices)],dtype=torch.long)
    return float(stats['gram'].index_select(0,ix).index_select(1,ix).sum())/stats['pv_count']
def exhaustive(stats,objective):
    g,k=stats['probability'].shape; scores=[]; best=None; selected=None
    for choices in itertools.product(range(k),repeat=g):
        score=objective_score(stats,choices,objective); require(math.isfinite(score),'Nonfinite score'); scores.append(score)
        if best is None or score<best: best,selected=score,list(choices)
    return selected,{'method':'finite_grid_exhaustive','ordering':'lexicographic; first tie','tuple_shape':[k]*g,'all_scores':scores,'candidate_score_count':len(scores),'final_choices':selected,'final_objective':best}
def capture(backend,model,image,calls=False):
    saved,handles={},[]
    edge=model.get_submodule('blocks.5.attn')._ptq_probability
    def save(key,t):
        require(key not in saved,'Repeated edge '+key); saved[key]=t.detach().clone()
    def bh(name):
        def hook(m,args,kwargs,out):
            save(name,boundary(out))
            if calls and name=='blocks.5': saved['block_call']=clone_tree((args,kwargs)); saved['block_input']=boundary(args[0]).detach().clone()
        return hook
    def ph(m,args): save('p',args[0]); save('v_pre',args[1])
    def vh(m,args,out): save('v',out)
    def qh(m,args,out): save('qp',out)
    def oh(m,args,out): save('pv',out)
    def nh(m,args,out): save('normalized',boundary(out))
    def hh(m,args): save('head_input',args[0])
    try:
        for name in ('blocks.5','blocks.8','blocks.11'): handles.append(model.get_submodule(name).register_forward_hook(bh(name),with_kwargs=True))
        handles.append(model.norm.register_forward_hook(nh)); handles.append(model.head.register_forward_pre_hook(hh))
        handles.append(edge.register_forward_pre_hook(ph)); handles.append(edge.register_forward_hook(oh))
        if hasattr(edge,'quant_b'): handles.append(edge.quant_b.register_forward_hook(vh))
        if hasattr(edge,'quant_a'): handles.append(edge.quant_a.register_forward_hook(qh))
        logits=backend.forward_model(model,image)
        require(tuple(logits.shape)==(1,1000),'Bound 1000-class logits required')
        if 'v' not in saved: saved['v']=saved['v_pre'].clone()
        if 'qp' not in saved: saved['qp']=saved['p'].clone()
        require(tuple(saved['p'].shape)==(1,12,201,201) and tuple(saved['v'].shape)==(1,12,201,64),'Unexpected P/V shapes')
        require(torch.equal(saved['head_input'],saved['normalized'][:,0]),'Head must consume normalized CLS')
        require(torch.equal(saved['pv'],saved['qp']@saved['v']),'Actual PV edge mismatch')
        saved['cls']=saved['normalized'][:,0].clone(); saved['logits']=logits.detach().clone()
        return saved
    finally:
        for h in handles: h.remove()
def check_replay(a,b):
    result={}
    for key in ('blocks.5','blocks.8','blocks.11','cls','logits','p','v','qp','pv'):
        result[key]=error(a[key],b[key]); require(torch.equal(a[key],b[key]),'Reload/replay mismatch '+key)
    return result
def downstream_metrics(fp,q):
    result={'block5_output':error(fp['blocks.5'],q['blocks.5']),'block8_patch':error(fp['blocks.8'][:,5:],q['blocks.8'][:,5:]),'block11_patch':error(fp['blocks.11'][:,5:],q['blocks.11'][:,5:]),'x_norm_clstoken':error(fp['cls'],q['cls'])}
    a,b=fp['logits'].double(),q['logits'].double(); result['teacher_kl']=float((a.softmax(-1)*(a.log_softmax(-1)-b.log_softmax(-1))).sum(-1).mean())
    return result
def total_metrics(fp,parent,actual):
    base=parent['p']@parent['v']; target=fp['pv']
    u=base.double()-target.double(); l=actual['pv'].double()-base.double(); total=actual['pv'].double()-target.double()
    fit=(actual['qp']-parent['p'])@parent['v']; result={}
    for name,(a,b) in groups(201).items():
        sl=(slice(None),slice(None),slice(a,b),slice(None)); us,ls,ts=u[sl],l[sl],total[sl]
        result[name]={'total':error(target[sl],actual['pv'][sl]),'upstream_squared_norm':float(us.square().sum()),'local_squared_norm':float(ls.square().sum()),'cross_term':float(2*(us*ls).sum()),'decomposition_residual_squared_norm':float((ts-us-ls).square().sum()),'fit_vs_actual_L':error(ls,fit[sl].double())}
        require(torch.allclose(ts,us+ls,atol=1e-12,rtol=1e-12),'PV decomposition mismatch')
    return result
def flatten_metrics(x,prefix=''):
    out={}
    for k,v in x.items():
        path=prefix+'/'+k
        if isinstance(v,dict): out.update(flatten_metrics(v,path))
        elif isinstance(v,(int,float)) and (k in ('mse','nmse','teacher_kl') or 'row_sum' in path): out[path]=float(v)
    return out
def comparison(arms,left,right):
    rows=[]
    for a,b in zip(arms[left]['search_per_image'],arms[right]['search_per_image']):
        require(a['sample_id']==b['sample_id'],'Unpaired sample'); fa,fb=flatten_metrics(a['metrics']),flatten_metrics(b['metrics'])
        rows.append({'sample_id':a['sample_id'],'left_minus_right':{k:fa[k]-fb[k] for k in fa}})
    summary={k:{'mean_left_minus_right':sum(r['left_minus_right'][k] for r in rows)/len(rows),'left_lower':sum(r['left_minus_right'][k]<0 for r in rows),'equal':sum(r['left_minus_right'][k]==0 for r in rows),'left_higher':sum(r['left_minus_right'][k]>0 for r in rows)} for k in rows[0]['left_minus_right']}
    return {'left':left,'right':right,'per_image':rows,'all':summary,'interpretation':'Errors/KL lower is favorable; signed bias descriptive; one paired execution.'}
def batches(data,split,ids,indices):
    wanted=dict(zip(indices,ids)); last=max(indices)
    for i,(image,sid) in enumerate(data.batches(split)):
        if i in wanted:
            require(sid==wanted[i],'Data ordering mismatch'); require(image.dtype==torch.float32 and image.shape[0]==1,'FP32 batch one required'); yield image,sid
        if i>=last: break

def execute(experiment_payload,context):
    backend,data=context['backend'],context['data']; reference=context['reference_model']; ex=experiment_payload['execution']; p=ex['parameters']
    declarations=[m['training'] for m in (ex,ex.get('resolved_choices',{})) if 'training' in m]
    require(declarations and all(v is False for v in declarations),'Explicit training=false required')
    require(ex['calibration']=={'count':32,'offset':0,'split':'calibration'},'Calibration mismatch')
    require(p['base_solution_id']=='adalog_w4a4' and p['block']=='blocks.5' and p['site']=='blocks.5.attn::post_softmax','Identity mismatch')
    require(p['search_count']==64 and p['search_offset']==64 and p['exclude_calibration_and_previous_search'] is True,'Search protocol mismatch')
    require(p['log_base_q_candidates']==[19,37,55,74] and p['scale_factors']==[0.0625,0.125,0.25,0.5,1.0] and p['nonsemantic_group_seeds']==[42,43,44],'Grid mismatch')
    require(all(not m.training for m in reference.modules()) and all(not t.requires_grad for t in reference.parameters()),'Protected source must be frozen eval')
    require(all(t.dtype==torch.float32 for t in list(reference.parameters())+list(reference.buffers()) if t.is_floating_point()),'FP32 source required')
    source_hash=model_hash(reference); flags=[(n,m.training) for n,m in reference.named_modules()]; gradflags=[t.requires_grad for t in reference.parameters()]; head_hash=model_hash(reference.head)
    cal=list(data.ids('calibration')); search_order=list(data.ids('search')); old=search_order[:64]
    require(len(cal)==32 and len(set(cal))==32,'Exactly 32 calibration IDs required')
    require(digest(old)=='8c4d7b2166b81be1ace62d273bf5e511b18a35218a2ed932500a881f5c396b17','C12 old search identity mismatch')
    require(digest(cal)=='746d2f3a9e48bd2dd3f2d4f7eba9fc97b220891bac238f80914e572ded639a25','Calibration identity mismatch')
    excluded=set(cal)|set(old); selected=[]; rejected=[]
    for i in range(64,len(search_order)):
        sid=search_order[i]
        if sid in excluded: rejected.append({'index':i,'sample_id':sid})
        else: selected.append((i,sid))
        if len(selected)==64: break
    require(len(selected)==64 and len(set(s for i,s in selected))==64,'Insufficient new unique search samples; no backfill')
    search_ids=[s for i,s in selected]; search_indices=[i for i,s in selected]
    edge=None; original=None
    try:
        with torch.inference_mode():
            plan=build_baseline(backend,'adalog',quantize_head=False)
            require(plan['execution']['compute_dtype']=='FP32' and plan['execution']['activation_calibration']=='static','Invalid parent execution')
            require(all(l['activation'] and l['activation']['bits']==4 and (not l['weight'] or l['weight']['bits']==4) for l in plan['layers']),'Parent bitwidth mismatch')
            require(sum(l['kind']=='post_softmax' for l in plan['layers'])==12 and sum(l['kind']=='attention_value' for l in plan['layers'])==12,'Incomplete parent attention coverage')
            executor=RTNExecutor(backend,context['configuration'],data)
            prepared,weights,activations,audit=executor.prepare(plan,{})
            parent_dir=str(context['directory'])+'/adalog_parent'
            manifest=executor.export(prepared,plan,weights,activations,parent_dir,audit)
            loader=RTNExecutor(backend,context['configuration'],data); model,loaded_manifest=loader.load(parent_dir)
            require(model_hash(prepared)==model_hash(model),'Parent persistent state reload mismatch')
            def all_buffers(m): return {n:t.detach().cpu().clone() for n,t in m.named_buffers()}
            ab,bb=all_buffers(prepared),all_buffers(model)
            require(set(ab)==set(bb) and all(torch.equal(ab[k],bb[k]) for k in ab),'Parent full buffer reload mismatch')
            require(model_hash(model.head)==head_hash,'Parent head changed')
            edge=model.get_submodule('blocks.5.attn')._ptq_probability
            require(hasattr(edge,'quant_a') and hasattr(edge,'quant_b') and edge.quant_a.enabled and edge.quant_b.enabled,'Missing P/V quantization')
            original=edge.quant_a; parent_hash=model_hash(model); original_hash=model_hash(original)
            parent_description={'layer':copy.deepcopy(original.layer),'execution':copy.deepcopy(original.execution),'params':{k:v.detach().cpu().tolist() for k,v in original.params().items()},'enabled':original.enabled}
            fpmodel=copy.deepcopy(reference).eval().requires_grad_(False); install_attention(fpmodel,['blocks.5.attn']); fpmodel.eval()
            records=[]; reload_checks=[]; shapes={}; maximum=0.0
            for image,sid in batches(data,'calibration',cal,list(range(32))):
                a=capture(backend,prepared,image); b=capture(backend,model,image,True)
                reload_checks.append({'sample_id':sid,'checks':check_replay(a,b)})
                args,kwargs=b['block_call']; out=boundary(model.get_submodule('blocks.5')(*args,**kwargs)); require(torch.equal(out,b['blocks.5']),'List/RoPE block replay mismatch')
                maximum=max(maximum,float(b['p'].max()))
                records.append({'sample_id':sid,'p':b['p'].cpu(),'v':b['v'].cpu(),'qp_parent':b['qp'].cpu()})
                shapes={k:list(b[k].shape) for k in ('p','v','v_pre','qp','pv','block_input','blocks.5')}
            del prepared,weights,activations,ab,bb
            require(maximum>0 and math.isfinite(maximum),'Invalid calibration maximum')
            device=next(model.parameters()).device; books=make_books(maximum,p['log_base_q_candidates'],p['scale_factors'])
            members={'shared':[list(range(201))],'prefix':[list(range(5)),list(range(5,201))]}; permutations={}
            for seed in (42,43,44):
                perm=torch.randperm(201,generator=torch.Generator(device='cpu').manual_seed(seed)).tolist(); name='random'+str(seed)
                permutations[name]=perm; members[name]=[perm[:5],perm[5:]]
            stats={name:candidate_statistics(records,gs,books,device) for name,gs in members.items()}
            fits={}
            for objective in ('probability_mse','pv_mse'):
                choices,trace=exhaustive(stats['shared'],objective); fits['shared_'+objective]=('shared',objective,choices,trace)
            for grouping in ('prefix','random42','random43','random44'):
                choices,trace=exhaustive(stats[grouping],'pv_mse'); fits[grouping+'_pv']=(grouping,'pv_mse',choices,trace)
            states={}; qs={}; originals={}; checks={}; arms={'original_parent':{'fit_identity':'Original parent calibration; outside common 20-codebook grid','original_quantizer':parent_description,'calibration_per_image':[],'search_per_image':[]}}
            for name,(grouping,obj,choices,trace) in fits.items():
                state={'n':201,'grouping':grouping,'objective':obj,'members':members[grouping],'candidate_ids':choices,'books':[copy.deepcopy(books[c]) for c in choices],'rounding':'ties_to_even','row_renormalization':False}
                restored=json.loads(canonical(state)); require(state==restored,'JSON state mismatch'); states[name]=restored
                originals[name]=FrozenProbability(state).eval().to(device); qs[name]=FrozenProbability(restored).eval().to(device)
                checks[name]={'frozen_hash_before':digest(state),'calibration_equal_samples':0,'search_equal_samples':0}
                arms[name]={'fit':trace,'selected_candidate_ids':choices,'selected_boundary_flags':[{'q_at_grid_boundary':books[c]['q'] in (19,74),'scale_at_grid_boundary':books[c]['scale_factor'] in (0.0625,1.0)} for c in choices],'calibration_per_image':[],'search_per_image':[]}
            for row in records:
                pp,v=row['p'].to(device),row['v'].to(device)
                arms['original_parent']['calibration_per_image'].append({'sample_id':row['sample_id'],'metrics':local_metrics(pp,v,row['qp_parent'].to(device))})
                for name,q in qs.items():
                    qp=q(pp); require(torch.equal(qp,originals[name](pp)),'Calibration reconstruction mismatch'); checks[name]['calibration_equal_samples']+=1
                    arms[name]['calibration_per_image'].append({'sample_id':row['sample_id'],'metrics':local_metrics(pp,v,qp)})
            audits={}
            for name,(grouping,obj,choices,trace) in fits.items():
                actual=aggregate([r['metrics'] for r in arms[name]['calibration_per_image']])['pv']['all']['mse']; predicted=objective_score(stats[grouping],choices,'pv_mse'); tol=1e-9+1e-5*abs(actual)
                audits[name]={'gram_pv_objective':predicted,'direct_reconstructed_pv_objective':actual,'absolute_difference':abs(predicted-actual),'tolerance':tol,'passed':abs(predicted-actual)<=tol}
            for image,sid in batches(data,'search',search_ids,search_indices):
                fp=capture(backend,fpmodel,image); parent=capture(backend,model,image)
                truefp=backend.forward_model(reference,image); require(torch.allclose(fp['logits'],truefp,atol=1e-4,rtol=1e-4),'FP attention adaptation mismatch')
                for name in arms:
                    q=original if name=='original_parent' else qs[name]
                    if name!='original_parent':
                        q.verify(states[name]); require(torch.equal(q(parent['p']),originals[name](parent['p'])),'Search reconstruction mismatch'); checks[name]['search_equal_samples']+=1
                    edge.quant_a=q
                    try: actual=capture(backend,model,image)
                    finally: edge.quant_a=original
                    require(torch.equal(parent['p'],actual['p']) and torch.equal(parent['v'],actual['v']),'Shared Pq/Vq context changed')
                    require(torch.equal(actual['qp'],q(parent['p'])),'Installed P quantizer mismatch')
                    metrics={'local':local_metrics(parent['p'],parent['v'],actual['qp']),'total_pv':total_metrics(fp,parent,actual),'downstream':downstream_metrics(fp,actual)}
                    arms[name]['search_per_image'].append({'sample_id':sid,'overlaps_calibration':False,'metrics':metrics})
                require(model_hash(model)==parent_hash,'Non-target frozen state changed')
            summaries={}
            for name,arm in arms.items():
                arm['calibration_aggregate']=aggregate([r['metrics'] for r in arm['calibration_per_image']]); arm['search_aggregate']=aggregate([r['metrics'] for r in arm['search_per_image']])
                summaries[name]={'calibration':arm['calibration_aggregate'],'search':arm['search_aggregate']}
                if name in qs:
                    qs[name].verify(states[name]); checks[name]['frozen_hash_after']=qs[name].state_hash
            pairs=[('prefix_pv','random42_pv'),('prefix_pv','random43_pv'),('prefix_pv','random44_pv'),('prefix_pv','shared_pv_mse'),('shared_pv_mse','shared_probability_mse')]+[(n,'original_parent') for n in fits]
            require(model_hash(model)==parent_hash and model_hash(original)==original_hash,'Frozen parent changed')
            return {'observations':{'status':'completed','scope':'Seven local P replacements on one complete frozen static W4A4 AdaLog parent; proxy evaluation only','identity':backend.identity(),'source_state_sha256':source_hash,'head_state_sha256':head_hash,'parent':{'model_dir':parent_dir,'package_role':'original_parent_only_not_prefix_candidate','plan':plan,'audit':audit,'manifest':manifest,'reload_manifest':loaded_manifest,'state_sha256':parent_hash,'all_parameters_and_buffers_equal_on_reload':True,'calibration_output_checks':reload_checks,'original_probability_quantizer':parent_description},'shapes':shapes,'calibration_maximum_M':maximum,'candidate_codebooks':books,'group_members':members,'random_permutations':permutations,'states':states,'reconstruction_checks':checks,'score_audits':audits,'all_score_audits_passed':all(a['passed'] for a in audits.values()),'optimality_scope':'Finite-grid Gram minimizers only; consult selected-state direct audits; no continuous or task optimality claim','sample_ids':{'calibration':cal,'previous_search':old,'search':search_ids,'original_indices':search_indices,'excluded_after_offset':rejected,'intersection':[],'search_batch_iteration_range':[0,max(search_indices)]},'arms':arms,'arm_summaries':summaries,'paired_comparisons':{a+'_vs_'+b:comparison(arms,a,b) for a,b in pairs},'numerics':{'Pq':'actual quant_a input','Vq':'actual quant_b output, not matmul pre-hook V','fit':'(Q(Pq)-Pq)@Vq, FP32 contributions, FP64 Gram with cross terms','aggregation':'pooled SSE/energy/count for errors; image mean KL and scalar diagnostics','codec':'C12 reserve_zero, ties-to-even, 15 positive plus zero; no row normalization','search_cost':'20 shared per objective; 400 per two-group arm','state_reload':'parent full standard export/load; new local states JSON only'},'limitations':['No complete custom prefix package exported.','No ADE/final or task accuracy measured.','Random groups are equal-capacity algorithm controls in one execution, not independent repetitions.','Sequential data iteration reads skipped search batches but never forwards or fits on them.','Selected-state direct audit does not establish FP32 optimality of every unselected grid combination.']}}
    finally:
        if edge is not None and original is not None: edge.quant_a=original
        require(model_hash(reference)==source_hash and model_hash(reference.head)==head_hash,'Protected FP source/head changed')
        require(flags==[(n,m.training) for n,m in reference.named_modules()] and gradflags==[t.requires_grad for t in reference.parameters()],'Protected source flags changed')
        backend.reset()
