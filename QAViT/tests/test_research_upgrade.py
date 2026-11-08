"""Scientific capability regression: real tiny math and deterministic lifecycle fixtures."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_infrastructure import TempCase
from core.cards import ContractError, fingerprint
from core.journal import atomic_json, read_json, file_sha256
from core.static_w4a4 import check_plan, check_audit, native_audit
from core.solutions import rank, SolutionLibrary
from tools.runtime_interface import execute_custom, EngineeringBlocked, interface_fixture
from tests.support import experiment_payload


class UpgradeTests(TempCase):
    def test_zero_reserved_actual_codebooks_and_old_semantics(self):
        import torch
        from ptq_agent.execution.plugin_math import activation_qdq
        from ptq_agent.execution.adalog import apply, parameters
        from ptq_agent.execution.recipes import spec
        s = spec(4, False)
        for name, multiplier in [('log2',1), ('log_sqrt2',2)]:
            x = torch.cat([torch.zeros(1), 2. ** (-torch.arange(16)/multiplier)])
            p = {'scale': torch.tensor(1.)}
            self.assertEqual(activation_qdq(x,s,method=name,params=p).unique().numel(),17)
            y = activation_qdq(x,s,method=name,params=p,options={'reserve_zero':True})
            self.assertEqual(y.unique().numel(),16)
            self.assertEqual(y[0],0)
        x = torch.cat([torch.zeros(1), 2. ** -torch.arange(16)])
        p=parameters(torch.tensor(1.),4)
        self.assertEqual(apply(x,s,p).unique().numel(),17)
        self.assertEqual(apply(x,s,p,{'reserve_zero':True}).unique().numel(),16)

    def test_affine_constant_channel_extension(self):
        import torch
        from ptq_agent.execution.grids import affine_params, affine_qdq
        for value in (-2., 0., 3.):
            x=torch.tensor(value)
            p=affine_params(x,x,4)
            self.assertTrue(torch.isfinite(p['scale']))
            self.assertTrue(torch.allclose(affine_qdq(x,4,p),x))

    def test_gptaq_adaptation_inside_compensation(self):
        import torch
        from ptq_agent.execution.gptaq import fit
        from ptq_agent.execution.recipes import spec
        from ptq_agent.execution.plugin_math import decode_weight
        s=spec(4,True,'per_channel',0)
        torch.manual_seed(31)
        w=torch.randn(3,16); x=torch.randn(16,20); h=x@x.T
        for minimum in (-8,-7):
            r,y=fit(w,h,torch.randn(16,16)*.1,s,{'gptaq_signed_min':minimum})
            codes=(y/r['params']['scale']).round()
            self.assertGreaterEqual(codes.min(),minimum)
            self.assertLessEqual(codes.max(),7)
            self.assertTrue(torch.allclose(decode_weight(r,'cpu'),y))

    def test_boundary_contract_does_not_fix_granularity_or_head(self):
        from ptq_agent.execution.recipes import spec
        catalog={n:{'name':n,'kind':k,'edge':n+':input'} for n,k in [('fc','linear'),('head','linear'),('patch','conv2d')]}
        plan={'execution':{'activation_calibration':'static'},'layers':[{'module':n,'kind':catalog[n]['kind'],
            'weight':spec(4,True,'per_channel',0),'activation':spec(4,False)} for n in ('fc','patch')]}
        check_plan(plan,catalog)
        plan['layers'][0]['activation']=None
        with self.assertRaises(ContractError):check_plan(plan,catalog)
        plan['layers'][0]['activation']=spec(4,False)
        plan['execution']['activation_calibration']='dynamic'
        with self.assertRaises(ContractError):check_plan(plan,catalog)

    def test_internal_precision_and_log_capacity(self):
        from ptq_agent.execution.recipes import spec
        catalog={'fc':{'name':'fc','kind':'linear','edge':'fc:input'},
                 'attn::attention_logits':{'name':'attn','kind':'attention_logits','edge':'attn:logits'}}
        plan={'execution':{'activation_calibration':'static'},'layers':[
            {'module':'fc','kind':'linear','weight':spec(4,True),'activation':spec(4,False)},
            {'module':'attn','kind':'attention_logits','activation':spec(8,True)}]}
        check_plan(plan,catalog)
        plan['layers'][0]['activation_method']='log2'
        with self.assertRaises(ContractError):check_plan(plan,catalog)
        plan['layers'][0]['options']={'reserve_zero':True}
        check_plan(plan,catalog)

    def test_repq_fractional_endpoint_capacity_is_not_sample_unique(self):
        import torch
        from ptq_agent.execution.pipeline import ActivationQuantizer
        from ptq_agent.execution.plugin_math import activation_qdq
        from ptq_agent.execution.recipes import spec
        layer={'module':'fc','kind':'linear','weight':None,'activation':spec(4,False),'activation_method':'uniform_affine'}
        plan={'layers':[layer],'execution':{'activation_calibration':'static'}}
        catalog={'fc':{'name':'fc','kind':'linear','edge':'fc:input'}}
        p={'scale':torch.tensor(1.),'zero':torch.tensor(3.25),'repq_affine':torch.tensor(True)}
        model=ActivationQuantizer(layer,plan['execution'],p)
        values=activation_qdq(torch.arange(-30.,31.),layer['activation'],'uniform_affine',p)
        self.assertEqual(values.unique().numel(),17)
        with self.assertRaises(ContractError):
            native_audit(model,plan,catalog,{'required_boundaries':['fc/activation']})
        p['zero']=p['zero'].round();model.set_params(p)
        native_audit(model,plan,catalog,{'required_boundaries':['fc/activation']})

    def test_evaluation_list_normalization_keeps_actual_receipts(self):
        payload=experiment_payload()
        backend,cfg,data=interface_fixture(payload['execution'],{'shape':[1,3,16,16],
            'sites':['blocks.0.mlp.fc1'],'seed':3,'counts':{'calibration':2,'search':1,'final':1}})
        payload['execution'].pop('calibration',None)
        payload['execution']['evaluation']['count']=1
        def execute(p,c):
            ids=[sample for _,sample in c['data'].batches('search')]
            return {'observations':{'evaluations':[{'kind':'classification','split':'search',
                'sample_ids':ids,'sample_count':len(ids),'full_dataset':False,'metrics':{'top1':0.}}]}}
        value=execute_custom(SimpleNamespace(execute=execute),payload,backend,cfg,data,self.root)
        row=value['observations']['evaluations']['classification']
        self.assertEqual(row['sample_ids'],data.ids('search'))
        self.assertFalse(row['full_dataset'])
        payload['execution'].pop('evaluation')
        empty=SimpleNamespace(execute=lambda *a:{'observations':{'statistic':1,'evaluations':[]}})
        self.assertEqual(execute_custom(empty,payload,backend,cfg,data,self.root)['observations']['evaluations'],{})

    def test_custom_audit_counts_zero_and_freezes(self):
        required={'required_boundaries':['a/activation']}
        row={'bits':4,'max_codewords':16,'includes_zero':True}
        report={'frozen':True,'boundaries':{'a/activation':row}}
        check_audit(report,required)
        row['max_codewords']=17
        with self.assertRaises(ContractError):check_audit(report,required)

    def test_native_v2_rtn_and_classifier_exception(self):
        import torch
        from ptq_agent.execution import codecs
        from ptq_agent.execution.model import QuantizedLinear
        from ptq_agent.execution.recipes import spec
        from ptq_agent.execution.baselines import build_baseline
        from ptq_agent.execution.sites import site_catalog
        s=spec(4,True)
        row={'module':'fc','method':'rtn','weight':s,'activation':s}
        execution={'mode':'reference_qdq','compute_dtype':'FP32','activation_calibration':'static'}
        model=torch.nn.Module();fc=torch.nn.Linear(4,4);x=torch.randn(2,4)
        model.add_module('fc',QuantizedLinear(fc,row,execution,codecs.encode_weight(fc.weight,s),codecs.fit(x,s)))
        catalog={'fc':{'name':'fc','kind':'linear','edge':'fc:input'}}
        plan={'schema_version':2,'execution':execution,'layers':[row]}
        required=check_plan(plan,catalog)
        native_audit(model,plan,catalog,required)
        self.assertTrue(torch.isfinite(model.fc(x)).all())
        backend,_,_=interface_fixture({}, {'shape':[1,4],'sites':['fc','classifier'],'seed':3,
            'counts':{'calibration':2,'search':1,'final':1}})
        for name in ('repq','gptaq','adalog'):
            plan=build_baseline(backend,name)
            self.assertEqual([r['module'] for r in plan['layers']],['fc'])
            check_plan(plan,site_catalog(backend))

    def test_three_complete_native_baselines_prepare_reload(self):
        import torch
        from ptq_agent.execution.baselines import build_baseline
        from ptq_agent.execution.sites import site_catalog
        from ptq_agent.execution.model import RTNExecutor
        from tools.experiment_worker import plan_constraints
        payload=experiment_payload()
        payload['execution']['backend']={'config_path':'QAViT/configs/deitb_backend.json'}
        backend,cfg,data=interface_fixture(payload['execution'],{'shape':[1,3,16,16],
            'sites':['patch_embed.proj','blocks.0.attn.qkv','blocks.0.mlp.fc2'],'seed':3,
            'counts':{'calibration':2,'search':1,'final':1}})
        backend.identity = lambda: {'fixture': 'seed3', 'state': backend.original_state}
        for name in ('repq','gptaq','adalog'):
            with self.subTest(name=name):
                plan=build_baseline(backend,name)
                coverage=plan_constraints(plan,{},backend,cfg,{'protocol':'static_w4a4/1'})
                executor=RTNExecutor(backend,cfg,data)
                model,w,a,audit=executor.prepare(plan,{})
                native_audit(model,plan,site_catalog(backend),coverage['static_w4a4'])
                folder=self.root/name
                executor.export(model,plan,w,a,folder,audit)
                loaded,_=RTNExecutor(backend,cfg,data).load(folder)
                x,_=next(data.batches('search'))
                self.assertTrue(torch.equal(model(x),loaded(x)))
                # norm1 output has a shape/getattr use; native tensor-edge checking still works.
                if name=='repq':
                    self.assertEqual(len(plan['preprocess']),2)

    def test_disjoint_repq_ptf_composition_remains_executable(self):
        from ptq_agent.execution.baselines import build_baseline
        from ptq_agent.execution.recipes import spec
        from ptq_agent.execution.model import RTNExecutor
        from ptq_agent.schemas import PTQError
        payload=experiment_payload()
        payload['execution']['backend']={'config_path':'QAViT/configs/deitb_backend.json'}
        backend,cfg,data=interface_fixture(payload['execution'],{'shape':[1,3,16,16],
            'sites':['patch_embed.proj','blocks.0.attn.qkv','blocks.0.mlp.fc2'],'seed':3,'counts':{'calibration':2,'search':1,'final':1}})
        plan=build_baseline(backend,'repq')
        prep=next(p for p in plan['preprocess'] if p['module']=='blocks.0.mlp.fc1')
        plan['preprocess'].remove(prep)
        layer=next(l for l in plan['layers'] if l['module']=='blocks.0.mlp.fc1')
        layer.update(activation=spec(4,False,'per_channel',2),activation_method='ptf',activation_calibration={'strategy':'minmax'})
        RTNExecutor(backend,cfg,data).prepare(plan,{})
        plan['preprocess'].append(prep)
        with self.assertRaises(PTQError):RTNExecutor(backend,cfg,data).prepare(plan,{})

    def test_blocked_return_not_measurement_or_coding_retry(self):
        from core.experiment_requests import ExperimentRequests
        import torch
        payload={'execution':{'parameters':{},'resolved_choices':{}}}
        data=SimpleNamespace(labels={},identity={'split_ids':{}})
        plugin=SimpleNamespace(execute=lambda p,c:{'observations':{'status':'engineering_blocked','reason':'incompatible graph'}})
        with self.assertRaises(EngineeringBlocked):
            execute_custom(plugin,payload,SimpleNamespace(model=torch.nn.Linear(2,2)),{},data,self.root)
        self.assertFalse(ExperimentRequests.revisable_failure({'error':{'code':'EngineeringBlocked','origin':'implementation'}}))

    def test_top3_persistence_protocol_and_miou_priority(self):
        wf=self.workflow(1)
        self.assertEqual(len(wf.solutions.snapshot()),3)
        self.assertTrue(all(r['metrics']=={'top1':None,'mIoU':None} for r in wf.solutions.snapshot()))
        for row in wf.solutions.catalog():self.assertTrue(Path(row['configuration_asset']).is_file())
        def promote(round_,top1,miou,valid=True):
            evidence={'id':'e'+str(round_),'payload':{'measurement_source':'real','status':'completed',
                'provenance':{'model_ref':'m'+str(round_)},'observations':{'evaluations':{
                'classification':{'full_dataset':True,'metrics':{'top1':top1}},
                'segmentation':{'full_dataset':valid,'metrics':{'mIoU':miou}}}},'artifacts':[]}}
            experiment={'id':'x','payload':{'level':'L3','execution':{}}}
            selection={'id':'s','payload':{'selected_slot':'H1','selected_hypothesis_ref':'h'}}
            wf.solutions.consider_round(round_,evidence,experiment,selection,validator=lambda *a,**k:None)
        promote(1,80,40);promote(2,81,39);promote(3,79,41);promote(4,99,0)
        self.assertEqual([r['metrics']['mIoU'] for r in wf.solutions.snapshot()],[41,40,39])
        before=wf.solutions.snapshot();promote(5,100,99,False)
        self.assertEqual(before,wf.solutions.snapshot())
        restored=SolutionLibrary(wf.journal,wf.brief['payload'],shared=False)
        self.assertEqual(before,restored.snapshot())
        self.assertTrue((restored.root/'top3.json').is_file())
        self.assertEqual(read_json(restored.root/'top3.json')['entries'],before)
        # A new run reads the same durable protocol bucket; it gets exact source assets.
        fresh=self.journal('second')
        atomic_json(fresh.root/'research_run.json',read_json(wf.run_dir/'research_run.json'))
        second=SolutionLibrary(fresh,wf.brief['payload'],shared=False);second.root=restored.root
        second.initialize()
        self.assertEqual(second.snapshot(),before)
        self.assertIn(str(wf.run_dir/'research_run.json'),second.assets())


    def test_cross_request_source_and_failure_handoff(self):
        from tools.safeguards import write_artifact
        wf=self.workflow(1)
        # Same researcher, a failed previous request and a fresh request.
        author=wf.enqueue_role('generation','generate',refs=[wf.brief['id']],context={'round':1,'slot':'H1','hypothesis_type':'existing_method'})
        draft=experiment_payload();draft['execution'].update(requires_code=True,interface='quantizer_v1')
        prior=wf.queue.enqueue('prior','experiment_request',inputs={'parent_task':author,'requester':'generation','refs':[wf.brief['id']],'draft':draft})
        request=wf.broker.request(draft,requester='generation')
        wf.journal.commit('request',cards=[request]);wf.queue.change(prior,'ready')
        wf.queue.change(prior,'running',patch={'request_ref':request['id']})
        code=wf.enqueue_role('prior/code','code',refs=[request['id']],context={'experiment_ref':request['id']},parent_task=prior)
        artifact=write_artifact(wf.run_dir,request,{'interface':'quantizer_v1','source':'# PRIOR_SOURCE\ndef fit(x,parameters): return {}\ndef apply(x,state,parameters): return x\n'},'coding')
        wf.queue.change(code,'ready');wf.queue.change(code,'running')
        wf.queue.finish(code,result={'artifact':artifact})
        wf.queue.finish(prior,result={'status':'failed'})
        current=wf.queue.enqueue('current','experiment_request',inputs={'parent_task':author,'requester':'generation','refs':[wf.brief['id']],'draft':draft})
        handoff=wf.implementation_handoff(wf.queue.get(current))
        self.assertEqual(handoff[0]['candidate']['source_sha256'],artifact['source_sha256'])
        coding=wf.enqueue_role('current/code','code',refs=[request['id']],context={'experiment_ref':request['id'],'implementation_history':handoff},parent_task=current)
        self.assertIn(artifact['path'],wf.authorized_assets(wf.queue.get(coding)))
        self.assertIn(request['id'],wf.memory.recording_batch()['source_refs'])

    def test_frozen_loader_does_not_prepare_and_checks_hashes(self):
        import torch
        from tools.frozen import load_frozen
        folder=self.root/'model';folder.mkdir();atomic_json(folder/'manifest.json',{'test':'fixture'})
        code=self.root/'loader.py';code.write_text('import torch\nclass E:\n def load(self,p): return torch.nn.Linear(2,2), {}\n def prepare(self,*a): raise AssertionError("must not fit")\ndef make_executor(b,c,d): return E()\n')
        binding={'evidence_ref':'ev','model_ref':'m','model_dir':str(folder),
            'candidate':{'path':str(code),'source_sha256':file_sha256(code)},
            'assets':[{'path':str(folder/'manifest.json'),'sha256':file_sha256(folder/'manifest.json')}]}
        _,receipt=load_frozen(binding,SimpleNamespace(reset=lambda:None),{},SimpleNamespace())
        self.assertTrue(receipt['loaded_without_prepare'])
        code.write_text('tampered')
        with self.assertRaises(ContractError):load_frozen(binding,SimpleNamespace(),{},SimpleNamespace())

if __name__=='__main__':unittest.main()
