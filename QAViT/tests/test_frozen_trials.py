"""Paired scores against an independent direct forward and explicit acceptance on tiny native ViT."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from tests.test_calibration_search import fixture
from ptq_agent.execution.pipeline import QuantizationExecutor, quantizers
from ptq_agent.execution.calibration_search import apply_grid_changes, search_frozen_scales, state_identity
from ptq_agent.execution.frozen_trials import score_frozen_candidates, accepted_copy
from ptq_agent.execution.model import forward
from ptq_agent.schemas import PTQError

class FrozenTrials(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b,cls.cfg,cls.data,cls.plan=fixture()
        cls.e=QuantizationExecutor(cls.b,cls.cfg,cls.data)
        cls.source,_,_,_=cls.e.prepare(cls.plan,{})
        cls.site='blocks.0.attn.qkv'
        cls.grid=[{'id':'noop','scales':{}},{'id':'changed','scales':{cls.site:{'activation':.73}}}]
        cls.obs=[{'id':'logit','view':'classification','output':'classification_logits','metrics':['mse','nmse','cosine','teacher_kl']},
                 {'id':'patch','view':'classification','output':'block','block':'blocks.1','tokens':'patch','reference':'local_fp','metrics':['nmse']}]

    def run_grid(self, model, grid, **kw):
        return score_frozen_candidates(self.b,model,self.data,grid,source_ref='test-parent',input_shapes=self.e.input_shapes,
               observations_spec=self.obs,**kw)

    def test_independent_logits_conditional_block_order_and_receipts(self):
        model=deepcopy(self.source)
        original=state_identity(model)
        objectives={'choice':{'logit.teacher_kl':1,'patch.nmse':2}}
        first,second=[self.run_grid(model,g,objectives=objectives) for g in (self.grid,self.grid[::-1])]
        self.assertEqual(first['winners'],second['winners'])
        self.assertEqual(first['views'],second['views'])
        self.assertEqual(first['candidates'],second['candidates'][::-1])
        self.assertEqual(original,state_identity(model))
        for row in first['candidates']:
            q=deepcopy(self.source)
            if row['id']=='changed':
                site=quantizers(q)[self.site]
                pars=site.params()
                site.set_params({**pars,'scale':pars['scale']*.73})
            with torch.inference_mode():
                for x,sid in self.data.batches('calibration'):
                    ref,act=forward(self.b,self.b.model,x).double(),forward(self.b,q,x).double()
                    d=(act-ref).square()
                    expected={'mse':d.mean(),'nmse':d.sum()/ref.square().sum(),
                              'cosine':1-torch.nn.functional.cosine_similarity(ref.flatten(),act.flatten(),dim=0),
                              'teacher_kl':(ref.softmax(1)*(ref.log_softmax(1)-act.log_softmax(1))).sum()}
                    for k,n in expected.items(): self.assertAlmostEqual(n.item(),row['per_image'][sid]['logit.'+k],places=10)
                    captured={}
                    def capture(_m,args,kwargs,_out):
                        captured['ref']=self.b.model.blocks[1](*args,**kwargs)
                    with model.blocks[1].register_forward_hook(capture,with_kwargs=True):
                        forward(self.b,model,x)
                    with q.blocks[1].register_forward_hook(lambda m,a,out: captured.update(act=out)):
                        forward(self.b,q,x)
                    r,a=[captured[k][:,1:].double() for k in ('ref','act')]
                    self.assertAlmostEqual(((r-a).square().sum()/r.square().sum()).item(),row['per_image'][sid]['patch.nmse'],places=10)
            child,receipt=accepted_copy(model,row['receipt'],source_ref='test-parent',input_shapes=self.e.input_shapes)
            self.assertEqual(state_identity(q),receipt['child_state_sha256'])
            self.assertEqual(state_identity(child),state_identity(q))
            if row['id']=='changed':
                with tempfile.TemporaryDirectory() as directory:
                    pkg=Path(directory)/'child'
                    self.e.export(child,self.plan,{}, {},pkg,self.e.audit(child,self.plan))
                    cold,_=QuantizationExecutor(self.b,self.cfg,self.data).load(pkg)
                    self.assertEqual(state_identity(cold),state_identity(q))
                with self.assertRaises(PTQError):
                    accepted_copy(child,row['receipt'],source_ref='test-parent',input_shapes=self.e.input_shapes)
        self.assertEqual(original,state_identity(model))

    def test_noop_failure_rejection_and_default_search(self):
        model=deepcopy(self.source)
        original=state_identity(model)
        q=quantizers(model)[self.site]
        with patch.object(type(q),'set_params',side_effect=AssertionError('unexpected refit/set')):
            noop=self.run_grid(model,[self.grid[0]])['candidates'][0]['receipt']
            _,rec=accepted_copy(model,noop,source_ref='test-parent',input_shapes=self.e.input_shapes)
            self.assertEqual(rec['child_state_sha256'],original)
        bad=deepcopy(self.grid)
        bad[-1]['scales'][self.site]['weight']=-1
        with self.assertRaises(PTQError):self.run_grid(model,bad)
        with self.assertRaises(PTQError):accepted_copy(model,{**noop,'changes':{'invented':1}},source_ref='test-parent',input_shapes={})
        self.assertFalse(search_frozen_scales(self.b,model,self.data,[self.grid[1]])['parameters_installed'])
        self.assertEqual(original,state_identity(model))
        self.assertFalse(model.blocks[1]._forward_hooks)
        self.assertFalse(self.b.model.blocks[1]._forward_hooks)

    def test_local_grid_uses_existing_groups_and_legal_zero(self):
        model=deepcopy(self.source)
        q=quantizers(model)[self.site]
        p=q.weight_record['params']
        self.assertGreater(p['scale'].numel(),1)
        before=p['scale'].clone()
        apply_grid_changes(model,{self.site:{'weight':{'factor':.75,'indices':[0],'zero':0}}},self.e.input_shapes)
        after=q.weight_record['params']
        self.assertTrue(torch.equal(after['scale'].flatten()[1:],before.flatten()[1:]))
        self.assertTrue(torch.equal(after['scale'].flatten()[0],before.flatten()[0]*.75))
        self.assertEqual(after['zero'].flatten()[0],0)
        state=state_identity(model)
        with self.assertRaises(PTQError):
            apply_grid_changes(model,{self.site:{'activation':.9,'weight':{'indices':[0],'zero':.2}}},self.e.input_shapes)
        self.assertEqual(state_identity(model),state)


    def test_approved_custom_exporter_receives_the_measured_child(self):
        from types import SimpleNamespace
        from core.cards import fingerprint
        from core.journal import atomic_json, read_json
        from tools.experiment_worker import evaluate_services
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            parent=root/'source'
            self.e.export(self.source,self.plan,{}, {},parent,self.e.audit(self.source,self.plan))
            def seal(path, ids):
                m=read_json(path/'manifest.json')
                m.update(calibration_ids=ids,preparation_training=False)
                m.pop('content_sha256')
                m['content_sha256']=fingerprint(m)
                atomic_json(path/'manifest.json',m)
            ids=self.data.ids('calibration')
            seal(parent,ids)
            from core.journal import file_sha256
            binding={'model_dir':str(parent),'assets':[{'path':str(f),'sha256':file_sha256(f)} for f in parent.iterdir()],
                     'evidence_ref':'source','model_ref':'test-parent','verified_increment':state_identity(self.source)}
            from tools.experiment_worker import data_view
            selection = {'offset':1, 'count':1}
            scored = score_frozen_candidates(self.b, self.source, data_view(self.data,selection,'calibration'),
                        self.grid, source_ref='test-parent', input_shapes=self.e.input_shapes,
                        observations_spec=self.obs)
            score = scored['candidates'][1]['receipt']
            _,selected=accepted_copy(self.source,score,source_ref='test-parent',input_shapes=self.e.input_shapes)
            class Custom(QuantizationExecutor):
                def prepare_from_parent(this, plan, model, manifest):
                    self.assertEqual(state_identity(model),selected['child_state_sha256'])
                    this.input_shapes=manifest['input_shapes']
                    return model, {}, {}, this.audit(model,plan)
                def export(this, model, plan, weights, activations, path, audit):
                    super().export(model,plan,weights,activations,path,audit)
                    seal(path,ids)
            plugin=SimpleNamespace(build_plan=lambda *args:deepcopy(self.plan),make_executor=Custom)
            spec={'service':'quantize_evaluate','parent_ref':'test-parent','frozen_sources':[binding],
                  'accept_trial':{'evidence_ref':'fixture-score','candidate_id':'changed'},'accepted_receipt':score,
                  'scored_calibration_ids':scored['sample_ids'],'calibration':selection,
                  'requires_code':True,'resolved_choices':{'training':False},
                  'evaluations':[{'kind':'classification','split':'search','count':1}]}
            package=root/'child';package.mkdir()
            # Only approval lookup and artifact confinement are fixtures. The exporter, load and forwards execute.
            with patch('tools.experiment_worker.module_from_approval',return_value=plugin), \
                 patch('tools.experiment_worker.asset',side_effect=lambda f:{'path':str(f),'sha256':file_sha256(f)}):
                got=evaluate_services({'payload':{'level':'L2','execution':spec}},self.b,self.cfg,self.data,package,None)
            lineage = got['observations']['refinement']
            self.assertEqual(lineage['child_state_sha256'],selected['child_state_sha256'])
            self.assertEqual(lineage['inherited_calibration_ids'],ids)
            self.assertEqual(lineage['incremental_calibration_ids'],scored['sample_ids'])
            self.assertEqual(lineage['job_calibration_reads'].get('calibration', []), [])
            self.assertTrue(got['observations']['reload']['host_verified'])
