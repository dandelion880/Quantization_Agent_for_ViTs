"""Batch critique, version recovery and a real CPU tensor-probe through the existing worker."""
from copy import deepcopy
import time
from unittest.mock import patch

from agents.roles import role_spec, tools_for
from agents.working_materials import RESEARCH
from core.cards import ContractError, canonical, fingerprint
from core.journal import atomic_json, read_json
from core.workflow import Workflow
from runtime.configuration import config_check, model_defaults
from tests.support import advance_with_topic_fixture, experiment_payload, scheduler_l3_draft, validation_config
from tests.test_infrastructure import TempCase
from tests import test_gonly as gonly, test_candidate_revision as revision, test_memory_delivery as delivery


class BatchCritic(TempCase):
    drive = gonly.GOnlyRegression.drive
    begin = delivery.DeliveryRegression.begin
    revision = revision.CandidateRevision.revision

    def new(self, name='trial', enabled=True):
        cfg = validation_config()
        cfg.update(purpose='infrastructure_test', enable_critic=enabled, enable_l3=True)
        cfg['budget'].update(max_rounds=1, min_free_bytes=0)
        return Workflow(self.root / name, cfg)

    def freeze_call(self, wf, running):
        body = wf.request_body(running)
        path = wf.run_dir / 'jobs' / running['task_id'] / 'scripted-source.json'
        atomic_json(path, body)
        wf.queue.change(running['task_id'], 'running', patch={'request_path':str(path)})
        return wf.queue.get(running['task_id']), body['spec']['context']

    def test_disabled_path_and_old_fingerprint(self):
        wf, sel = self.drive(self.new(enabled=False), stop=lambda t:t['inputs']['purpose']=='select')
        self.assertNotIn('r001/critique', wf.tasks())
        body = wf.request_body(sel)['spec']
        self.assertNotIn('critic_feedback', body['context']['task'])
        self.assertNotIn('critic', wf.config['llm']['agent_models'])
        self.assertEqual(role_spec('generation','select',{},wf.config['llm'])['system_prompt'], body['system_prompt'])
        # Actual historical semantics: old manifest has neither flag nor route.
        saved = read_json(wf.run_dir/'research_run.json')
        original = saved['config']
        original.pop('enable_critic')
        self.assertEqual(fingerprint(config_check(deepcopy(original),historical=True)), fingerprint(original))
        saved['config_sha256'] = fingerprint(original)
        atomic_json(wf.run_dir/'research_run.json',saved)
        old = Workflow(wf.run_dir)
        self.assertNotIn('enable_critic',old.config)
        self.assertEqual(old.config['llm']['agent_models'],model_defaults())
        self.assertEqual(model_defaults(critic=True)['critic'],['gpt-6.2-astra','medium'])
        self.assertIn('critique',RESEARCH)
        self.assertNotIn('revise_candidate',{t['name'] for t in tools_for('critique')})

    def test_partial_candidates_and_unavailable_critic_fall_through(self):
        wf, failed = self.drive(self.new(), stop=lambda t:t['key']=='r001/generate/H2')
        wf.queue.fail(failed['task_id'],{'code':'fixture_generation_failure'})
        wf, critic = self.drive(wf,stop=lambda t:t['inputs']['purpose']=='critique')
        self.assertEqual(set(critic['inputs']['context']['candidate_refs']),{'H1','H3'})
        self.assertNotIn('r001/select',wf.tasks())
        wf.queue.fail(critic['task_id'],{'code':'RequestBudgetInsufficient','message':'local research allowance'})
        wf, sel = self.drive(Workflow(wf.run_dir),stop=lambda t:t['inputs']['purpose']=='select')
        c=sel['inputs']['context']
        self.assertEqual(c['incomplete_slots'],['H2'])
        self.assertEqual(c['critic_feedback']['status'],'failed')
        self.assertIn('RequestBudgetInsufficient',c['critic_feedback']['failure'])
        self.assertFalse(c['critic_feedback']['review_ref'])
        wf.submit(sel,{'rationale':'Keep both submitted scientific versions.'},None)
        wf.advance()
        plans=[t['inputs']['context']['slot'] for t in wf.tasks().values() if t['inputs'].get('purpose')=='plan_l3']
        self.assertEqual(set(plans),{'H1','H3'})
        self.assertIn(wf.queue.get(critic['task_id'])['task_card'],wf.memory.recording_batch()['scientific_failure_refs'])
        wf.retry_failed('recover outstanding planning tasks')
        self.assertEqual(wf.queue.get(critic['task_id'])['state'],'failed')

    def test_default_route_and_local_transport_failure_preserve_global_constraints(self):
        cfg = validation_config()
        cfg.pop('architecture')
        cfg.pop('enable_critic')
        wf = Workflow(self.root / 'default', cfg)
        self.assertTrue(wf.config['enable_critic'])
        self.assertEqual(wf.config['llm']['agent_models']['critic'], ['gpt-6.2-astra','medium'])
        wf, critic = self.drive(wf,stop=lambda t:t['inputs']['purpose']=='critique')
        folder=wf.run_dir/'jobs'/'route-check'
        atomic_json(folder/'llm_audit.json',{'model_call_started':False})
        receipt={'status':'failed','error':{'code':'llm_unsupported_model','message':'local route fixture'}}
        with patch.object(wf.runtime,'settle'):
            wf.accept(critic,receipt,folder/'result.json')
        self.assertIsNone(wf.transport_blocked)
        wf.advance()
        self.assertEqual(wf.tasks()['r001/select']['inputs']['context']['critic_feedback']['status'],'failed')
        sel,_ = self.begin(wf,wf.tasks()['r001/select'])
        receipt['error']={'code':'llm_quota_exhausted','message':'global quota fixture'}
        with patch.object(wf.runtime,'settle'):
            wf.accept(sel,receipt,folder/'result.json')
        self.assertEqual(wf.transport_blocked['code'],'llm_quota_exhausted')

    def test_development_and_execution_refreeze(self):
        from tools.experiments import ExperimentBroker
        from types import SimpleNamespace
        rows=[{'kind':'classification','split':'dev','count':4},{'kind':'ade_frozen','split':'ade_dev','count':2}]
        brief={'evaluation_protocol':{'full_dataset':True},'execution_guidance':{
               'development':{'manifest':{'sha256':'bound'},'evaluations':rows}}}
        broker=ExperimentBroker(SimpleNamespace(card=lambda *a,**kw:brief),'brief',None)
        draft=scheduler_l3_draft()
        with self.assertRaises(ContractError):
            broker.check_critic_stage(draft,'critic')
        draft['execution'].update(evaluations=rows)
        broker.check_critic_stage(draft,'critic')
        for value in ([], rows[:1], [{'kind':'classification','split':'final'}]):
            bad=deepcopy(draft);bad['execution'].update(evaluations=value)
            with self.subTest(value=value), self.assertRaises(ContractError):
                broker.check_critic_stage(bad,'critic')

        # Recovered requests must be checked again after Coding fills execution choices.
        unbound = {'author':'critic','id':'request','content_sha256':'request-hash','payload':{
            'level':'L3','execution':{'service':'evaluate_artifact','requires_code':True}}}
        with patch('tools.safeguards.verify_artifact'), self.assertRaisesRegex(ContractError,'Generation'):
            broker.bind_execution(unbound,{'execution_choices':{'evaluations':[{'kind':'classification','split':'final'}]}})

    def test_bound_numeric_feedback_and_generation_revision_without_rereview(self):
        wf, critic = self.drive(self.new(), stop=lambda t:t['inputs']['purpose']=='critique')
        critic, context = self.freeze_call(wf,critic)
        candidates=context['task']['candidate_refs']
        self.assertEqual(set(candidates),{'H1','H2','H3'})
        self.assertEqual(context['candidate_research_bindings']['H2']['method_id'],
                         wf.journal.card(candidates['H2'])['payload']['method_id'])
        def ask(arguments):
            wf.receive_tools(critic,{'tool_calls':[{'id':'probe','name':'request_experiment','arguments':arguments}]},'scripted')
        draft=experiment_payload()
        for args in ({'experiment':draft},{'hypothesis_ref':candidates['H2'],'experiment':{**draft,'method_id':'another-method'}},
                     {'hypothesis_ref':candidates['H3'],'experiment':scheduler_l3_draft()}):
            with self.subTest(args=list(args)), self.assertRaises(ContractError): ask(args)
        ask({'hypothesis_ref':candidates['H2'],'experiment':draft})
        wf.queue.ready()
        trial=wf.queue.claim(wf.queue.get(critic['task_id'])['waiting_for'][0])
        self.assertEqual(trial['inputs']['refs'],[wf.brief['id'],candidates['H2']])
        for field,value in context['candidate_research_bindings']['H2'].items():
            self.assertEqual(trial['inputs']['draft'][field],value)
        wf.progress_request(trial)
        trial=wf.queue.get(trial['task_id'])
        ex=wf.journal.card(trial['experiment_ref'])
        self.assertEqual(ex['author'],'critic')
        self.assertEqual(ex['payload']['hypothesis_ref'],candidates['H2'])
        wf.queue.ready(); job=wf.tasks()[trial['key']+'/execute']
        started=time.monotonic()
        wf.runtime.launch(job,wf.request_body(job),input_versions={ex['id']:ex['content_sha256']})
        while wf.queue.get(job['task_id'])['state']=='running' and time.monotonic()-started<90:
            wf.poll()
            time.sleep(.05)
        finished=wf.queue.get(job['task_id'])
        self.assertEqual(finished['state'],'completed',finished.get('error'))
        evidence=wf.journal.card(finished['outputs'][0])
        self.assertEqual(evidence['payload']['experiment_ref'],ex['id'])
        wf.broker.validate_evidence_for(ex,evidence)
        wrong_target=deepcopy(ex)
        wrong_target['payload']['hypothesis_ref']=candidates['H1']
        with self.assertRaisesRegex(ContractError,'确切身份'):
            wf.broker.validate_evidence_for(wrong_target,evidence)
        self.assertIsNone(wf.broker.cached(wrong_target))
        wf.queue.ready();wf.progress_request(wf.queue.claim(trial['task_id']))
        wf=Workflow(wf.run_dir)
        critic,ctx=self.begin(wf,wf.queue.get(critic['task_id']))
        self.assertIn(evidence['id'],wf.required_refs(critic))
        self.assertIn(evidence['id'],canonical(ctx['memory']['cards']))
        note={'hypothesis_ref':candidates['H2'],'suggestion':'Compare operator-output error on paired inputs.',
              'evidence_refs':[evidence['id']]}
        response={'rationale':'The local probe supports a paired output measurement.','candidate_notes':[note]}
        with self.assertRaises(ContractError):
            wf.submit(critic,{**response,'candidate_notes':[{**note,'hypothesis_ref':ex['id']}]},None)
        # Interrupt after validation/staging, before the atomic task/card commit.
        with patch.object(wf.queue,'finish',side_effect=OSError('injected power loss')), self.assertRaises(OSError):
            wf.submit(critic,response,None)
        wf=Workflow(wf.run_dir);wf.submit(critic,response,None);wf.submit(critic,response,None)
        review=wf.result_card(wf.queue.get(critic['task_id']))
        self.assertEqual(review['payload']['candidate_refs'],candidates)
        wf.advance();wf.advance()
        sel,ctx=self.begin(wf,wf.tasks()['r001/select'])
        self.assertIn(note['suggestion'],canonical(ctx['memory']['cards']))
        trial_feedback=ctx['task']['critic_feedback']['trials'][0]
        self.assertEqual(trial_feedback['evidence_refs'],[evidence['id']])
        self.assertEqual(trial_feedback['hypothesis_ref'],candidates['H2'])
        child, result=self.revision(wf,sel,candidates['H2'],[evidence['id']])
        self.assertEqual(result['validation_status'],'unvalidated_revision')
        wf.queue.finish(child['task_id'],result=result)
        advance_with_topic_fixture(wf)
        next_select=wf.tasks()[wf.selection_key(1)]
        self.assertEqual(next_select['inputs']['context']['critic_feedback']['candidate_refs']['H2'],candidates['H2'])
        self.assertEqual(next_select['inputs']['context']['candidate_refs']['H2'],result['candidate_ref'])
        next_select,_=self.begin(wf,next_select)
        wf.submit(next_select,{'rationale':'Keep all candidates and the explicit H2 revision.'},None)
        wf.advance()
        self.assertEqual(len([t for t in wf.tasks().values() if t['inputs'].get('purpose')=='plan_l3']),3)
        self.assertEqual(len([t for t in wf.tasks().values() if t['inputs'].get('purpose')=='critique']),1)
        record=wf.queue.get(wf.enqueue_recorder(1,incomplete=True,key='early-record'))
        _,ctx=self.begin(wf,record)
        self.assertIn(note['suggestion'],canonical(ctx['memory']['cards']))
        self.assertIn(evidence['id'],wf.required_refs(record))
        self.receipt={'cpu_worker':finished['result'],'experiment_ref':ex['id'],'evidence':evidence,
                      'review_ref':review['id'],'recommendation':note,'revision':result,'select':next_select['task_id']}
