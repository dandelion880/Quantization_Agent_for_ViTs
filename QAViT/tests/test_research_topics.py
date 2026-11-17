"""State/identity fixtures only. No LLM output or scientific measurements here."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, fingerprint, make_card
from core.workflow import Workflow
from tests.test_infrastructure import TempCase
from tests.support import topic_decision, hypothesis, meta, experiment_payload, validation_config
from tests import test_inspirations as ideas
from tests import test_memory_delivery as delivery


class TopicTests(TempCase):
    def focus(self,wf,**changes):
        self.assertFalse(wf.advance())
        tasks=[t for t in wf.queue.ready() if t['inputs'].get('purpose')=='focus']
        self.assertEqual(len(tasks),1)
        task=wf.queue.claim(tasks[0]['task_id'])
        wf.submit(task,{**topic_decision(wf),**changes},'SCRIPTED_TOPIC')
        return task,wf.topics.active()

    def revision(self,wf,parent,**changes):
        args={**topic_decision(wf),'affected_slots':['H1','H2','H3'],'impact':'future work only',**changes}
        tid=wf.queue.enqueue('topic-tool/'+str(wf.journal.sequence),'host_tool',inputs={
            'name':'revise_topic','arguments':args,'parent_task':parent['task_id'],
            'read_view':{'sequence':wf.journal.sequence,'memory_version':wf.journal.memory_version}})
        wf.queue.ready();task=wf.queue.claim(tid)
        result=wf.host_tool(task)
        return task,result

    def test_initial_shared_decision_before_candidates_and_restart_idempotency(self):
        wf=self.workflow(2); original=fingerprint(wf.brief['payload']); budget=wf.budget.snapshot()['config']
        wf.advance()
        self.assertEqual(list(wf.tasks()),['r001/focus'])
        focus=wf.tasks()['r001/focus']
        self.assertIsNone(wf.context(focus)['research_topic'])
        self.focus(wf)
        wf=Workflow(wf.run_dir); wf.advance()
        ref=wf.topics.round_ref(1)
        for slot in ('H1','H2','H3'):
            self.assertEqual(wf.tasks()['r001/generate/'+slot]['inputs']['context']['topic_ref'],ref)
        before=wf.journal.sequence
        wf.advance();self.assertEqual(wf.journal.sequence,before)
        self.assertEqual(fingerprint(wf.brief['payload']),original)
        self.assertEqual(wf.budget.snapshot()['config'],budget)

    def test_pause_switch_resume_retains_actions_ideas_and_exact_methods(self):
        wf=self.workflow(3); self.focus(wf);wf.advance()
        wf.queue.ready();parent=wf.queue.claim(wf.tasks()['r001/generate/H3']['task_id'])
        identifier=ideas.InspirationTests.idea(self,wf,parent,status='active',reason='develop')
        request,artifact,_=ideas.InspirationTests.code_request(self,wf,parent)
        wf.queue.ready();parent=wf.queue.claim(parent['task_id'])
        wf.submit(parent,hypothesis(1,'H3'),'fixture')
        candidate=wf.result_card(wf.queue.get(parent['task_id']))
        first=wf.topics.active();mid=candidate['payload']['method_id']
        self.assertEqual(wf.topics.methods()[mid]['inspiration_ids'],[identifier])
        # Use a separate Generation common decision as an explicit later-round boundary.
        tid=wf.enqueue_role('next/focus','focus',context={'round':2});wf.queue.ready();next_task=wf.queue.claim(tid)
        context=wf.context(next_task)
        self.assertEqual(context['research_topic']['actions'][0]['id'],first['actions'][0]['id'])
        self.assertEqual(context['research_topic']['implementation_history'][0]['candidate'],artifact)
        wf.submit(next_task,{
            **{k:v for k,v in topic_decision(wf).items() if k!='topic_id'},'decision':'switch',
            'question':'Explain the gain with a simpler mechanism','objective':'explanation and simplification',
            'method_ids':[mid],'inspiration_ids':[identifier]},'fixture')
        self.assertEqual(wf.inspirations.active()['id'],identifier)
        self.assertEqual(wf.topics.snapshot()[first['id']]['status'],'paused')
        second=wf.topics.active()
        tid=wf.enqueue_role('resume/focus','focus',context={'round':3});wf.queue.ready();resume=wf.queue.claim(tid)
        wf.submit(resume,{**topic_decision(wf),'decision':'resume','topic_id':first['id']},'fixture')
        wf=Workflow(wf.run_dir)
        recovered=wf.topics.recovery(first['id'])
        self.assertEqual(recovered['actions'][0]['id'],first['actions'][0]['id'])
        self.assertEqual(recovered['actions'][0]['description'],first['actions'][0]['description'])
        self.assertEqual(recovered['implementation_history'][0]['candidate'],artifact)
        self.assertEqual(wf.topics.snapshot()[second['id']]['status'],'paused')
        self.assertEqual(wf.topics.method_recovery(mid)['candidate_refs'],[candidate['id']])
        self.assertEqual(wf.topics.versions()[first['ref']]['status'],'active')
        # Same method may have a new candidate in a different topic without losing lineage.
        tid=wf.enqueue_role('cross-topic/generate','generate',context={'round':3,'slot':'H3',
            'hypothesis_type':'new_algorithm','topic_ref':recovered['ref']})
        wf.queue.ready(); task=wf.queue.claim(tid)
        wf.submit(task,{**hypothesis(3,'H3'),'method_id':mid},'fixture')
        self.assertEqual(len(wf.topics.methods()[mid]['candidate_refs']),2)
        self.assertEqual(wf.inspirations.active()['id'],identifier)
        self.revision(wf,task,decision='pause')
        tid=wf.enqueue_role('after-pause/focus','focus',context={'round':4})
        self.assertEqual(wf.context(wf.queue.get(tid))['research_topic']['id'],first['id'])

    update=ideas.InspirationTests.update

    def test_midround_scope_changes_keep_old_candidates_and_l3_identity(self):
        config=validation_config();config.update(purpose='infrastructure_test',enable_l3=True)
        wf=Workflow(self.root/'paused-l3',config);self.focus(wf);wf.advance()
        for slot in ('H1','H2','H3'):
            wf.queue.ready();task=wf.queue.claim(wf.tasks()['r001/generate/'+slot]['task_id'])
            wf.submit(task,hypothesis(1,slot),'fixture')
        wf.advance();wf.queue.ready();select=wf.queue.claim(wf.tasks()['r001/select']['task_id'])
        candidates=wf.candidate_refs(1);old=wf.topics.round_ref(1)
        tool,result=self.revision(wf,select,decision='narrow',mutable_dimensions=['metric'],affected_slots=['H3'])
        before=wf.journal.sequence
        self.assertEqual(wf.host_tool(tool),result);self.assertEqual(wf.journal.sequence,before)
        wf=Workflow(wf.run_dir)
        self.assertEqual(wf.topics.binding(wf.tasks()['r001/generate/H1'])['topic_ref'],old)
        self.assertEqual(wf.topics.binding(wf.tasks()['r001/generate/H3'])['topic_ref'],result['topic_ref'])
        for slot,ref in candidates.items():
            self.assertEqual(wf.journal.card(ref)['payload']['topic_ref'],old)
            tid=wf.enqueue_role('l3/'+slot,'plan_l3',context={'round':1,'slot':slot,'hypothesis_ref':ref})
            self.assertEqual(wf.research_request_binding(wf.queue.get(tid))['topic_ref'],old)
        self.revision(wf,select,decision='expand',mutable_dimensions=['metric','lookahead','grouping'])
        self.revision(wf,select,decision='pause')
        self.assertIsNone(wf.topics.active())
        self.assertEqual(wf.candidate_refs(1),candidates)
        wf.submit(select,{'rationale':'all submitted candidates still need L3'},'fixture');wf.advance()
        for slot,ref in candidates.items():
            key=wf.l3_request_keys(1,slot,ref)[0]
            plan=wf.tasks()[key]
            self.assertEqual(wf.research_request_binding(plan)['topic_ref'],old)

    def test_recorder_notes_are_advice_and_cannot_change_topic(self):
        wf=self.workflow(1);self.focus(wf)
        tid=wf.enqueue_recorder(1);wf.queue.ready();recorder=wf.queue.claim(tid)
        original=wf.topics.active()
        with self.assertRaises(ContractError):
            self.revision(wf,recorder,decision='pause')
        value=meta(1);value['topic_notes']=[{'topic_ref':original['ref'],'progress':'measurement missing',
            'reconsideration':'revisit after paired measurement; do not infer algorithm failure'}]
        wf.submit(recorder,value,'fixture')
        self.assertEqual(wf.topics.active(),original)
        self.assertEqual(wf.topics.recovery(original['ref'])['recorder_notes'][0]['note'],value['topic_notes'][0])

    def test_no_scope_gate_and_global_parent_catalog_and_B_survive(self):
        wf=self.workflow(1);self.focus(wf,mutable_dimensions=['metric']);wf.advance()
        task=wf.tasks()['r001/generate/H2']
        before=wf.solutions.catalog()
        context=wf.context(task)
        self.assertEqual(context['complete_solutions'],before)
        # A necessary global diagnostic outside the current focus remains admissible.
        spec=experiment_payload();spec['question']='global paired diagnostic outside current metric focus'
        self.assertTrue(wf.broker.preflight(spec))
        self.assertEqual(wf.solutions.fallback(),context['fallback_B'])
        self.assertEqual(wf.brief['payload']['hard_constraints']['training'],False)

    def test_legacy_round_keeps_unknown_topic_and_next_round_gets_decision(self):
        wf=self.workflow(2)
        for slot in ('H1','H2','H3'):
            wf.enqueue_role('r001/generate/'+slot,'generate',context={
                'round':1,'slot':slot,'hypothesis_type':hypothesis(1,slot)['hypothesis_type']})
        wf.advance();self.assertNotIn('r001/focus',wf.tasks());self.assertFalse(wf.topics.snapshot())
        wf.event('round_finished','legacy-finish',round=1)
        wf.advance();self.assertIn('r002/focus',wf.tasks());self.assertFalse(wf.topics.snapshot())

    def test_saved_call_cutoff_cannot_relabel_old_submission_or_see_future(self):
        wf=self.workflow(1);self.focus(wf);wf.advance()
        task=wf.tasks()['r001/generate/H3']
        task,context=delivery.DeliveryRegression.begin(self,wf,task)
        old=context['research_topic']['ref']
        tool,new=self.revision(wf,task,decision='expand')
        wf.submit(task,hypothesis(1,'H3'),'fixture')
        card=wf.result_card(wf.queue.get(task['task_id']))
        self.assertEqual(card['payload']['topic_ref'],old)
        with self.assertRaises(ContractError):wf.topics.recovery(new['topic_ref'],context['read_view']['sequence'])

    def test_legacy_unscoped_method_sources_are_readable_and_handed_to_coding(self):
        wf=self.workflow(2)
        parent=ideas.InspirationTests.author(self,wf,slot='H1')
        request,artifact,code=ideas.InspirationTests.code_request(self,wf,parent)
        parent=wf.queue.claim(parent['task_id'])
        wf.submit(parent,hypothesis(1,'H1'),'fixture')
        candidate=wf.result_card(wf.queue.get(parent['task_id']))
        mid=candidate['payload']['method_id']
        self.assertNotIn('topic_ref',candidate['payload'])
        self.assertFalse(wf.inspirations.snapshot())
        tid=wf.enqueue_role('legacy-source/focus','focus',context={'round':2})
        focus=wf.queue.get(tid)
        self.assertIsNone(wf.context(focus)['research_topic'])
        recovered=wf.topics.method_recovery(mid)
        self.assertEqual(recovered['implementation_history'][0]['candidate'],artifact)
        self.assertTrue(wf.can_read_card(focus,wf.journal.card(code['task_card'])))
        self.assertIn(artifact['path'],wf.authorized_assets(focus))
        self.assertIn(recovered['requests'][0]['host_source']['source_snapshot']['path'],wf.authorized_assets(focus))
        tid=wf.enqueue_role('method-next/generate','generate',context={
            'round':2,'slot':'H1','hypothesis_type':'existing_method'})
        future={'task_id':'future','inputs':{'parent_task':tid,'method_id':mid}}
        self.assertEqual(wf.implementation_handoff(future)[0]['candidate'],artifact)
        # A researcher may compare an explicitly different method before submitting.
        # Its ownership must not relabel that experiment as the new method.
        comparison=wf.queue.enqueue('method-comparison','experiment_request',inputs=future['inputs'])
        wf.queue.ready();next_task=wf.queue.claim(tid)
        wf.submit(next_task,hypothesis(2,'H1'),'fixture')
        new_mid=wf.result_card(wf.queue.get(tid))['payload']['method_id']
        ref=wf.queue.get(comparison)['task_card']
        self.assertIn(ref,[r['task_ref'] for r in wf.topics.method_recovery(mid)['requests']])
        self.assertNotIn(ref,[r['task_ref'] for r in wf.topics.method_recovery(new_mid)['requests']])

    def test_action_conditions_update_without_rewriting_history_and_focus_receipt_replays(self):
        wf=self.workflow(2)
        task,first=self.focus(wf,unresolved_questions=['missing paired measurement'])
        original=deepcopy(wf.result_card(wf.queue.get(task['task_id']))['payload'])
        before=wf.journal.sequence
        wf=Workflow(wf.run_dir);wf.submit(task,original,'fixture')
        self.assertEqual(wf.journal.sequence,before)
        tid=wf.enqueue_role('action/focus','focus',context={'round':2});wf.queue.ready();task=wf.queue.claim(tid)
        decision={**topic_decision(wf),'reconsider_when':'after the missing paired measurement is repaired',
                  'unresolved_questions':[]}
        wf.submit(task,decision,'fixture')
        latest=wf.topics.active()
        self.assertEqual(latest['actions'][0]['id'],first['actions'][0]['id'])
        self.assertEqual(latest['actions'][0]['reconsider_when'],decision['reconsider_when'])
        self.assertEqual(wf.topics.versions()[first['ref']],first)
        self.assertEqual(latest['unresolved_questions'],[])
        with self.assertRaises(ContractError):
            self.revision(wf,task,completed_action_ids=[first['actions'][0]['id']],
                          deferred_action_ids=[first['actions'][0]['id']])

    def test_precommit_interruption_reuses_exact_focus_and_revision_cards(self):
        wf=self.workflow(1);wf.advance();wf.queue.ready();task=wf.queue.claim(wf.tasks()['r001/focus']['task_id'])
        body=topic_decision(wf);commit=wf.journal.commit
        def crash(key,**kwargs):
            if any(c['kind']=='ReviewDecisionCard' and c['payload']['phase']=='research_direction'
                   for c in kwargs.get('cards',[])):
                def fault(stage):
                    if stage=='cards_written':raise OSError('fixture interruption before commit')
                kwargs['fault']=fault
            return commit(key,**kwargs)
        with patch.object(wf.journal,'commit',side_effect=crash),self.assertRaises(OSError):
            wf.submit(task,deepcopy(body),'fixture')
        self.assertFalse(wf.topics.snapshot())
        wf=Workflow(wf.run_dir);wf.submit(task,deepcopy(body),'fixture')
        self.assertEqual(len(wf.topics.versions()),1)
        wf.advance();parent=wf.tasks()['r001/generate/H3'];commit=wf.journal.commit
        with patch.object(wf.journal,'commit',side_effect=crash),self.assertRaises(OSError):
            self.revision(wf,parent,decision='narrow')
        tool=next(t for t in wf.tasks().values() if t['inputs'].get('name')=='revise_topic')
        wf=Workflow(wf.run_dir);result=wf.host_tool(tool)
        before=wf.journal.sequence
        self.assertEqual(wf.host_tool(tool),result)
        self.assertEqual(wf.journal.sequence,before)
        self.assertEqual(len(wf.topics.versions()),2)
        self.assertEqual(len(wf.journal.find_cards('ReviewDecisionCard')),2)

if __name__=='__main__':unittest.main()
