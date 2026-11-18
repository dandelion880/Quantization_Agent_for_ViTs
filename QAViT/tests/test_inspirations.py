"""CPU-only persistence/handoff regressions; scripted results are not science."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, make_card
from core.journal import file_sha256
from core.workflow import Workflow
from tests.support import advance_with_topic_fixture
from tests.test_infrastructure import TempCase
from tests.support import hypothesis, experiment_payload, meta, validation_config
from tests import test_multi_candidate_l3 as multi
from tests import test_fallback_solutions as fallback


class InspirationTests(TempCase):
    def setUp(self):
        super().setUp()
        block = patch('runtime.task_runtime.TaskRuntime.launch', side_effect=AssertionError('offline only'))
        block.start(); self.addCleanup(block.stop)

    def author(self, wf, slot='H3', number=1):
        tid = wf.enqueue_role('author/' + str(wf.journal.sequence), 'generate', refs=[wf.brief['id']],
            context={'round': number, 'slot': slot, 'hypothesis_type': {'H1':'existing_method','H2':'engineering','H3':'new_algorithm'}[slot]})
        wf.queue.ready()
        return wf.queue.claim(tid)

    def update(self, wf, parent, args, *, execute=True):
        tid = wf.queue.enqueue('update/' + str(wf.journal.sequence), 'host_tool', inputs={
            'name': 'update_inspiration', 'arguments': args, 'parent_task': parent['task_id'],
            'read_view': {'sequence': wf.journal.sequence, 'memory_version': wf.journal.memory_version}})
        wf.queue.ready(); task = wf.queue.claim(tid)
        if not execute:
            return task
        result = wf.host_tool(task)
        wf.queue.finish(tid, result=result)
        return result

    def idea(self, wf, parent, **kwargs):
        return self.update(wf, parent, {'title':'Persistent mechanism', 'idea':'bounded residual correction',
            'basis':'theoretical fixture, not empirical evidence', 'next_step':'repair clipping at the original site', **kwargs})['inspiration_id']

    def code_request(self, wf, parent):
        payload = experiment_payload()
        payload['execution'].update(requires_code=True, interface='quantizer_v1')
        wf.receive_tools(parent, {'tool_calls':[{'id':'request-' + str(wf.journal.sequence),
            'name':'request_experiment','arguments':{'experiment':payload, 'research_note':'keep original implementation; repair clipping'}}]}, 'scripted')
        tid = wf.queue.get(parent['task_id'])['waiting_for'][0]
        wf.queue.ready(); trial = wf.queue.claim(tid)
        wf.progress_request(trial)
        code = wf.tasks()[trial['key'] + '/code']
        wf.queue.ready(); code = wf.queue.claim(code['task_id'])
        wf.submit(code, {'source':'# EXACT_H3_V1\ndef fit(x, parameters):\n    return parameters\ndef apply(x, state, parameters):\n    return x\n'}, 'scripted')
        artifact = wf.queue.get(code['task_id'])['result']['artifact']
        # Keep an incomplete implementation without fabricating a positive result.
        wf.queue.ready(); wf.queue.claim(tid)
        wf.queue.finish(tid, result={'status':'incomplete', 'reason':'scripted boundary, no worker launched'})
        wf.queue.ready()
        return wf.queue.get(tid), artifact, wf.queue.get(code['task_id'])

    def test_idea_only_pause_switch_resume_and_replay(self):
        wf = self.workflow(2); author = self.author(wf)
        before = wf.solutions.snapshot(); budget = wf.budget.snapshot()['reservations']
        first = self.idea(wf, author)
        self.assertEqual(wf.inspirations.snapshot()[first]['status'], 'paused')
        self.assertIsNone(wf.inspirations.active())
        self.assertFalse(any(t['kind'] == 'experiment_request' for t in wf.tasks().values()))
        self.assertEqual(wf.budget.snapshot()['reservations'], budget)
        self.assertEqual(wf.solutions.snapshot(), before)
        self.update(wf, author, {'inspiration_id':first,'status':'active','reason':'specific next step is feasible'})
        second = self.idea(wf, author, status='active', reason='alternative can resolve an observed limitation')
        self.assertEqual(wf.inspirations.snapshot()[first]['status'], 'paused')
        tool = self.update(wf, author, {'inspiration_id':first,'status':'active','reason':'resume preserved implementation'}, execute=False)
        result = wf.host_tool(tool)  # Crash after Journal commit, before queue.finish.
        sequence = wf.journal.sequence
        wf = Workflow(wf.run_dir)
        self.assertEqual(wf.host_tool(tool), result)
        self.assertEqual(wf.journal.sequence, sequence)
        self.assertEqual(wf.inspirations.active()['id'], first)
        self.assertEqual(wf.inspirations.snapshot()[second]['status'], 'paused')
        self.assertEqual(len(wf.inspirations.snapshot()), 2)

    def test_low_unranked_h3_restores_exact_code_config_and_results_next_round(self):
        cfg = validation_config(); cfg.update(purpose='infrastructure_test', enable_l3=True)
        cfg['budget'].update(max_rounds=2, min_free_bytes=0)
        wf = Workflow(self.root/'continuation', cfg)
        # B and three stronger formal-shaped outcomes; validator is intentionally
        # isolated here. Nothing below is a real measurement.
        for i in range(1,4):
            fallback.FallbackTests.promote(self,wf,i,80,40+i)
        before_B = wf.solutions.fallback(); before_top = wf.solutions.snapshot()
        advance_with_topic_fixture(wf)
        for slot in ('H1','H2','H3'):
            wf.queue.ready(); task = wf.queue.claim(wf.tasks()['r001/generate/'+slot]['task_id'])
            if slot == 'H3':
                identifier = self.idea(wf, task, status='active', reason='develop the same mechanism')
                request, artifact, code = self.code_request(wf, task)
                wf.queue.ready(); task = wf.queue.claim(task['task_id'])
            body = hypothesis(1,slot)
            if slot in ('H1','H2'):
                body['solution_derivation'] = {'base_solution_id': wf.solutions.seeds()[0 if slot == 'H1' else 1]['id'],
                    'changes':'nontrivial calibration objective change','rollback':'reload immutable parent configuration'}
            wf.submit(task, body, 'scripted')
        advance_with_topic_fixture(wf); wf.queue.ready()
        select = wf.queue.claim(wf.tasks()['r001/select']['task_id'])
        wf.submit(select, {'rationale':'all three enter L3'}, 'scripted'); advance_with_topic_fixture(wf)
        for task in wf.queue.ready():
            task = wf.queue.claim(task['task_id'])
            draft = experiment_payload('L3')
            draft['execution'].update(service='evaluate_artifact', parameters={}, evaluations=[{'kind':'classification','split':'final','count':2}])
            wf.submit(task, draft, 'scripted')
        advance_with_topic_fixture(wf)
        trials = {t['inputs']['slot']:t for t in wf.queue.ready() if t['kind']=='experiment_request'}
        self.assertEqual(set(trials), {'H1','H2','H3'})
        for slot, trial in trials.items():
            ev = multi.MultiCandidateL3.measured(self,wf,trial)
            if slot == 'H3':
                h3_ev = ev
                exp = wf.journal.card(ev['payload']['experiment_ref'])
                # The independent ranking path must reject a low but valid-shaped
                # H3 from Top-3 without touching its active idea.
                shaped = deepcopy(ev)
                shaped['id'] += '-formal-shape'
                shaped['payload'].update(measurement_source='real', provenance={'model_ref':'low-H3'},
                    observations={'evaluations':{'classification':{'full_dataset':True,'metrics':{'top1':1}},
                        'segmentation':{'full_dataset':True,'metrics':{'mIoU':1}}}})
                wf.solutions.consider_round(1, shaped, exp, wf.selection_card(1), validator=lambda *a,**k:None)
        advance_with_topic_fixture(wf); wf.queue.ready()
        recorder = wf.queue.claim(wf.tasks()['r001/record']['task_id'])
        self.update(wf, recorder, {'inspiration_id':identifier, 'progress':'low result; clipping remains unresolved',
            'next_step':'repair clipping in EXACT_H3_V1', 'source_refs':[h3_ev['id']]})
        wf.submit(recorder, meta(1), 'scripted'); advance_with_topic_fixture(wf)
        self.assertEqual(wf.solutions.snapshot(), before_top)
        self.assertEqual(wf.solutions.fallback(), before_B)
        wf = Workflow(wf.run_dir); advance_with_topic_fixture(wf)
        next_task = wf.tasks()['r002/generate/H3']
        context = wf.context(next_task)
        resume = context['inspiration_continuation']
        self.assertEqual(resume['id'],identifier)
        self.assertEqual(resume['next_step'],'repair clipping in EXACT_H3_V1')
        self.assertIn(h3_ev['id'], [r for row in resume['requests'] for r in row['evidence_refs']])
        self.assertEqual(resume['implementation_history'][0]['candidate'],artifact)
        self.assertEqual(file_sha256(artifact['path']), artifact['source_sha256'])
        self.assertIn(request['request_ref'],[r['request_ref'] for r in resume['requests']])
        self.assertIn(artifact['path'],wf.authorized_assets(next_task))
        wf.queue.ready(); next_task = wf.queue.claim(next_task['task_id'])
        next_request, _, next_code = self.code_request(wf, next_task)
        handed = next_code['inputs']['context']['implementation_history']
        self.assertEqual(handed[0]['candidate'],artifact)
        self.assertIn('EXACT_H3_V1',wf.context(next_code)['prior_implementation_0'])
        self.assertIn(artifact['id'],next_code['result']['parent_code_ids'])
        self.assertEqual(next_request['inputs']['inspiration_id'],identifier)
        self.assertEqual(len(wf.inspirations.snapshot()),1)
        h1 = wf.context(wf.tasks()['r002/generate/H1'])['previous_derivation']
        self.assertEqual(h1['solution_derivation']['base_solution_id'],wf.solutions.seeds()[0]['id'])
        h2=wf.context(wf.tasks()['r002/generate/H2'])['previous_derivation']
        self.assertEqual(h2['solution_derivation']['base_solution_id'],wf.solutions.seeds()[1]['id'])

    def test_repair_versions_keep_their_own_execution_and_result_bindings(self):
        wf=self.workflow(1); parent=self.author(wf)
        identifier=self.idea(wf,parent,status='active',reason='develop')
        request, artifact, code=self.code_request(wf,parent)
        original=wf.journal.card(request['request_ref'])
        executed=[]
        for index in range(2):
            if index:
                tid=wf.enqueue_role('repair/code','code',refs=[original['id']],
                    context={'experiment_ref':original['id']}, parent_task=request['task_id'])
                wf.queue.ready(); repair=wf.queue.claim(tid)
                wf.submit(repair,{'source':'# EXACT_H3_V2\ndef fit(x, parameters):\n    return parameters\ndef apply(x, state, parameters):\n    return x\n'},'scripted')
                artifact=wf.queue.get(tid)['result']['artifact']
            spec=wf.broker.bind_execution(original,artifact)
            evidence=make_card('EvidenceCard',wf.journal.run_id,{
                'experiment_ref':spec['id'],'execution_fingerprint':'fixture-'+str(index),
                'status':'failed' if index == 0 else 'completed', 'observations':{'score':index},
                'provenance':{},'resources':{},'artifacts':[], 'measurement_source':'test_double',
                'execution_id':'repair-fixture-'+str(index)},refs=[spec['id']])
            wf.journal.commit('repair/spec/'+str(index),cards=[spec,evidence])
            tid=wf.queue.enqueue('repair/execute/'+str(index),'experiment',inputs={
                'parent_task':request['task_id'], 'experiment_ref':spec['id'], 'approval':{'candidate':artifact}})
            wf.queue.ready(); wf.queue.claim(tid)
            wf.queue.finish(tid,outputs=[evidence['id']])
            executed.append((artifact,spec,evidence))
        recovery=wf.inspirations.recovery(identifier)
        for row, (artifact,spec,evidence) in zip(recovery['implementation_history'],executed):
            self.assertEqual(row['candidate'],artifact)
            self.assertEqual(row['experiment_ref'],spec['id'])
            self.assertEqual(row['execution_refs'],[spec['id']])
            self.assertEqual(row['evidence_refs'],[evidence['id']])
        self.assertEqual(len(recovery['implementation_history']),2)
        self.assertEqual(recovery['requests'][0]['evidence_refs'],[r[2]['id'] for r in executed])

    def test_recorder_absorption_updates_without_scientific_state_change(self):
        wf=self.workflow(1); parent=self.author(wf)
        identifier=self.idea(wf,parent,status='active',reason='develop')
        target=make_card('HypothesisCard',wf.journal.run_id,hypothesis(1,'H1'))
        wf.journal.commit('target',cards=[target])
        tid=wf.enqueue_recorder(1); wf.queue.ready()
        # Release the subscription resource held by the fixture author.
        wf.queue.finish(parent['task_id'])
        wf.queue.ready(); recorder=wf.queue.claim(tid)
        update={'inspiration_id':identifier,'progress':'usable part transferred',
                'absorbed_parts':[{'part':'bounded transform only','target_ref':target['id']}]}
        self.update(wf,recorder,update)
        self.assertEqual(wf.inspirations.active()['absorbed_parts'],update['absorbed_parts'])
        with self.assertRaises(ContractError):
            self.update(wf,recorder,{'inspiration_id':identifier,'status':'paused','reason':'low score'})
        self.assertEqual(wf.inspirations.active()['id'],identifier)

    def test_paused_code_is_retained_and_hash_tampering_is_rejected(self):
        wf=self.workflow(2); parent=self.author(wf)
        identifier=self.idea(wf,parent,status='active',reason='develop')
        request, artifact, _=self.code_request(wf,parent)
        self.update(wf,parent,{'inspiration_id':identifier,'status':'paused','reason':'implementation needs a smaller repair'})
        wf=Workflow(wf.run_dir)
        recovery=wf.inspirations.recovery(identifier)
        self.assertEqual(recovery['status'],'paused')
        self.assertEqual(recovery['implementation_history'][0]['candidate'],artifact)
        self.assertEqual(recovery['requests'][0]['host_source'],wf.solutions.source_record())
        self.update(wf,parent,{'inspiration_id':identifier,'status':'active','reason':'resume the concrete clipping repair'})
        next_task=self.author(wf,number=2)
        body=wf.request_body(next_task)
        permission=body['read_permissions'][artifact['path']]
        self.assertEqual(permission,artifact['source_sha256'])
        Path(artifact['path']).write_text('# corrupted fixture source')
        with self.assertRaises(ContractError):
            wf.host_tool({'inputs':{'name':'read_asset','arguments':{'asset':artifact['path']},
                'parent_task':next_task['task_id'], 'read_view':{
                    **body['spec']['context']['read_view'], 'assets':body['read_permissions']}}})

    def test_candidate_identity_stays_bound_after_generation_switches(self):
        wf=self.workflow(1); parent=self.author(wf)
        first=self.idea(wf,parent,status='active',reason='develop')
        bad=hypothesis(1,'H3'); bad['inspiration_id']='invented'
        with self.assertRaises(ContractError): wf.submit(parent,bad,'scripted')
        wf.submit(parent,hypothesis(1,'H3'),'scripted')
        candidate=wf.result_card(wf.queue.get(parent['task_id']))
        for slot in ('H1', 'H2'):
            parent_branch = self.author(wf, slot)
            with self.assertRaises(ContractError):
                self.idea(wf, parent_branch, status='active', reason='ordinary parent improvement')
            self.assertEqual(wf.inspiration_request_binding(parent_branch, {'inspiration_id': first}),
                             {'inspiration_refs': [first]})
            self.assertNotIn('inspiration_continuation', wf.context(parent_branch))
            wf.queue.finish(parent_branch['task_id'], result={})
        chooser=self.author(wf,'H3')
        second=self.idea(wf,chooser,status='active',reason='next round should explore a different concrete mechanism')
        tid=wf.enqueue_role('plan/exact','plan_l3',context={'round':1,'slot':'H3','hypothesis_ref':candidate['id']})
        plan=wf.queue.get(tid)
        self.assertEqual(wf.inspiration_request_binding(plan),{'inspiration_id':first})
        self.assertEqual(wf.context(plan)['inspiration_continuation']['id'],first)
        with self.assertRaises(ContractError): wf.inspiration_request_binding(plan,{'inspiration_id':second})
        self.assertEqual(wf.inspirations.active()['id'],second)

    def test_snapshot_hides_future_idea_and_rejects_stale_update(self):
        wf=self.workflow(1); parent=self.author(wf)
        identifier=self.idea(wf,parent,status='active',reason='develop')
        frozen=wf.journal.sequence
        stale=self.update(wf,parent,{'inspiration_id':identifier,'progress':'old'},execute=False)
        self.update(wf,parent,{'inspiration_id':identifier,'progress':'new'})
        with self.assertRaises(ContractError): wf.host_tool(stale)
        self.assertEqual(wf.inspirations.recovery(identifier,frozen)['progress'],'尚未实施')
        self.assertEqual(wf.inspirations.active()['progress'],'new')

if __name__=='__main__': unittest.main()
