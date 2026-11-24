"""Recovery/protocol regression fixtures; no LLM or performance claims."""
from copy import deepcopy
from pathlib import Path
import hashlib
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, canonical
from core.workflow import Workflow
from core.experiment_requests import ExperimentRequests
from core.journal import read_json, atomic_json, Journal
from runtime.gpu_resources import ResourceBusy, ResourceUnavailable
from runtime.budget import BudgetLedger
from tests.test_infrastructure import TempCase
from tests.support import topic_decision, meta
from tests import test_research_topics as topics
from tests import test_gonly as gonly
from tools.evaluation.rules import Rules
from tools.evaluation.common import QAVIT


class FrozenEvaluation(unittest.TestCase):
    def test_package_validation_is_independent_of_current_task_catalog(self):
        with patch('tools.evaluation.rules.task_catalog', return_value={'future':{'role':'generation'}}):
            for version in ('v0_1','gonly_v0_2','gonly_v0_3'):
                rules=Rules(QAVIT/'configs/evaluation'/version)
                self.assertEqual(rules.coverage(),{'future':'not_covered'})
                self.assertTrue(rules.accepts_role('generate','generation'))
                self.assertFalse(rules.accepts_role('generate','coding'))
        current=Rules(QAVIT/'configs/evaluation/gonly_v0_3')
        self.assertIn('focus',current.selected)
        self.assertIn('跨轮',canonical(current.selected['generate']))
        self.assertNotIn('focus',Rules(QAVIT/'configs/evaluation/gonly_v0_2').selected)

    def test_structured_error_origin_beats_names_and_wording(self):
        for message in ('CUDA error: device-side assert triggered','out of memory in quoted code', '批准 path mismatch'):
            self.assertTrue(ExperimentRequests.revisable_failure({'error':{
                'code':'ContractError','message':message,'origin':'implementation'}}))
            for origin in ('host','resource','request'):
                self.assertFalse(ExperimentRequests.revisable_failure({'error':{
                    'code':'TypeError','message':message,'origin':origin}}))
        self.assertFalse(ExperimentRequests.revisable_failure({'error':{'code':'RuntimeError','message':'CUDA error'}}))

    def test_wrapped_trusted_failure_preserves_origin_and_child_receipt(self):
        from runtime.failures import RemoteWorkerError, failure_details
        try:
            try:
                raise RemoteWorkerError({'origin':'implementation','code':'ValueError','message':'bad tensor shape'},
                    worker='classification',receipt='original.json',worker_log='worker.log')
            except RemoteWorkerError as exc:
                raise RuntimeError('CUDA error / out of memory quoted by wrapper') from exc
        except RuntimeError as exc:
            error=failure_details(exc,{})
        self.assertEqual(error['origin'],'implementation')
        self.assertEqual(error['worker_cause']['code'],'ValueError')
        self.assertEqual(error['subprocess']['receipt'],'original.json')


class TopicRecovery(TempCase):
    focus=topics.TopicTests.focus

    def test_shared_recovery_note_keeps_evidence_configuration_and_code_handles(self):
        request={'task_ref':'task@1','request_ref':'request@1','experiment_ref':'experiment@1',
                 'evidence_refs':['measurement@1'],'configuration':{'ref':'experiment@1','field':'/payload'},
                 'research_note':'An exact retained scientific note.'}
        code={'candidate':{'id':'implementation','path':'original.py','source_sha256':'sha'},
              'evidence_refs':['measurement@1'],'execution_task_refs':['execution@1']}
        recovery={'requests':[request],'implementation_history':[code]}
        context={'research_topic':deepcopy(recovery),'inspiration_continuation':deepcopy(recovery)}
        Workflow.coalesce_recovery_bodies(context)
        actual=context['inspiration_continuation']
        self.assertEqual(actual['implementation_history'],[code])
        self.assertEqual({k:v for k,v in actual['requests'][0].items() if k!='research_note'},
                         {k:v for k,v in request.items() if k!='research_note'})
        self.assertEqual(context['research_topic']['requests'][0]['research_note'],request['research_note'])
        self.assertEqual(actual['requests'][0]['research_note']['body_location'],
                         'context.research_topic.requests[0].research_note')

    def decide(self,wf,**changes):
        tid=wf.enqueue_role('decision/'+str(wf.journal.sequence),'focus',context={'round':2})
        wf.queue.ready();task=wf.queue.claim(tid)
        wf.submit(task,{**topic_decision(wf),**changes},'state fixture')
        return wf.topics.snapshot()

    def test_defer_pause_resume_complete_do_not_recreate_text_matched_actions(self):
        wf=self.workflow(2);_,first=self.focus(wf)
        aid=first['actions'][0]['id']
        self.decide(wf,decision='pause',deferred_action_ids=[aid])
        paused=wf.topics.snapshot()[first['id']]
        self.assertEqual(len(paused['actions']),1)
        self.assertEqual(paused['actions'][0]['status'],'deferred')
        wf=Workflow(wf.run_dir)
        self.decide(wf,decision='resume',topic_id=first['id'],resumed_action_ids=[aid],next_action_id=aid,
                    next_step='reworded next step, same explicitly identified action',reconsider_when='after paired counterevidence')
        active=wf.topics.active()
        self.assertEqual(len(active['actions']),1)
        self.assertEqual(active['actions'][0]['id'],aid)
        self.assertEqual(active['actions'][0]['description'],first['actions'][0]['description'])
        self.assertEqual(active['actions'][0]['reconsider_when'],'after paired counterevidence')
        self.decide(wf,completed_action_ids=[aid],next_step=first['next_step'])
        self.assertEqual([a['status'] for a in wf.topics.active()['actions']],['completed'])
        self.decide(wf,new_action=first['next_step'])
        rows=wf.topics.active()['actions']
        self.assertEqual(len(rows),2);self.assertNotEqual(rows[0]['id'],rows[1]['id'])
        self.assertEqual(wf.topics.versions()[first['ref']],first)

    def test_invalid_topic_note_is_retained_locally_with_valid_recorder_body(self):
        wf=self.workflow(1);_,topic=self.focus(wf)
        tid=wf.enqueue_recorder(1);wf.queue.ready();task=wf.queue.claim(tid)
        bad=[{'topic_ref':'topic-missing@1','progress':'keep original'},
             {'topic_ref':topic['id'],'progress':'exact version required'},
             {'topic_ref':topic['ref'],'progress':'invalid source','source_refs':['Card-nonexistent@1']},
             {'topic_ref':topic['ref'],'progress':17}]
        good={'topic_ref':topic['ref'],'progress':'valid conditional advice','source_refs':[wf.brief['id']]}
        body={**meta(1),'topic_notes':[bad[0],good,*bad[1:]]}
        wf.submit(task,body,'fixture')
        card=wf.result_card(wf.queue.get(tid))
        self.assertEqual(card['payload']['topic_notes'],[good])
        self.assertEqual([x['submission'] for x in card['payload']['submission_issues']],bad)
        self.assertEqual(card['payload']['scientific_analysis'],body['scientific_analysis'])
        self.assertEqual(wf.topics.active(),topic)
        self.assertEqual(len(wf.topics.recovery(topic['ref'])['recorder_notes']),1)

    def test_exact_source_member_and_archive_isolation_survive_saved_legacy_permissions(self):
        wf=self.workflow(1);wf.advance();task=wf.tasks()['r001/focus']
        source=wf.solutions.available()[0]
        archive=source['source_snapshot'];path=Path(archive['path'])
        self.assertNotIn(str(path),wf.authorized_assets(task))
        task['_read_view']={'sequence':wf.journal.sequence,'memory_version':wf.journal.memory_version,
                            'assets':{str(path):archive['sha256']}}
        self.assertNotIn(str(path),wf.authorized_assets(task))
        args={'snapshot_field':archive['field'],'member':'tools/ptq_agent/execution/calibration.py'}
        result=wf.read_snapshot_asset(task,path,args)
        raw=read_json(path)['source_snapshot'][args['member']]
        self.assertEqual(result['text'],raw[:wf.PAGE_CHARACTERS])
        self.assertEqual(result['sha256'],hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(result['source_snapshot'],archive)
        for member in ('configs/evaluation/v0_1/rubrics/generation.json','tools/evaluation/rules.py',
                       'agents/prompts/judger.md','../research_run.json'):
            with self.assertRaises(ContractError):wf.read_snapshot_asset(task,path,{**args,'member':member})
        listing=wf.read_snapshot_asset(task,path,{'snapshot_field':archive['field'],'limit':12000})
        self.assertNotIn('rubrics/',listing['text'])
        modified=read_json(path);modified['source_snapshot'][args['member']]+='\n# corrupted fixture\n'
        atomic_json(path,modified)
        with self.assertRaises(ContractError):wf.read_snapshot_asset(task,path,args)

    def test_catalog_pages_keep_tail_and_do_not_inject_idea_bodies(self):
        wf=self.workflow(1);wf.advance();task=wf.tasks()['r001/focus']
        events=[{'type':'inspiration_updated','entry':{'id':f'idea-{i}','revision':1,'status':'paused',
                 'title':f'retain-{i}','idea':'BODY_SENTINEL'*1000,'next_step':'retained action',
                 'source_refs':[],'absorbed_parts':[]}} for i in range(30)]
        wf.journal.commit('long-notes',events=events)
        context=wf.context(task)
        self.assertLessEqual(len(context['inspiration_library']),8)
        self.assertNotIn('BODY_SENTINEL',canonical(context))
        self.assertEqual(context['inspiration_library_directory']['total'],30)
        rows=wf.research_directory('inspiration',wf.journal.sequence,offset=24,limit=8)['entries']
        self.assertIn('idea-0',[r['id'] for r in rows])
        self.assertIn('BODY_SENTINEL',wf.inspirations.recovery('idea-0')['idea'])


class SynthesisRecovery(TempCase):
    workflow_case=gonly.GOnlyRegression.workflow_case
    drive=gonly.GOnlyRegression.drive
    def test_failed_selection_keeps_candidates_and_library_without_selecting_h1(self):
        # Replace the former test's assumed round closure with explicit recovery.
        wf,select=self.drive(self.workflow_case(),stop=lambda t:t['inputs']['purpose']=='select')
        candidates=wf.candidate_refs(1);library=wf.solutions.snapshot()
        wf.queue.fail(select['task_id'],{'code':'fixture_selection_failure'})
        wf=Workflow(wf.run_dir)
        self.assertFalse(wf.advance())
        self.assertFalse(wf.summary()['completed_rounds'])
        self.assertIsNone(wf.selection_card(1))
        pending=next(e for e in wf.journal.events if e['type']=='synthesis_pending')
        self.assertEqual(pending['candidate_refs'],candidates)
        self.assertEqual(wf.solutions.snapshot(),library)
        before=wf.journal.sequence;wf.advance();self.assertEqual(wf.journal.sequence,before)
        wf.retry_failed('fixture transport restored, same budget')
        wf.queue.ready();task=wf.queue.claim(select['task_id'])
        wf.submit(task,{'rationale':'fixture scientific review of all exact versions'},'fixture')
        wf.advance()
        for slot,ref in candidates.items():
            self.assertEqual(wf.tasks()[wf.l3_request_keys(1,slot,ref)[0]]['inputs']['context']['hypothesis_ref'],ref)


class ResourceAndCostRecovery(TempCase):
    def test_gpu_launch_without_authorization_does_not_claim_or_spend(self):
        wf=self.workflow(1)
        tid=wf.queue.enqueue('needs-gpu','experiment');wf.queue.ready()
        with self.assertRaises(ResourceUnavailable):
            wf.runtime.launch(wf.queue.get(tid),{'requires_gpu':True,'gpu_pool':[]},input_versions=[])
        self.assertEqual(wf.queue.get(tid)['state'],'ready')
        self.assertEqual(wf.queue.get(tid)['attempt'],0)
        self.assertFalse(wf.budget.snapshot()['reservations'])

    def test_model_training_input_normalizes_without_exposing_internal_choices(self):
        from agents.roles import EXPERIMENT_SUBMISSION_SCHEMA
        from core.cards import validate_schema
        from tests.support import experiment_payload
        wf=self.workflow(1);draft=experiment_payload()
        draft['execution'].pop('resolved_choices')
        draft['execution']['training']=False
        for field in ('model_scope','backend','model_identity_sha256','measurement_implementation'):
            draft['execution'].pop(field,None)
        for field in ('stage','request_ref','request_sha256','original_request','scope','validation'):
            draft.pop(field,None)
        draft['resources'].pop('cpu_threads',None)
        validate_schema(draft,EXPERIMENT_SUBMISSION_SCHEMA,'model draft')
        normalized=wf.broker.preflight(draft)
        self.assertFalse(normalized['execution']['resolved_choices']['training'])
        self.assertNotIn('resolved_choices',draft['execution'])
        conflict=deepcopy(draft);conflict['execution']['resolved_choices']={'training':True}
        with self.assertRaises(ContractError):wf.broker.preflight(conflict)

    def test_worker_accepts_offline_archive_changes_but_rejects_execution_changes(self):
        from contextlib import redirect_stderr
        from io import StringIO
        import time
        from runtime import task_worker
        from runtime.subscription import StructuredOutputError
        for executable_change in (False, True):
            for malformed in (False, True):
                with self.subTest(executable_change=executable_change, malformed=malformed):
                    path=self.root/f'worker-{executable_change}-{malformed}'/'request.json'
                    atomic_json(path,{'kind':'llm','task_id':'fixture','execution_id':'fixture',
                        'deadline_unix':time.time()+60,'core_sha256':'original-runtime'})
                    before={'core_sha256':'original-runtime','archive_sha256':'original-archive'}
                    after={'core_sha256':'new-runtime' if executable_change else 'original-runtime',
                           'archive_sha256':'new-offline-archive'}
                    error=StructuredOutputError('bad','fixture',ValueError('bad'),{'input_tokens':1},
                                                stage='fixture',path='$') if malformed else None
                    with patch.object(sys,'argv',['task_worker.py',str(path)]), \
                         patch.dict('os.environ'),patch.object(task_worker.signal,'signal'), \
                         patch.object(task_worker.signal,'setitimer'), \
                         patch.object(task_worker,'code_identity',side_effect=[before,after]), \
                         patch.object(task_worker,'work',return_value={'fixture':True},side_effect=error), \
                         redirect_stderr(StringIO()):
                        task_worker.main()
                    result=read_json(path.with_name('result.json'))
                    if executable_change:
                        self.assertEqual(result['status'],'failed')
                        self.assertEqual(result['error']['code'],'ContractError')
                    elif malformed:
                        self.assertEqual(result['error']['code'],'llm_structured_output_error')
                        self.assertEqual(result['data']['usage'],{'input_tokens':1})
                    else:
                        self.assertEqual(result['status'],'completed')
                        self.assertEqual(result['data'],{'fixture':True})

    def test_memory_navigation_preserves_results_and_validates_full_originals(self):
        from tests.test_serial_debate_memory import SerialDebateMemoryRegression
        wf=self.workflow(1)
        cards=SerialDebateMemoryRegression.science(self,wf,'navigation',count=3,large=True)
        tid=wf.enqueue_role('directory','generate',context={'round':1,'slot':'H3'})
        task=wf.queue.get(tid)
        visible=lambda c:wf.can_read_card(task,c)
        for query in ({},{'text':'observation'},{'kind':'EvidenceCard'}):
            with wf.journal.read_session():
                expected=wf.retriever.retrieve('generation',visible=visible,limit=1,query=query)
                actual=wf.retriever.retrieve('generation',visible=visible,limit=1,query=query,
                                             visibility_card=wf.visibility_card)
                self.assertEqual(actual,expected)
        with wf.journal.read_session():
            expected=wf.memory.current_research_state(visible=visible,compact=True)
            self.assertEqual(wf.memory.current_research_state(visible=visible,compact=True,
                            visibility_card=wf.visibility_card),expected)
        evidence=next(c for c in cards if c['kind']=='EvidenceCard')
        path=wf.journal.card_path(evidence['id']);broken=read_json(path)
        broken['payload']['observations']['mean']=999
        atomic_json(path,broken)
        with self.assertRaises(ContractError):
            wf.retriever.retrieve('generation',visible=visible,limit=1,visibility_card=wf.visibility_card)

    def test_run_returns_resource_blocked_with_durable_ready_task(self):
        wf=self.workflow(1)
        tid=wf.queue.enqueue('numeric','experiment');wf.queue.ready()
        def dispatch():
            wf.note_resource_wait(wf.queue.get(tid),ResourceUnavailable('no authorized device'))
            return 0
        with patch.object(wf,'advance',return_value=False),patch.object(wf,'dispatch',side_effect=dispatch):
            result=wf.run()
        self.assertEqual(result['status'],'resource_blocked')
        self.assertTrue(result['state_saved'])
        self.assertEqual(read_json(wf.run_dir/'status.json')['status'],'resource_blocked')
        self.assertEqual(Workflow(wf.run_dir).queue.get(tid)['state'],'ready')

    def test_trusted_implementation_failure_does_not_transport_retry_same_source(self):
        wf=self.workflow(1)
        tid=wf.queue.enqueue('numeric','experiment');wf.queue.ready();task=wf.queue.claim(tid)
        error={'code':'InterruptedError','origin':'implementation','message':'source-specific failure'}
        with patch.object(wf.runtime,'settle'):
            wf.accept(task,{'status':'failed','error':error},wf.run_dir/'jobs/fixture/result.json')
        self.assertEqual(wf.queue.get(tid)['state'],'failed')
        self.assertTrue(ExperimentRequests.revisable_failure(wf.queue.get(tid)))
    def test_unavailable_and_busy_wait_preserve_budget_and_attempts(self):
        wf=self.workflow(1)
        tid=wf.queue.enqueue('numeric','experiment');wf.queue.ready();task=wf.queue.get(tid)
        with patch('runtime.driver.time.monotonic',return_value=10):
            wf.note_resource_wait(task,ResourceBusy('leased'))
            self.assertIsNone(wf.resource_blockage())
        with patch('runtime.driver.time.monotonic',return_value=910):
            self.assertEqual(wf.resource_blockage()['code'],'resource_blocked')
        self.assertEqual(wf.queue.get(tid)['attempt'],0)
        self.assertFalse(wf.budget.snapshot()['reservations'])
        restored=Workflow(wf.run_dir)
        self.assertIsNone(restored.resource_blockage())
        restored.note_resource_wait(restored.queue.get(tid),ResourceUnavailable('no authorized device'))
        self.assertEqual(restored.resource_blockage()['tasks'][0]['task_id'],tid)
        self.assertEqual(restored.queue.get(tid)['state'],'ready')

    def test_live_sibling_resets_no_progress_wait_and_status_write_is_bounded(self):
        wf=self.workflow(1)
        a=wf.queue.enqueue('a','experiment');b=wf.queue.enqueue('b','experiment');wf.queue.ready()
        with patch('runtime.driver.time.monotonic',return_value=0):
            wf.note_resource_wait(wf.queue.get(a),ResourceBusy('busy'))
        wf.queue.claim(b)
        with patch('runtime.driver.time.monotonic',return_value=2000):
            self.assertIsNone(wf.resource_blockage())
        wf.queue.finish(b)
        with patch('runtime.driver.time.monotonic',return_value=2001):
            self.assertIsNone(wf.resource_blockage())
        with patch('runtime.driver.atomic_json') as write,patch('runtime.driver.time.monotonic',return_value=2001):
            for _ in range(20):wf.persist_status('running')
            self.assertEqual(write.call_count,1)

    def test_journal_detects_external_append_deletion_and_budget_projection_is_independent(self):
        journal=self.journal();budget=BudgetLedger(journal,{'wall_seconds':100},clock=lambda:10)
        reader=Journal(journal.root,journal.run_id)
        with patch.object(Path,'glob',wraps=Path.glob) as glob:
            reader.refresh();self.assertEqual(glob.call_count,0)
        journal.commit('append',events=[{'type':'budget_extended','seconds':20}])
        reader.refresh();self.assertEqual(reader.sequence,journal.sequence)
        state=budget.snapshot();self.assertEqual(state['remaining_seconds'],120)
        state['extensions'][0]['seconds']=999;state['config']['wall_seconds']=0
        self.assertEqual(budget.snapshot()['remaining_seconds'],120)
        (journal.root/'commits/000000002.json').unlink()
        with self.assertRaises(ContractError):reader.refresh()

    def test_offline_files_archive_without_changing_runtime_or_numeric_identity(self):
        from runtime import identity
        files={'runtime/worker.py':'x=1','tools/numeric.py':'x=1',
               'configs/evaluation/v1/rules.json':'{}','tools/evaluation/offline.py':'x=1',
               'tests/test_only.py':'x=1','agents/prompts/judger.md':'history'}
        for name,body in files.items():
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body)
        with patch.object(identity,'QAVIT',self.root):
            before=identity.code_identity();numeric=identity.measurement_implementation()
            for name in list(files)[2:]: (self.root/name).write_text(files[name]+'\n')
            after=identity.code_identity()
            self.assertEqual(before['core_sha256'],after['core_sha256'])
            self.assertNotEqual(before['archive_sha256'],after['archive_sha256'])
            self.assertEqual(numeric,identity.measurement_implementation())
            self.assertIn('tests/test_only.py',identity.code_snapshot(after))
            (self.root/'tools/numeric.py').write_text('x=2')
            self.assertNotEqual(after['core_sha256'],identity.code_identity()['core_sha256'])
            self.assertNotEqual(numeric,identity.measurement_implementation())

if __name__=='__main__':unittest.main()
