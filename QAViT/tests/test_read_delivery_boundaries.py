"""Read delivery and immutable ancestor coverage regressions; CPU only."""
import sys
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.cards import canonical, fingerprint, make_card, ContractError
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as delivery
from tests.support import hypothesis, experiment_payload

class ReadDeliveryBoundaries(TempCase):
    task=delivery.DeliveryRegression.task
    begin=delivery.DeliveryRegression.begin
    tools=delivery.DeliveryRegression.tools
    science=delivery.DeliveryRegression.science
    screening=delivery.DeliveryRegression.screening

    def test_result_batch_is_complete_isolated_and_does_not_copy_queue_per_row(self):
        from unittest.mock import patch
        wf=self.workflow(1);rows=[]
        for i in range(3):
            tid=wf.queue.enqueue('batch-result-'+str(i),'host_tool')
            wf.queue.ready();wf.queue.claim(tid)
            wf.queue.finish(tid,result={'nested':[i,{'scientific_value':i/10}]})
            rows.append({'task':tid,'name':'read_card'})
        expected=[(r,wf.queue.get(r['task']),wf.queue.get(r['task'])['result']) for r in rows]
        with patch.object(wf.queue,'snapshot',wraps=wf.queue.snapshot) as snapshots:
            actual=list(wf.result_values({'tool_results':rows}))
            self.assertEqual(actual,expected)
            self.assertEqual(snapshots.call_count,1)
        actual[0][2]['nested'][1]['scientific_value']=999
        self.assertEqual(wf.queue.get(rows[0]['task'])['result']['nested'][1]['scientific_value'],0)
        with patch.object(wf.queue,'snapshot',side_effect=AssertionError('empty results need no queue read')):
            self.assertEqual(list(wf.result_values({'tool_results':[]})),[])
        with self.assertRaisesRegex(ContractError,'未知任务'):
            list(wf.result_values({'tool_results':[{'task':'missing'}]}))

    def test_unrelated_read_is_not_loaded_or_counted_as_target_coverage(self):
        from unittest.mock import patch
        wf=self.workflow(1);cards=self.science(wf,'unrelated_coverage')
        target,unrelated=cards[1],cards[2]
        task,_=self.begin(wf,self.task(wf,refs=[target['id'],unrelated['id']]))
        task,_=self.tools(wf,task,[('read_card',{'ref':unrelated['id']})])
        with patch.object(wf.journal,'card',wraps=wf.journal.card) as reads:
            self.assertEqual(wf.read_ranges(task,target['id'],'/payload',fingerprint(target['payload'])),[])
            self.assertEqual([call.args[0] for call in reads.call_args_list],[target['id']])
        task,_=self.tools(wf,task,[('read_card',{'ref':target['id']})])
        self.assertEqual(wf.read_ranges(task,target['id'],'/payload',fingerprint(target['payload'])),[(0,float('inf'))])

    def test_stable_task_input_retains_ancestor_coverage_across_versions(self):
        from unittest.mock import patch
        wf=self.workflow(1);task=self.task(wf)
        inputs=deepcopy(task['inputs']);inputs['context']['stable']='scientific input'
        wf.queue.change(task['task_id'],task['state'],patch={'inputs':inputs})
        task=wf.queue.get(task['task_id']);old=wf.journal.card(task['task_card'])
        wf.queue.change(task['task_id'],task['state'],patch={'step':1})
        current=wf.queue.get(task['task_id']);field='/payload/inputs/context/stable'
        child={'state':'completed','kind':'host_tool','inputs':{'name':'read_card'}}
        with patch.object(wf,'result_values',return_value=iter([({},child,old)])):
            self.assertEqual(wf.read_ranges(current,current['task_card'],field,fingerprint('scientific input')),[(0,float('inf'))])

    def test_bulk_array_state_keeps_every_site_statistic_and_exact_read_access(self):
        wf=self.workflow(1);cards=self.science(wf,'bulk_arrays',commit=False)
        payload=deepcopy(cards[1]['payload'])
        payload['observations']={'rows':[{'site':'block.'+str(i), 'nmse':i/100,
            'input_count':32, 'states':{'T':{'scale':list(range(2000))}}} for i in range(49)],
            'summary':{'valid_count':49,'median_ratio':0.25}}
        evidence=make_card('EvidenceCard',wf.journal.run_id,payload,refs=[cards[0]['id']])
        wf.journal.commit('bulk_arrays',cards=[cards[0],evidence])
        task,c=self.begin(wf,self.task(wf,refs=[evidence['id']]))
        self.assertFalse(any(r['ref']==evidence['id'] for r in c['memory']['required_reads']))
        view=next(v for v in c['memory']['cards'] if v.get('id')==evidence['id'])
        self.assertGreater(len(canonical(view['payload'])),wf.PAGE_CHARACTERS)
        self.assertLessEqual(len(canonical(view['payload'])),4*wf.PAGE_CHARACTERS)
        shown=view['payload']['observations']
        self.assertEqual(shown['summary'],payload['observations']['summary'])
        for original,row in zip(payload['observations']['rows'],shown['rows']):
            self.assertEqual({k:v for k,v in row.items() if k!='states'},
                             {k:v for k,v in original.items() if k!='states'})
            entry=row['states'];args=entry['read_card']
            self.assertEqual(wf.card_field(evidence,args['field']),original['states'])
            self.assertEqual(entry['sha256'],fingerprint(original['states']))
        args=shown['rows'][0]['states']['read_card']
        task,c=self.tools(wf,task,[('read_card',args)])
        self.assertEqual(wf.read_ranges(task,evidence['id'],args['field'],
            fingerprint(payload['observations']['rows'][0]['states'])),[(0,float('inf'))])
        self.assertEqual(wf.journal.card(evidence['id'])['payload'],payload)
        wf.submit(task,self.screening(),'CPU')

    def test_bulk_projection_preserves_named_scalars_flags_and_escaped_paths(self):
        wf=self.workflow(1);cards=self.science(wf,'bulk_paths',commit=False)
        payload=deepcopy(cards[1]['payload'])
        payload['observations']={'named_statistics':{'metric_'+str(i):i/100 for i in range(1000)},
            'flags':[True]*2000,'labels':['scientific label']*1000,
            'a/b~c':list(range(2000)), 'small_array':[0.1,0.2], 'meaning':'retain the whole explanation'}
        evidence=make_card('EvidenceCard',wf.journal.run_id,payload,refs=[cards[0]['id']])
        view=wf.present_card(evidence)['payload']['observations']
        for key in payload['observations']:
            if key!='a/b~c':self.assertEqual(view[key],payload['observations'][key])
        entry=view['a/b~c']
        self.assertEqual(entry['read_card']['field'],'/payload/observations/a~1b~0c')
        self.assertEqual(wf.card_field(evidence,entry['read_card']['field']),payload['observations']['a/b~c'])
        self.assertEqual(entry['sha256'],fingerprint(payload['observations']['a/b~c']))
        payload['observations']['meaning']='required full scientific text '*8000
        large=make_card('EvidenceCard',wf.journal.run_id,payload,refs=[cards[0]['id']])
        wf.journal.commit('large_science_with_arrays',cards=[cards[0],large])
        task,c=self.begin(wf,self.task(wf,refs=[large['id']]))
        self.assertTrue(any(r['ref']==large['id'] for r in c['memory']['required_reads']))
        with self.assertRaises(ContractError):wf.submit(task,self.screening(),'CPU')

    def test_repeated_counts_are_exact_references_and_unique_results_remain_complete(self):
        wf=self.workflow(1);cards=self.science(wf,'repeated_counts',commit=False)
        counts={'block.'+str(i):32 for i in range(97)}
        changed={**counts,'block.96':31}
        payload=deepcopy(cards[1]['payload'])
        payload['observations']={'a/b~c':counts,'rows':[
            {'phase':'replay:'+str(i),'counts':deepcopy(counts),'nmse':i/100} for i in range(97)],
            'different':changed,'unique_statistics':{'metric_'+str(i):i/100 for i in range(2000)}}
        evidence=make_card('EvidenceCard',wf.journal.run_id,payload,refs=[cards[0]['id']])
        wf.journal.commit('repeated_counts',cards=[cards[0],evidence])
        task,c=self.begin(wf,self.task(wf,refs=[evidence['id']]))
        self.assertFalse(any(r['ref']==evidence['id'] for r in c['memory']['required_reads']))
        shown=next(v for v in c['memory']['cards'] if v.get('id')==evidence['id'])['payload']['observations']
        self.assertEqual(shown['a/b~c'],counts)
        self.assertEqual(shown['different'],changed)
        self.assertEqual(shown['unique_statistics'],payload['observations']['unique_statistics'])
        self.assertGreater(len(canonical(shown)),4*wf.PAGE_CHARACTERS)
        for i,row in enumerate(shown['rows']):
            self.assertEqual(row['phase'],'replay:'+str(i))
            self.assertEqual(row['nmse'],i/100)
            self.assertEqual(row['counts']['read_card'],{'ref':evidence['id'],'field':'/payload/observations/a~1b~0c'})
            self.assertEqual(row['counts']['sha256'],fingerprint(counts))
            self.assertEqual(wf.card_field(evidence,row['counts']['read_card']['field']),counts)
        self.assertEqual(wf.journal.card(evidence['id'])['payload'],payload)
        wf.submit(task,self.screening(),'CPU')

    def test_only_exact_bound_research_contract_is_delivered_through_brief(self):
        wf=self.workflow(1);p=deepcopy(wf.brief['payload'])
        p['quantization_contract']={'required_sites':['site-'+str(i) for i in range(400)]}
        brief=make_card('ResearchBrief',wf.journal.run_id,p)
        wf.journal.commit('scope_brief',cards=[brief]);wf.brief=brief
        payload=experiment_payload();payload['execution']['research_quantization_contract']=deepcopy(p['quantization_contract'])
        payload['execution']['quantization_contract']={'scientific_intervention':'KEEP_EXACT_ACTUAL_CONTRACT'}
        card=make_card('ExperimentCard',wf.journal.run_id,payload,refs=[brief['id']])
        shown=wf.present_card(card)['payload']['execution']
        self.assertEqual(shown['research_quantization_contract']['read_card'],
                         {'ref':brief['id'],'field':'/payload/quantization_contract'})
        self.assertEqual(shown['research_quantization_contract']['sha256'],fingerprint(p['quantization_contract']))
        self.assertEqual(shown['quantization_contract'],payload['execution']['quantization_contract'])
        self.assertEqual(card['payload'],payload)
        payload['execution']['research_quantization_contract']['required_sites'].append('different-site')
        different=make_card('ExperimentCard',wf.journal.run_id,payload,refs=[brief['id']])
        self.assertEqual(wf.present_card(different)['payload']['execution']['research_quantization_contract'],
                         payload['execution']['research_quantization_contract'])

    def test_one_page_evidence_is_complete_and_larger_science_still_requires_reading(self):
        wf=self.workflow(1);cards=self.science(wf,'bounded_evidence',commit=False)
        payload=deepcopy(cards[1]['payload']);payload['observations']['details']='EVIDENCE_FULL_SCIENCE_'+('x'*7400)
        evidence=make_card('EvidenceCard',wf.journal.run_id,payload,refs=[cards[0]['id']])
        wf.journal.commit('bounded_evidence',cards=[cards[0],evidence])
        task,c=self.begin(wf,self.task(wf,refs=[evidence['id']]))
        self.assertFalse(any(r['ref']==evidence['id'] for r in c['memory']['required_reads']))
        shown=next(v for v in c['memory']['cards'] if v.get('id')==evidence['id'])
        self.assertEqual(shown['payload']['observations'],payload['observations'])
        wf.submit(task,self.screening(),'CPU')
        payload['observations']['details']+='y'*12000
        large=make_card('EvidenceCard',wf.journal.run_id,payload,refs=[cards[0]['id']])
        wf.journal.commit('large_evidence',cards=[large])
        task,c=self.begin(wf,self.task(wf,refs=[large['id']]))
        self.assertTrue(any(r['ref']==large['id'] and r['next_offset']==0 for r in c['memory']['required_reads']))
        with self.assertRaises(ContractError):wf.submit(task,self.screening(),'CPU')

    def test_bounded_brief_is_whole_without_relaxing_scientific_body_gate(self):
        wf=self.workflow(1)
        p=deepcopy(wf.brief['payload']);p['scientific_question']='BOUNDED_BRIEF_'+('x'*9000)
        brief=make_card('ResearchBrief',wf.journal.run_id,p)
        hp=hypothesis(1,'H1');hp['mechanism']='BOUND_HYPOTHESIS_'+('y'*9000)
        card=make_card('HypothesisCard',wf.journal.run_id,hp)
        wf.journal.commit('bounded_brief',cards=[brief,card]);wf.brief=brief
        task,c=self.begin(wf,self.task(wf,refs=[card['id']]))
        self.assertEqual(c['research_brief'],brief)
        self.assertIn(brief['id'],c['memory']['read_manifest']['refs'])
        self.assertEqual({r['ref'] for r in c['memory']['required_reads']},{card['id']})
        with self.assertRaises(ContractError):wf.submit(task,self.screening(),'CPU')
        task,c=self.tools(wf,task,[('read_card',{'ref':card['id'],'field':'/payload'})])
        self.assertEqual(canonical(c).count('BOUNDED_BRIEF_'),1)
        wf.submit(task,self.screening(),'CPU')

    def test_large_brief_still_requires_every_page(self):
        wf=self.workflow(1)
        p=deepcopy(wf.brief['payload']);p['scientific_question']='LARGE_BRIEF_'+('x'*18000)
        brief=make_card('ResearchBrief',wf.journal.run_id,p)
        wf.journal.commit('large_brief',cards=[brief]);wf.brief=brief
        task,c=self.begin(wf,self.task(wf,refs=[brief['id']]))
        self.assertEqual(c['memory']['required_reads'][0]['ref'],brief['id'])
        task,c=self.tools(wf,task,[('read_card',{'ref':brief['id'],'field':'/payload','offset':0})])
        with self.assertRaises(ContractError):wf.submit(task,self.screening(),'CPU')
        task,c=self.tools(wf,task,[('read_card',{'ref':brief['id'],'field':'/payload','offset':12000})])
        self.assertIsNone(c['memory']['required_reads'][0]['next_offset'])
        self.assertEqual(canonical(c).count('LARGE_BRIEF_'),1)
        wf.submit(task,self.screening(),'CPU')

    def test_omitted_identity_expansion_survives_next_request_once(self):
        wf=self.workflow(1);cards=self.science(wf,'identity',commit=False)
        p=deepcopy(cards[1]['payload']);p['provenance']['model_identity']={'name':'OMITTED_MODEL_IDENTITY_SENTINEL'}
        e=make_card('EvidenceCard',wf.journal.run_id,p,refs=[cards[0]['id']])
        wf.journal.commit('identity',cards=[cards[0],e])
        task,c=self.begin(wf,self.task(wf,refs=[e['id']]))
        self.assertNotIn('OMITTED_MODEL_IDENTITY_SENTINEL',canonical(c))
        args={'ref':e['id'],'field':'/payload/provenance/model_identity'}
        task,c=self.tools(wf,task,[('read_card',args)])
        self.assertEqual(canonical(c).count('OMITTED_MODEL_IDENTITY_SENTINEL'),1)
        task,c=self.tools(wf,task,[('read_card',args)])
        self.assertEqual(canonical(c).count('OMITTED_MODEL_IDENTITY_SENTINEL'),1)

    def test_engineering_business_status_and_full_result_expand_once(self):
        wf=self.workflow(1);tid=wf.queue.enqueue('local','experiment_request')
        wf.queue.ready();wf.queue.claim(tid);wf.queue.finish(tid,result={'status':'rejected','reason':'LOCAL_REJECTION','detail':'HIDDEN_FULL_RESULT'})
        card=wf.journal.card(wf.queue.get(tid)['task_card'])
        view=wf.present_card(card)
        self.assertEqual(view['payload']['result']['status'],'rejected')
        self.assertIn('LOCAL_REJECTION',canonical(view))
        task,c=self.begin(wf,self.task(wf,refs=[card['id']]))
        self.assertNotIn('HIDDEN_FULL_RESULT',canonical(c))
        task,c=self.tools(wf,task,[('read_card',{'ref':card['id']})])
        self.assertEqual(canonical(c).count('HIDDEN_FULL_RESULT'),1)
        self.assertEqual(canonical(c).count('LOCAL_REJECTION'),1)
        task,c=self.tools(wf,task,[('read_asset',{'asset':str(wf.journal.card_path(card['id']))})])
        self.assertEqual(canonical(c).count('HIDDEN_FULL_RESULT'),1)

    def test_root_pages_cover_payload_but_missing_page_rejected(self):
        wf=self.workflow(1);p=hypothesis(1,'H1');p['mechanism']='ROOT_PAGES_'+('x'*26000)
        card=make_card('HypothesisCard',wf.journal.run_id,p);wf.journal.commit('long',cards=[card])
        task,c=self.begin(wf,self.task(wf,refs=[card['id']]))
        task,c=self.tools(wf,task,[('read_card',{'ref':card['id'],'offset':i}) for i in (0,24000)])
        with self.assertRaises(ContractError):wf.submit(task,self.screening(),'CPU')
        task,c=self.tools(wf,task,[('read_card',{'ref':card['id'],'offset':12000})])
        self.assertIsNone(c['memory']['required_reads'][0]['next_offset'])
        wf.submit(task,self.screening(),'CPU')

    def test_ancestor_field_covers_child_with_exact_identity(self):
        wf=self.workflow(1);cards=self.science(wf,'ancestor');card=cards[1]
        task,c=self.begin(wf,self.task(wf,refs=[card['id']]))
        task,c=self.tools(wf,task,[('read_card',{'ref':card['id'],'field':'/payload'})])
        field='/payload/observations'
        self.assertEqual(wf.read_ranges(task,card['id'],field,fingerprint(card['payload']['observations'])),[(0,float('inf'))])
        self.assertEqual(wf.read_ranges(task,card['id'],field,'incorrect-hash'),[])
        p=deepcopy(card['payload']);p['observations']['mean']=99
        changed=make_card('EvidenceCard',wf.journal.run_id,p,previous=card,refs=card['refs']);wf.journal.commit('changed',cards=[changed])
        self.assertEqual(wf.read_ranges(task,changed['id'],field,fingerprint(p['observations'])),[])

    def test_raw_asset_expansion_is_kept_once_and_covers_card(self):
        wf=self.workflow(1);cards=self.science(wf,'assetidentity',commit=False)
        p=deepcopy(cards[1]['payload']);p['provenance']['data_identity']={'ids':['RAW_ASSET_IDENTITY_SENTINEL']}
        e=make_card('EvidenceCard',wf.journal.run_id,p,refs=[cards[0]['id']]);wf.journal.commit('asset',cards=[cards[0],e])
        task,c=self.begin(wf,self.task(wf,refs=[e['id']]))
        args={'asset':str(wf.journal.card_path(e['id']))}
        task,c=self.tools(wf,task,[('read_asset',args)])
        self.assertEqual(canonical(c).count('RAW_ASSET_IDENTITY_SENTINEL'),1)
        self.assertEqual(wf.read_ranges(task,e['id'],'/payload',fingerprint(p)),[(0,float('inf'))])
        task,c=self.tools(wf,task,[('read_asset',args)])
        self.assertEqual(canonical(c).count('RAW_ASSET_IDENTITY_SENTINEL'),1)

    def test_present_observation_field_is_equivalent_but_bad_pages_do_not_cover(self):
        from unittest.mock import patch
        wf=self.workflow(1);cards=self.science(wf,'FIELD_EQUIVALENT');e=cards[1]
        task,c=self.begin(wf,self.task(wf,refs=[e['id']]))
        task,c=self.tools(wf,task,[('read_card',{'ref':e['id'],'field':'/payload/observations'})])
        self.assertEqual(canonical(c).count('FIXTURE observation FIELD_EQUIVALENT/0'),1)
        rows=list(wf.result_values(task));row,child,result=rows[0]
        bad=deepcopy(result);bad['sha256']='bad-hash'
        with patch.object(wf,'result_values',return_value=iter([(row,child,bad)])):
            self.assertEqual(wf.read_ranges(task,e['id'],'/payload/observations',fingerprint(e['payload']['observations'])),[])

    def test_id_collision_is_not_complete_body_and_new_task_field_is_unread(self):
        from unittest.mock import patch
        wf=self.workflow(1);cards=self.science(wf,'collision');card=cards[2]
        task,c=self.begin(wf,self.task(wf,refs=[card['id']]))
        task,c=self.tools(wf,task,[('read_card',{'ref':card['id']})])
        rows=list(wf.result_values(task));row,child,result=rows[0]
        with patch.object(wf,'result_values',return_value=iter([(row,child,{'id':card['id'],'extra':'NOT_EQUIVALENT'})])):
            self.assertIn('NOT_EQUIVALENT',canonical(wf.compact_results(task,{card['id']:card})))
        old=wf.journal.card(task['task_card']);field='/payload/inputs/context/added_later'
        inputs=deepcopy(task['inputs']);inputs['context']['added_later']='new field'
        wf.queue.change(task['task_id'],'running',patch={'inputs':inputs})
        new=wf.queue.get(task['task_id'])
        with patch.object(wf,'result_values',return_value=iter([(row,child,old)])):
            self.assertEqual(wf.read_ranges(new,new['task_card'],field,fingerprint('new field')),[])
