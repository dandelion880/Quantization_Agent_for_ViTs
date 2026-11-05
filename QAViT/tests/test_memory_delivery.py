"""MemFS-inspired delivery: actual request/tool/submit chain, entirely scripted CPU."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, canonical, fingerprint, make_card
from core.journal import atomic_json, file_sha256
from core.workflow import Workflow
from tests.support import hypothesis, review, meta, experiment_payload
from tests.test_infrastructure import TempCase
from tests import test_serial_debate_memory as serial


class DeliveryRegression(TempCase):
    science = serial.SerialDebateMemoryRegression.science
    debate_task = serial.SerialDebateMemoryRegression.debate_task
    read = serial.SerialDebateMemoryRegression.read
    index = serial.SerialDebateMemoryRegression.index
    full_card = serial.SerialDebateMemoryRegression.full_card

    def setUp(self):
        super().setUp()
        worker = patch('runtime.task_runtime.TaskRuntime.launch', side_effect=AssertionError('scripted CPU: no LLM/GPU workers'))
        worker.start(); self.addCleanup(worker.stop)

    def task(self, wf, purpose='screen', refs=(), **context):
        tid = wf.enqueue_role('fixture/' + purpose + '/' + str(wf.journal.sequence), purpose, refs=refs, context=context)
        return wf.queue.get(tid)

    def begin(self, wf, task):
        wf.queue.ready()
        task = wf.queue.get(task['task_id'])
        body = wf.request_body(task)
        path = wf.run_dir / 'jobs' / task['task_id'] / str(task['attempt'] + 1) / 'request.json'
        atomic_json(path, body)
        return wf.queue.claim(task['task_id'], request_path=str(path)), body['spec']['context']

    def tools(self, wf, task, calls):
        wf.receive_tools(task, {'tool_calls': [{'id': str(i), 'name': name, 'arguments': args}
                        for i, (name, args) in enumerate(calls)]}, 'SCRIPTED')
        waiting = wf.queue.get(task['task_id'])
        wf.queue.ready()
        for tid in waiting['waiting_for']:
            child = wf.queue.claim(tid)
            self.assertEqual(child['kind'], 'host_tool')
            result = wf.host_tool(child)
            wf.queue.finish(tid, result=result)
        return self.begin(wf, wf.queue.get(task['task_id']))

    def screening(self):
        return review('screening', eligible=True, hard_rejection_reasons=[], similarity_analysis='CPU fixture', evidence_gaps=[])

    def test_purpose_bound_objects_and_same_version_body_dedup(self):
        wf = self.workflow(1)
        cards = self.science(wf, 'visible', count=4)
        chosen = cards[2]
        task = self.task(wf, refs=[wf.brief['id'], chosen['id']], round=1, hypothesis_ref=chosen['id'])
        task, context = self.begin(wf, task)
        self.assertEqual([c['id'] for c in context['memory']['cards']],
                         [chosen['id'], cards[3]['id'], cards[1]['id'], cards[0]['id']])
        self.assertEqual(context['research_brief'], wf.brief)
        task, context = self.tools(wf, task, [('read_card', {'ref': chosen['id']}),
            ('read_card', {'ref': wf.brief['id']})])
        text = canonical(wf.request_body(task)['spec'])
        self.assertEqual(text.count(chosen['payload']['mechanism']), 1)
        self.assertEqual(text.count(wf.brief['payload']['scientific_question']), 1)
        wf.submit(task, self.screening(), 'SCRIPTED')

    def test_long_body_pages_remain_readable_across_restart(self):
        wf = self.workflow(1)
        payload = hypothesis(1, 'H1'); payload['mechanism'] = 'LONG_BOUND_MECHANISM_' + 'x' * 26000
        card = make_card('HypothesisCard', wf.journal.run_id, payload)
        wf.journal.commit('large', cards=[card])
        task, context = self.begin(wf, self.task(wf, refs=[card['id']], round=1, hypothesis_ref=card['id']))
        self.assertEqual(context['memory']['required_reads'][0]['next_offset'], 0)
        task, context = self.tools(wf, task, [('read_card', {'ref': card['id'], 'field': '/payload', 'offset': 0})])
        self.assertEqual(context['memory']['required_reads'][0]['next_offset'], 12000)
        self.assertEqual(context['tool_results'][0]['arguments']['field'], '/payload')
        wf = Workflow(wf.run_dir)
        task = wf.queue.get(task['task_id'])
        task, context = self.tools(wf, task, [('read_card', {'ref': card['id'], 'field': '/payload', 'offset': offset})
                                              for offset in (12000, 24000)])
        self.assertIsNone(context['memory']['required_reads'][0]['next_offset'])
        pages = [r['result']['text'] for r in context['tool_results'] if 'text' in r.get('result', {})]
        self.assertEqual(json.loads(''.join(pages)), payload)
        self.assertEqual(canonical(context).count('LONG_BOUND_MECHANISM_'), 1)
        wf.submit(task, self.screening(), 'SCRIPTED')

    def test_future_cards_assets_and_indexes_are_hidden_until_next_call(self):
        wf = self.workflow(1)
        task, old = self.begin(wf, self.task(wf, 'generate', refs=[wf.brief['id']], round=1, slot='H1', hypothesis_type='existing_method'))
        cards = self.science(wf, 'FUTURE', large=True)
        for name, args in [('read_card', {'ref': cards[1]['id']}),
                           ('read_asset', {'asset': str(wf.journal.card_path(cards[1]['id']))}),
                           ('read_asset', {'asset': cards[1]['payload']['artifacts'][0]['path']})]:
            with self.assertRaises(ContractError): self.read(wf, task, name, **args)
        page = self.read(wf, task, 'retrieve_memory', query={'browse': True})
        self.assertNotIn('FUTURE', canonical(page))
        self.assertEqual(page['memory_version'], old['read_view']['memory_version'])
        task, context = self.tools(wf, task, [('retrieve_memory', {'query': {'browse': True}})])
        self.assertIn('FUTURE', canonical(context['memory']['directory']))
        self.assertEqual(self.full_card(wf, task, cards[2]['id']), cards[2])

    def test_same_exchange_transitive_and_asset_isolation_and_reference(self):
        wf = self.workflow(1)
        cards = self.science(wf, 'public')
        own, other = self.debate_task(wf), self.debate_task(wf, side='B')
        wf.queue.ready(); other = wf.queue.claim(other['task_id'])
        secret_path = wf.run_dir / 'jobs/secret.txt'; secret_path.parent.mkdir(parents=True); secret_path.write_text('PRIVATE_SIDE_B')
        secret = make_card('ReviewDecisionCard', wf.journal.run_id, review('position', round=1, pair=['H1','H2'],
            exchange=1, side='B', arguments=['PRIVATE_SIDE_B'], artifacts=[{'path': str(secret_path), 'sha256': file_sha256(secret_path)}]))
        wf.queue.finish(other['task_id'], cards=[secret], outputs=[secret['id']])
        linked = make_card('ReviewDecisionCard', wf.journal.run_id, review('interpretation', rationale='INDIRECT_PRIVATE'), refs=[secret['id']])
        wf.journal.commit('indirect', cards=[linked])
        own, context = self.begin(wf, own)
        self.assertNotIn('PRIVATE', canonical(context))
        self.assertNotIn(linked['id'], self.index(wf, own))
        for ref in (secret['id'], linked['id'], wf.queue.get(other['task_id'])['task_card']):
            with self.assertRaises(ContractError): self.read(wf, own, 'read_card', ref=ref)
        with self.assertRaises(ContractError): self.read(wf, own, 'read_asset', asset=str(secret_path))
        wf.queue.finish(own['task_id'])

    def test_generation_evolution_discover_experiments_statistics_and_interpretations(self):
        for purpose in ('generate', 'evolve'):
            # TempCase.workflow has a fixed output name; each loop gets a separate config path.
            from tests.support import validation_config
            cfg = validation_config(); cfg['budget'].update(max_rounds=1, min_free_bytes=0)
            wf = Workflow(self.root / purpose, cfg)
            cards = self.science(wf, 'old_public', count=15)
            parent = cards[2]
            kwargs = {'round': 2, 'slot': 'H1', 'hypothesis_type': 'existing_method'}
            refs = [wf.brief['id']]
            if purpose == 'evolve':
                refs.append(parent['id']); kwargs.update(parent_ref=parent['id'], parent_hypothesis=parent)
            task = self.task(wf, purpose, refs=refs, **kwargs)
            index = self.index(wf, task)
            self.assertTrue({c['id'] for c in cards} <= index.keys())
            for kind in ('ExperimentCard', 'EvidenceCard', 'ReviewDecisionCard', 'HypothesisCard'):
                chosen = next(c for c in cards if c['kind'] == kind)
                self.assertEqual(self.full_card(wf, task, chosen['id']), chosen)

    def test_optional_reads_are_retained_once_across_more_than_four_tools(self):
        wf = self.workflow(1)
        cards = self.science(wf, 'optional', count=3)
        task, context = self.begin(wf, self.task(wf, 'generate', refs=[wf.brief['id']], round=1, slot='H1', hypothesis_type='existing_method'))
        targets = [cards[i]['id'] for i in (0,2,3,4,6,7)]
        task, context = self.tools(wf, task, [('read_card', {'ref': ref}) for ref in targets])
        task, context = self.tools(wf, task, [('read_card', {'ref': targets[0]})])
        full = [row['result'] for row in context['tool_results'] if 'id' in (row.get('result') or {})]
        self.assertEqual({c['id'] for c in full}, set(targets))
        self.assertEqual(sum(c['id'] == targets[0] for c in full), 1)

    def recorder(self, wf, key='fixture/record'):
        tid = wf.enqueue_recorder(1, incomplete=True, key=key)
        return wf.queue.get(tid)

    def record_payload(self, task):
        c = task['inputs']['context']
        return meta(c['round'], c['recording_batch']['judgment_refs'])

    def test_recorder_l3_engineering_results_are_compact_and_original_is_readable(self):
        wf = self.workflow(1)
        self.science(wf, 'engineering')
        tid = wf.queue.enqueue('r001/L3', 'experiment_request')
        wf.queue.ready(); wf.queue.claim(tid)
        wf.queue.finish(tid, result={'status': 'rejected', 'reason': 'fixture rejection reason',
                                     'verification_log': 'ENGINEERING_LONG_LOG_' + 'x' * 26000})
        task, context = self.begin(wf, self.recorder(wf))
        rendered = canonical(context)
        self.assertIn('fixture rejection reason', rendered)
        self.assertNotIn('ENGINEERING_LONG_LOG_', rendered)
        self.assertFalse(any(r['field'].endswith('/l3_outcome') for r in context['memory']['required_reads']))
        fact = context['task']['l3_outcome']['tasks'][0]
        original = self.read(wf, task, 'read_card', **fact['read_card'])
        self.assertIn('ENGINEERING_LONG_LOG_', original['text'])
        wf.submit(task, self.record_payload(task), 'SCRIPTED')

    def test_recorder_automatic_sources_selective_analysis_atomic_cursor_and_idempotence(self):
        wf = self.workflow(1)
        cards = self.science(wf, 'batch')
        task, context = self.begin(wf, self.recorder(wf))
        value = self.record_payload(task)
        forged = deepcopy(value); forged['source_refs'] = []
        with self.assertRaisesRegex(ContractError, '来源绑定'): wf.submit(task, forged, 'SCRIPTED')
        self.assertTrue(wf.memory.recording_batch()['source_refs'])
        value['judgment_audit'] = []  # selective synthesis does not require a row for every source.
        wf.submit(task, value, 'SCRIPTED')
        commit = wf.journal.commits[-1]
        self.assertTrue({'recorder_batch_completed', 'memory_projection', 'task_snapshot'} <= {e['type'] for e in commit['events']})
        seq = wf.journal.sequence
        wf.submit(task, value, 'SCRIPTED')
        self.assertEqual(wf.journal.sequence, seq)
        self.assertEqual(wf.memory.recording_batch()['source_refs'], [])
        self.assertEqual(wf.memory.snapshot()['L2'], {})
        wf = Workflow(wf.run_dir)
        new = self.science(wf, 'next_batch')
        later = self.recorder(wf, 'fixture/next_record')
        self.assertEqual(set(later['inputs']['refs']), {c['id'] for c in new})
        self.assertFalse({c['id'] for c in cards}.intersection(later['inputs']['refs']))

    def test_recorder_storage_failure_preserves_batch_and_retry_publishes_once(self):
        wf = self.workflow(1); self.science(wf, 'recover')
        task, _ = self.begin(wf, self.recorder(wf)); value = self.record_payload(task)
        original = wf.journal.commit
        def fail(key, **kwargs):
            if key.startswith('complete/'):
                def fault(stage):
                    if stage == 'cards_written': raise OSError('CPU injected disk failure')
                kwargs['fault'] = fault
            return original(key, **kwargs)
        with patch.object(wf.journal, 'commit', side_effect=fail), self.assertRaises(OSError):
            wf.submit(task, value, 'SCRIPTED')
        self.assertFalse(any(e['type'] == 'recorder_batch_completed' for e in wf.journal.events))
        wf = Workflow(wf.run_dir); task = wf.queue.get(task['task_id'])
        self.assertTrue(wf.memory.recording_batch()['source_refs'])
        wf.submit(task, value, 'SCRIPTED')
        self.assertEqual(wf.memory.recording_batch()['source_refs'], [])
        self.assertEqual(len(wf.journal.find_cards('MetareviewCard')), 1)

    def test_partial_recorder_repair_retains_valid_fields(self):
        wf = self.workflow(1); self.science(wf, 'repair')
        task, _ = self.begin(wf, self.recorder(wf)); value = self.record_payload(task)
        invalid = deepcopy(value); invalid['scientific_analysis'] = ''
        receipt = {'status': 'completed', 'data': {'response': {'decision': {'action': 'submit', 'content': invalid}}}}
        with patch.object(wf.runtime, 'poll', return_value=[(task, receipt, 'SCRIPTED')]), patch.object(wf.runtime, 'settle'):
            wf.poll()
        task, context = self.begin(wf, wf.queue.get(task['task_id']))
        self.assertEqual(context['repair_feedback']['rejected_response']['decision']['content'], invalid)
        bad_patch = {'status': 'completed', 'data': {'response': {'decision': {'action': 'submit', 'content': {'repair_patch': {'scientific_analysis': ''}}}}}}
        with patch.object(wf.runtime, 'poll', return_value=[(task, bad_patch, 'SCRIPTED_PATCH')]), patch.object(wf.runtime, 'settle'):
            wf.poll()
        task, context = self.begin(wf, wf.queue.get(task['task_id']))
        self.assertEqual(task['repair_count'], 2)
        self.assertEqual(context['repair_feedback']['rejected_response']['decision']['content']['generation_guidance'], value['generation_guidance'])
        wf.submit(task, {'repair_patch': {'scientific_analysis': value['scientific_analysis']}}, 'SCRIPTED')
        self.assertEqual(wf.result_card(wf.queue.get(task['task_id']))['payload'], wf.memory.prepare_metareview(value, task['inputs']['context']['recording_batch'], 1))

    def test_hard_stop_preserves_pending_without_running_recorder(self):
        wf = self.workflow(1); self.science(wf, 'interrupted')
        with patch.object(wf.budget, 'check', side_effect=KeyboardInterrupt), patch.object(wf.runtime, 'poll', return_value=[]):
            summary = wf.run()
        self.assertEqual(summary['status'], 'interrupted')
        self.assertTrue(summary['pending_recording']['source_refs'])
        self.assertFalse(any(e['type'] == 'recorder_batch_completed' for e in wf.journal.events))
        self.assertTrue(any(e['type'] == 'recorder_pending' for e in wf.journal.events))
        from runtime.budget import BudgetExhausted
        with patch.object(wf.budget, 'check', side_effect=BudgetExhausted('CPU hard budget')), patch.object(wf.runtime, 'poll', return_value=[]):
            summary = wf.run()
        self.assertEqual(summary['status'], 'budget_exhausted')
        self.assertTrue(summary['pending_recording']['source_refs'])

    def test_normal_finish_drains_late_increment_without_replaying_prior_batch(self):
        wf = self.workflow(1)
        self.science(wf, 'before')
        task, _ = self.begin(wf, self.recorder(wf)); wf.submit(task, self.record_payload(task), 'SCRIPTED')
        wf.event('round_finished', 'fixture/round_finished', round=1)
        self.assertTrue(wf.advance())
        cards = self.science(wf, 'late')
        self.assertFalse(wf.advance())
        task = next(t for t in wf.tasks().values() if t['key'].startswith('closeout/record/'))
        self.assertEqual(set(task['inputs']['refs']), {c['id'] for c in cards})
        task, _ = self.begin(wf, task); wf.submit(task, self.record_payload(task), 'SCRIPTED')
        self.assertTrue(wf.advance())

    def test_paged_scope_preserves_exact_matching_and_rejects_unverified_claims(self):
        wf = self.workflow(1)
        cards = self.science(wf, 'legacy')
        task = self.task(wf, 'generate', refs=[wf.brief['id']], round=1, slot='H1', hypothesis_type='existing_method')
        page = self.read(wf, task, 'retrieve_memory', query={'kind': 'EvidenceCard', 'model': 'tiny_vit'})
        self.assertEqual(page['entries'], [])
        self.assertEqual(self.full_card(wf, task, cards[1]['id']), cards[1])
        from core.evidence_scope import validate_claim_scope
        with self.assertRaises(ContractError): validate_claim_scope({'model': 'tiny_vit'}, {})

    def test_real_code_artifact_schema_and_long_required_asset_pages(self):
        from tools.safeguards import write_artifact
        wf = self.workflow(1)
        payload = experiment_payload(); payload['execution'].update(requires_code=True, interface='quantizer_v1')
        spec = make_card('ExperimentCard', wf.journal.run_id, payload)
        wf.journal.commit('code/spec', cards=[spec])
        source = 'def fit(x, parameters):\n    return parameters\ndef apply(x, state, parameters):\n    return x\n' + '# CPU source fixture\n' * 3000
        candidate = write_artifact(wf.run_dir, spec, {'interface': 'quantizer_v1', 'source': source}, 'coding')
        task = self.task(wf, 'safeguard', refs=[wf.brief['id'], spec['id']], experiment_ref=spec['id'],
                         candidate=candidate)
        task, context = self.begin(wf, task)
        self.assertEqual(len(context['memory']['required_assets']), 1)
        decision = review('safeguard', approved=False, fidelity='fixture', numerical_correctness='fixture', experimental_validity='fixture')
        calls = [('read_asset', {'asset': a['path'], 'offset': offset}) for a in (candidate,)
                 for offset in range(0, len(Path(a['path']).read_text()), 12000)]
        task, context = self.tools(wf, task, calls)
        self.assertTrue(all(r['next_offset'] is None for r in context['memory']['required_assets']))
        wf.submit(task, decision, 'SCRIPTED')

    def test_bounded_frozen_spec_and_authorized_sources_arrive_complete(self):
        from tools.safeguards import write_artifact
        wf = self.workflow(1)
        payload = experiment_payload()
        payload['question'] = 'CPU bounded scientific context ' + 'q' * 18000
        payload['execution'].update(requires_code=True, interface='quantizer_v1')
        spec = make_card('ExperimentCard', wf.journal.run_id, payload)
        wf.journal.commit('bounded/spec', cards=[spec])
        source = 'def fit(x, parameters):\n    return parameters\ndef apply(x, state, parameters):\n    return x\n' + '# CPU bounded source\n' * 800
        candidate = write_artifact(wf.run_dir, spec, {'interface':'quantizer_v1','source':source}, 'coding')
        task,context = self.begin(wf, self.task(wf, 'safeguard', refs=[wf.brief['id'],spec['id']],
            experiment_ref=spec['id'], candidate=candidate))
        self.assertEqual(context['original_request'],spec)
        self.assertEqual(context['candidate_source'],source)
        self.assertFalse(any(r['ref']==spec['id'] for r in context['memory']['required_reads']))
        self.assertFalse(context['memory'].get('required_assets'))
        for artifact in (candidate,):
            location=context['memory']['asset_locations'][artifact['path']]
            self.assertEqual(location['sha256'],artifact['source_sha256'])
        wf.submit(task,review('safeguard',approved=False,fidelity='CPU fixture',
            numerical_correctness='CPU fixture',experimental_validity='CPU fixture'),'SCRIPTED')

    def test_public_literature_source_is_discoverable_and_asset_hash_is_not_raw_hash(self):
        wf = self.workflow(1)
        path = wf.run_dir / 'literature/source.json'
        atomic_json(path, {'url': 'https://example.test/paper', 'pages': ['PUBLIC THEORY CPU FIXTURE']})
        tid = wf.queue.enqueue('fixture/literature', 'literature', inputs={'action': 'fetch', 'url': 'https://example.test/paper'})
        wf.queue.ready(); wf.queue.claim(tid)
        wf.queue.finish(tid, result={'asset': str(path), 'sha256': 'raw-document-hash-not-json',
                                    'text_preview': 'PUBLIC THEORY CPU FIXTURE'})
        task, context = self.begin(wf, self.task(wf, 'generate', refs=[wf.brief['id']], round=1, slot='H1', hypothesis_type='existing_method'))
        entry = next(r for r in self.index(wf, task).values() if r['kind'] == 'TaskCard')
        full = self.full_card(wf, task, entry['ref'])
        self.assertEqual(full['payload']['result']['text_preview'], 'PUBLIC THEORY CPU FIXTURE')
        result = self.read(wf, task, 'read_asset', asset=str(path))
        self.assertEqual(result['sha256'], file_sha256(path))
        self.assertIn('PUBLIC THEORY', result['text'])

    def test_long_task_field_reading_survives_taskcard_version_updates(self):
        wf = self.workflow(1)
        task, context = self.begin(wf, self.task(wf, 'supervise', diagnostic='CPU_LONG_DIAGNOSTIC_' + 'z' * 15000))
        required = context['memory']['required_reads'][0]
        task, context = self.tools(wf, task, [('read_card', {'ref': required['ref'], 'field': required['field'], 'offset': 0})])
        required = context['memory']['required_reads'][0]
        self.assertEqual(required['next_offset'], 12000)
        task, context = self.tools(wf, task, [('read_card', {'ref': required['ref'], 'field': required['field'], 'offset': 12000})])
        self.assertIsNone(context['memory']['required_reads'][0]['next_offset'])
        wf.submit(task, {'note': 'CPU state feedback only'}, 'SCRIPTED')

    def test_short_candidate_asset_is_delivered_only_once_after_read(self):
        from tools.safeguards import write_artifact
        wf = self.workflow(1)
        payload = experiment_payload(); payload['execution'].update(requires_code=True, interface='quantizer_v1')
        spec = make_card('ExperimentCard', wf.journal.run_id, payload); wf.journal.commit('spec', cards=[spec])
        source = 'def fit(x, parameters):\n    return parameters\ndef apply(x, state, parameters):\n    return x\n# UNIQUE_SHORT_CANDIDATE'
        candidate = write_artifact(wf.run_dir, spec, {'interface': 'quantizer_v1', 'source': source}, 'coding')
        task, context = self.begin(wf, self.task(wf, 'code', refs=[wf.brief['id'], spec['id']],
            experiment_ref=spec['id'], revision_feedback={'previous_candidate': candidate}))
        task, context = self.tools(wf, task, [('read_asset', {'asset': candidate['path']})])
        self.assertEqual(canonical(context).count('UNIQUE_SHORT_CANDIDATE'), 1)

    def test_knowledge_supersedes_and_contradictory_history_remain_readable(self):
        wf = self.workflow(1)
        spec = make_card('ExperimentCard', wf.journal.run_id, experiment_payload())
        wf.journal.commit('spec', cards=[spec])
        scope = {'schema_version': 'qavit.evidence_scope/1', 'model': 'CPU_contract_fixture', 'measurement_kind': 'fixture_only', 'data': {}}
        prior = None
        for revision, status in enumerate(('observed_once', 'refuted'), 1):
            evidence = make_card('EvidenceCard', wf.journal.run_id, {
                'experiment_ref': spec['id'], 'execution_fingerprint': 'CPU-fixture-' + str(revision),
                'status': 'completed', 'observations': {'fixture_value': revision}, 'provenance': {},
                'resources': {}, 'artifacts': [], 'measurement_source': 'synthetic_computation',
                'execution_id': 'CPU-fixture-' + str(revision), 'scope': scope}, refs=[spec['id']])
            wf.journal.commit('evidence/' + str(revision), cards=[evidence])
            task, _ = self.begin(wf, self.recorder(wf, 'record/' + str(revision)))
            value = self.record_payload(task); value['evidence_refs'] = [evidence['id']]
            update = {'claim_id': 'k', 'text': 'CPU protocol fixture only', 'status': status, 'scope': scope,
                      'evidence_refs': [evidence['id']], 'rationale': 'explicit synthetic fixture'}
            value['knowledge_updates'] = [update]
            if prior:
                self.assertIn(evidence['id'], wf.memory.recording_batch()['source_refs'])
            wf.submit(task, value, 'SCRIPTED')
            current = wf.result_card(wf.queue.get(task['task_id']))
            if prior:
                self.assertEqual(current['payload']['knowledge_updates'][0]['supersedes'], prior['id'])
                self.assertEqual(wf.journal.card(prior['id'])['payload']['knowledge_updates'][0]['status'], 'observed_once')
            prior = current
        claim = wf.memory.snapshot()['L2']['k']
        self.assertEqual((claim['status'], claim['version']), ('refuted', 2))
        self.assertEqual(len(wf.memory.snapshot()['L3']), 2)

    def test_opponent_cache_reuse_does_not_hide_previously_public_evidence(self):
        wf = self.workflow(1); cards = self.science(wf, 'public_cache')
        own, other = self.debate_task(wf), self.debate_task(wf, side='B')
        tid = wf.queue.enqueue(other['key'] + '/tools/cache', 'experiment_request', inputs={'parent_task': other['task_id']})
        wf.queue.ready(); wf.queue.claim(tid)
        wf.queue.finish(tid, outputs=[cards[1]['id']], result={'cache_reused': True})
        own, context = self.begin(wf, own)
        self.assertEqual(self.full_card(wf, own, cards[1]['id']), cards[1])
        self.assertIn(cards[1]['id'], self.index(wf, own))

    def test_malformed_recorder_responses_use_bounded_input_feedback(self):
        wf = self.workflow(1); self.science(wf, 'bad_response')
        task, _ = self.begin(wf, self.recorder(wf))
        for index, response in enumerate(([], {'decision': []}, {'decision': {'action': 'submit', 'content': []}})):
            receipt = {'status': 'completed', 'data': {'response': response}}
            with patch.object(wf.runtime, 'poll', return_value=[(task, receipt, 'SCRIPTED_BAD')]), patch.object(wf.runtime, 'settle'):
                wf.poll()
            current = wf.queue.get(task['task_id'])
            self.assertEqual(current['state'], 'retry' if index < 2 else 'failed')
            if index < 2: task, _ = self.begin(wf, current)
        self.assertEqual(current['repair_count'], 2)
        self.assertTrue(wf.memory.recording_batch()['source_refs'])

    def test_inherited_feedback_refs_do_not_reinject_full_metareview(self):
        wf = self.workflow(1)
        value = meta(1); value['scientific_analysis'] = 'LONG_HISTORICAL_REVIEW_' + 'x' * 18000
        card = make_card('MetareviewCard', wf.journal.run_id, value, author='recorder')
        wf.journal.commit('old_feedback', cards=[card], events=[wf.memory.apply_metareview(card)])
        for purpose in ('screen', 'interpret'):
            task = self.task(wf, purpose, refs=[wf.brief['id'], card['id']], experiment_draft=experiment_payload())
            context = wf.request_body(task)['spec']['context']
            self.assertNotIn(card['id'], {c['id'] for c in context['memory']['cards']})
            self.assertFalse(any(r['ref'] == card['id'] and r['field'] == '/payload' for r in context['memory']['required_reads']))
            self.assertNotIn(value['scientific_analysis'], canonical(context))
            self.assertIn(card['id'], self.index(wf, task))


if __name__ == '__main__': unittest.main()
