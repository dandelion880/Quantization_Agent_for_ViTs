"""All final-output stages use the same bounded worker/poll repair, CPU doubles only."""
import json
from unittest.mock import patch

from tests import test_inner_json_repair as inner
from tests.test_infrastructure import TempCase
from tests.support import hypothesis, meta
from core.cards import make_card
from core.journal import read_json
from runtime.subscription import StructuredOutputError, InnerJSONFormatError, DECISION
from ptq_agent.schemas import LLMRequest, PTQError


def final_for(name, arguments):
    return json.dumps({'name': name, 'arguments_json': arguments}, ensure_ascii=False)


BAD_FINALS = [
    ('outer_json', '{"name":', '$'),
    ('outer_schema', json.dumps({'name': DECISION, 'arguments_json': {}}), '$.arguments_json'),
    ('inner_json', inner.envelope(inner.BAD_INNER), '$.arguments_json'),
    ('inner_object', inner.envelope('[]'), '$.arguments_json'),
    ('decision_schema', inner.envelope('{"action":"submit","content":[]}'), '$.arguments_json.content'),
    ('tool_schema', final_for('read_card', '{"ref":7}'), '$.arguments_json.ref'),
]


class StructuredOutputRepair(TempCase):
    start = inner.InnerJSONRepairChain.start
    valid_decision = staticmethod(inner.InnerJSONRepairChain.valid_decision)

    def attempt(self, wf, tid, final, **kwargs):
        # Reuse the existing actual worker/model_router/receipt fixture; only CLI final is scripted.
        with patch('tests.test_inner_json_repair.envelope', return_value=final):
            return inner.InnerJSONRepairChain.attempt(self, wf, tid, 'unused', **kwargs)

    def isolated(self, name):
        self.root = self.root / name
        return self.start()

    def test_every_known_stage_returns_exact_same_task_feedback_then_submits(self):
        for stage, final, location in BAD_FINALS:
            with self.subTest(stage=stage):
                wf, tid = self.isolated(stage)
                receipt, directory, _ = self.attempt(wf, tid, final)
                self.assertEqual(receipt['status'], 'failed')
                error = receipt['error']
                self.assertEqual(error['stage'], stage)
                self.assertEqual(error['location']['path'], location)
                self.assertEqual(error['raw_output'], final)
                self.assertTrue(error['response_integrity_verified'])
                self.assertEqual(receipt['data'], {'usage': inner.USAGE})
                self.assertEqual(read_json(directory / 'llm_audit.json')['usage'], inner.USAGE)
                self.assertEqual(wf.poll(), 1)
                current = wf.queue.get(tid)
                self.assertEqual((current['state'], current['repair_count'], current['retry_count']), ('retry', 1, 0))
                feedback = wf.request_body(current)['spec']['context']['repair_feedback']
                self.assertEqual(feedback['task_id'], tid)
                self.assertEqual(feedback['execution_id'], receipt['execution_id'])
                self.assertEqual(feedback['format_error'], error)
                self.assertEqual(feedback['rejected_response'], {})
                settled = [e for e in wf.journal.events if e['type'] == 'budget_settled']
                self.assertEqual(settled[-1]['reported_usage'], inner.USAGE)
                self.attempt(wf, tid, inner.envelope(json.dumps(self.valid_decision())))
                wf.poll()
                self.assertEqual(wf.queue.get(tid)['state'], 'completed')
                self.assertEqual(wf.queue.get(tid)['attempt'], 2)
                self.assertEqual(len(wf.journal.find_cards('ReviewDecisionCard')), 1)
                self.assertEqual(wf.poll(), 0)

    def test_mixed_stages_exhaust_the_shared_allowance_without_transport_retries(self):
        wf, tid = self.start()
        for stage in ('outer_json', 'inner_object', 'tool_schema'):
            final = next(row[1] for row in BAD_FINALS if row[0] == stage)
            self.attempt(wf, tid, final)
            wf.poll()
        task = wf.queue.get(tid)
        self.assertEqual((task['state'], task['repair_count'], task['retry_count'], task['attempt']), ('failed', 2, 0, 3))
        self.assertEqual(len(wf.budget.snapshot()['reservations']), 3)
        self.assertEqual(wf.journal.find_cards('ReviewDecisionCard'), [])

    def test_schema_repair_still_rejects_scientific_contract_and_shares_its_allowance(self):
        wf, tid = self.start()
        self.attempt(wf, tid, BAD_FINALS[1][1]); wf.poll()
        invalid = self.valid_decision()
        invalid['content'].update(eligible=False, hard_rejection_reasons=[])
        receipt, _, _ = self.attempt(wf, tid, inner.envelope(json.dumps(invalid)))
        self.assertEqual(receipt['status'], 'completed')
        wf.poll()
        task = wf.queue.get(tid)
        self.assertEqual(task['repair_count'], 2)
        self.assertIn('拒绝必须有非空理由', task['repair_feedback']['error'])
        self.assertEqual(wf.journal.find_cards('ReviewDecisionCard'), [])
        self.attempt(wf, tid, BAD_FINALS[0][1]); wf.poll()
        self.assertEqual(wf.queue.get(tid)['state'], 'failed')
        self.assertEqual(wf.queue.get(tid)['retry_count'], 0)

    def test_recorder_draft_survives_format_failure_and_corrected_patch_is_merged(self):
        wf = self.workflow(1)
        card = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1, 'H1'))
        wf.journal.commit('fixture/record', cards=[card])
        tid = wf.enqueue_recorder(1, incomplete=True)
        batch = wf.queue.get(tid)['inputs']['context']['recording_batch']
        content = meta(1, batch['judgment_refs'])
        content['scientific_analysis'] = ''
        first, first_directory, _ = self.attempt(wf, tid, inner.envelope(json.dumps({'action': 'submit', 'content': content})))
        wf.poll()
        self.assertEqual(wf.rejected_draft(wf.queue.get(tid)), content)
        malformed_patch = inner.envelope('{"action":"submit","content":{"repair_patch":')
        second, _, _ = self.attempt(wf, tid, malformed_patch)
        wf.poll()
        task = wf.queue.get(tid)
        self.assertEqual(task['repair_count'], 2)
        self.assertEqual(wf.rejected_draft(task), content)
        feedback = wf.request_body(task)['spec']['context']['repair_feedback']
        self.assertEqual(feedback['format_error']['raw_output'], malformed_patch)
        self.assertEqual(feedback['rejected_response']['decision']['content'], content)
        self.assertEqual(feedback['rejected_response_receipt'], str(first_directory / 'result.json'))
        self.assertEqual(feedback['execution_id'], second['execution_id'])
        self.assertNotEqual(feedback['execution_id'], first['execution_id'])
        replacement = 'CPU fixture: reviewed without a new scientific claim'
        valid_patch = {'action': 'submit', 'content': {'repair_patch': {'scientific_analysis': replacement}}}
        self.attempt(wf, tid, inner.envelope(json.dumps(valid_patch))); wf.poll()
        task = wf.queue.get(tid)
        self.assertEqual(task['state'], 'completed')
        published = wf.journal.find_cards('MetareviewCard')
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0]['payload'], wf.memory.prepare_metareview(
            {**content, 'scientific_analysis': replacement}, batch, 1))
        self.assertEqual(sum(e['type'] == 'recorder_batch_completed' for e in wf.journal.events), 1)
        self.assertEqual(wf.memory.recording_batch()['source_refs'], [])
        self.assertEqual(wf.poll(), 0)

    def test_recorder_cannot_use_another_tasks_rejected_draft(self):
        wf = self.workflow(1)
        card = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1, 'H1'))
        wf.journal.commit('fixture/record', cards=[card])
        first = wf.enqueue_recorder(1, incomplete=True)
        batch = wf.queue.get(first)['inputs']['context']['recording_batch']
        content = meta(1, batch['judgment_refs'])
        content['scientific_analysis'] = ''
        self.attempt(wf, first, inner.envelope(json.dumps({'action': 'submit', 'content': content})))
        wf.poll()
        wf.queue.change(first, 'cancelled')
        second = wf.enqueue_recorder(1, incomplete=True, key='fixture/independent_record')
        self.attempt(wf, second, BAD_FINALS[0][1]); wf.poll()
        self.assertIsNone(wf.rejected_draft(wf.queue.get(second)))
        patch_content = {'action': 'submit', 'content': {'repair_patch': {'scientific_analysis': 'CPU replacement'}}}
        self.attempt(wf, second, inner.envelope(json.dumps(patch_content))); wf.poll()
        self.assertEqual(wf.queue.get(second)['state'], 'retry')
        self.assertIn('完整提交', wf.queue.get(second)['repair_feedback']['error'])
        self.assertEqual(wf.journal.find_cards('MetareviewCard'), [])

    def test_explicit_other_task_owner_cannot_merge_but_legacy_embedded_draft_can(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                self.root = self.root / ('legacy' if legacy else 'wrong-owner')
                wf = self.workflow(1)
                card = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1, 'H1'))
                wf.journal.commit('fixture/record', cards=[card])
                tid = wf.enqueue_recorder(1, incomplete=True)
                batch = wf.queue.get(tid)['inputs']['context']['recording_batch']
                content = meta(1, batch['judgment_refs'])
                content['scientific_analysis'] = ''
                self.attempt(wf, tid, inner.envelope(json.dumps({'action': 'submit', 'content': content})))
                wf.poll()
                feedback = wf.queue.get(tid)['repair_feedback']
                if legacy:
                    feedback.pop('task_id')  # Historical feedback was embedded only in this task.
                else:
                    other = wf.enqueue_role('fixture/other', 'generate', context={'round': 1, 'slot': 'H2'})
                    feedback['task_id'] = other  # Explicit adversarial owner mismatch in durable state.
                    wf.queue.change(other, 'cancelled')
                wf.queue.change(tid, 'retry', patch={'repair_feedback': feedback})
                correction = {'action': 'submit', 'content': {'repair_patch': {'scientific_analysis': 'CPU corrected analysis'}}}
                self.attempt(wf, tid, inner.envelope(json.dumps(correction))); wf.poll()
                if legacy:
                    self.assertEqual(wf.queue.get(tid)['state'], 'completed')
                    self.assertEqual(wf.journal.find_cards('MetareviewCard')[0]['payload'],
                                     wf.memory.prepare_metareview(
                                         {**content, 'scientific_analysis': 'CPU corrected analysis'}, batch, 1))
                else:
                    self.assertEqual(wf.queue.get(tid)['state'], 'retry')
                    self.assertIn('完整提交', wf.queue.get(tid)['repair_feedback']['error'])
                    self.assertEqual(wf.journal.find_cards('MetareviewCard'), [])

    def test_external_schema_error_and_unknown_exceptions_are_not_reclassified(self):
        for index, error in enumerate((PTQError('schema_error', 'outside final validation'),
                                       ValueError('unexpected program failure'), RuntimeError('unexpected program failure'))):
            with self.subTest(error=type(error).__name__):
                wf, tid = self.isolated('unknown-' + str(index))
                receipt, _, _ = self.attempt(wf, tid, BAD_FINALS[0][1], decoder_error=error)
                self.assertEqual(receipt['error']['code'], getattr(error, 'code', type(error).__name__))
                wf.poll()
                self.assertEqual(wf.queue.get(tid)['state'], 'failed')
                self.assertEqual(wf.queue.get(tid).get('repair_count', 0), 0)

    def test_auth_quota_identity_and_missing_usage_keep_their_boundaries(self):
        for index, (stderr, expected) in enumerate((('401 unauthorized authentication failed', 'llm_authentication'),
                                                  ('quota exceeded', 'llm_quota_exhausted'))):
            with self.subTest(expected=expected):
                wf, tid = self.isolated('service-' + str(index))
                receipt, directory, _ = self.attempt(wf, tid, BAD_FINALS[0][1], stderr=stderr)
                self.assertEqual(receipt['error']['code'], expected)
                self.assertEqual(read_json(directory / 'llm_audit.json')['usage'], inner.USAGE)
                wf.poll()
                self.assertEqual(wf.queue.get(tid)['state'], 'failed')
                self.assertEqual(wf.transport_blocked['code'], expected)
        wf, tid = self.isolated('identity')
        receipt, _, _ = self.attempt(wf, tid, BAD_FINALS[0][1], identity_change=True)
        self.assertEqual(receipt['error']['code'], 'ContractError')
        self.assertEqual(receipt['data']['usage'], inner.USAGE)
        wf.poll()
        self.assertEqual(wf.queue.get(tid)['state'], 'failed')
        wf, tid = self.isolated('no-usage')
        receipt, _, _ = self.attempt(wf, tid, BAD_FINALS[0][1], usage=None)
        self.assertNotIn('data', receipt)
        wf.poll()
        self.assertEqual(wf.queue.get(tid)['repair_count'], 1)
        self.assertIsNone([e for e in wf.journal.events if e['type'] == 'budget_settled'][-1]['reported_usage'])


class StructuredDecoderBoundary(TempCase):
    def request(self):
        return LLMRequest(role='judger', system_prompt='CPU fixture', messages=[], context={}, tools=[],
                          response_schema={'type': 'object', 'properties': {'text': {'type': 'string'}},
                                           'required': ['text'], 'additionalProperties': False})

    def test_invalid_keys_duplicates_nonfinite_and_unicode_control_remain_strict(self):
        cases = [('{name:1}', 'outer_json'), ('{"name":1,"name":2}', 'outer_json'),
                 ('{"a":NaN}', 'outer_json'), ('{"a":1e999}', 'outer_json'),
                 (json.dumps({'name': DECISION, 'arguments_json': '{}', 'extra': 1}), 'outer_schema'),
                 (inner.envelope('{"text":"a","text":"b"}'), 'inner_json'),
                 (inner.envelope('{"text":Infinity}'), 'inner_json'),
                 (inner.envelope('{"text":1e999}'), 'inner_json'),
                 (inner.envelope('{"text":"literal\x00control"}'), 'inner_json')]
        client = inner.transport()
        for final, stage in cases:
            with self.subTest(stage=stage, final=final), self.assertRaises(StructuredOutputError) as caught:
                client.decode(inner.event_text(inner.events(final)), final, self.request())
            self.assertEqual(caught.exception.stage, stage)
            self.assertEqual(caught.exception.raw_output, final)
            if stage == 'inner_json':
                self.assertIsInstance(caught.exception, InnerJSONFormatError)

    def test_legal_long_escaped_text_is_unchanged(self):
        text = 'line\nquote"backslash\\中文\t' * 7000
        final = inner.envelope(json.dumps({'text': text}, ensure_ascii=True))
        result, _ = inner.transport().decode(inner.event_text(inner.events(final)), final, self.request())
        self.assertEqual(result.decision['text'], text)

    def test_cli_damage_isolation_identity_and_stdout_equality_precede_final_decode(self):
        final = '{"name":'
        cases = [
            (inner.event_text(inner.events(final)[1:]), final, 'llm_tool_isolation'),
            (inner.event_text(inner.events(final)[:-1]), final, 'llm_format_error'),
            (inner.event_text(inner.events(final)), final + 'different', 'llm_format_error'),
            (inner.event_text([*inner.events(final), {'type': 'item.completed', 'item': {'type': 'command_execution'}}]), final, 'llm_tool_isolation'),
            (inner.event_text([*inner.events(final), {'type': 'turn.started', 'model': 'wrong'}]), final, 'llm_model_mismatch'),
            ('{"type":', final, None),
        ]
        for events, answer, code in cases:
            with self.subTest(code=code), self.assertRaises(Exception) as caught:
                inner.transport().decode(events, answer, self.request())
            self.assertNotIsInstance(caught.exception, StructuredOutputError)
            if code:
                self.assertEqual(caught.exception.code, code)
            else:
                self.assertIsInstance(caught.exception, json.JSONDecodeError)

    def test_only_local_known_validation_errors_are_repairable(self):
        request = self.request()
        final = inner.envelope('{"text":"valid"}')
        for exc in (ValueError('unexpected schema program error'), PTQError('llm_protocol_error', 'bad host schema')):
            with self.subTest(error=type(exc).__name__), patch('runtime.subscription.validate', side_effect=exc), self.assertRaises(type(exc)) as caught:
                inner.transport().decode(inner.event_text(inner.events(final)), final, request)
            self.assertNotIsInstance(caught.exception, StructuredOutputError)
        request.tools = [{'name': DECISION, 'parameters': {}}]
        with self.assertRaises(PTQError) as caught:
            inner.transport().decode(inner.event_text(inner.events(final)), final, request)
        self.assertEqual(caught.exception.code, 'llm_protocol_error')
