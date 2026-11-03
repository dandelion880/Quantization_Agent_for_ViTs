"""CPU-only strict decoder → worker receipt → poll → same-task repair regression."""
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

from tests.test_infrastructure import TempCase
from tests.support import hypothesis, review
from core.cards import ContractError, make_card
from core.journal import atomic_json, read_json
from runtime import codex_cli, task_worker
from runtime.identity import code_identity
from runtime.subscription import DECISION, InnerJSONFormatError, SubscriptionTransport, strict_json
from ptq_agent.schemas import LLMRequest, PTQError


USAGE = {'input_tokens': 101, 'output_tokens': 17, 'cached_input_tokens': 23}
BAD_INNER = '{"action":"submit","content":{"rationale":"literal\ncontrol"}}'


def transport():
    client = SubscriptionTransport.__new__(SubscriptionTransport)
    client.model, client.effort = 'cpu-scripted-model', 'medium'
    client.source = 'test_double'
    return client


def envelope(inner):
    return json.dumps({'name': DECISION, 'arguments_json': inner}, ensure_ascii=False)


def events(final, usage=USAGE):
    return [
        {'type': 'item.completed', 'item': {'type': 'error', 'message': codex_cli.ISOLATION_NOTICE}},
        {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': final}},
        {'type': 'turn.completed', 'usage': usage},
    ]


def event_text(rows):
    return '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows)


class DecoderBoundary(unittest.TestCase):
    def setUp(self):
        self.client = transport()
        self.request = LLMRequest(role='judger', system_prompt='CPU test double', messages=[], tools=[],
                                  context={}, response_schema={'type': 'object'})

    def test_control_character_preserves_raw_location_and_available_usage(self):
        final = envelope(BAD_INNER)
        with self.assertRaises(InnerJSONFormatError) as caught:
            self.client.decode(event_text(events(final)), final, self.request)
        value = caught.exception.to_dict()
        self.assertEqual(value['code'], 'llm_inner_json_error')
        self.assertEqual(value['raw_output'], final)
        self.assertEqual(json.loads(value['raw_output'])['arguments_json'], BAD_INNER)
        self.assertEqual(value['location'], {'path': '$.arguments_json', 'line': 1,
                                            'column': BAD_INNER.index('\n') + 1, 'offset': BAD_INNER.index('\n')})
        self.assertEqual(caught.exception.usage, USAGE)
        self.assertTrue(value['response_integrity_verified'])

    def test_no_lax_decode_or_silent_strict_json_repair(self):
        for inner in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', '{"a":1'):
            with self.subTest(inner=inner), self.assertRaises(InnerJSONFormatError) as caught:
                final = envelope(inner)
                self.client.decode(event_text(events(final)), final, self.request)
            self.assertEqual(caught.exception.raw_output, final)

    def test_integrity_identity_and_isolation_gates_precede_inner_repair(self):
        final = envelope(BAD_INNER)
        cases = [
            (events(final)[1:], final, 'llm_tool_isolation'),
            (events(final)[:-1], final, 'llm_format_error'),
            (events(final), final + ' ', None),  # existing whitespace equivalence is allowed
            (events(final), envelope('{}'), 'llm_format_error'),
            ([*events(final), {'type': 'item.completed', 'item': {'type': 'command_execution'}}], final, 'llm_tool_isolation'),
            ([*events(final), {'type': 'turn.started', 'model': 'wrong'}], final, 'llm_model_mismatch'),
            ([*events(final), {'type': 'turn.started', 'reasoning_effort': 'low'}], final, 'llm_model_mismatch'),
        ]
        for rows, answer, code in cases:
            with self.subTest(code=code), self.assertRaises(PTQError) as caught:
                self.client.decode(event_text(rows), answer, self.request)
            if code:
                self.assertEqual(caught.exception.code, code)
                self.assertNotIsInstance(caught.exception, InnerJSONFormatError)
            else:
                self.assertIsInstance(caught.exception, InnerJSONFormatError)

    def test_unknown_program_error_is_not_reclassified_as_format_error(self):
        final = envelope(BAD_INNER)
        def fail_inner(text):
            if text == BAD_INNER:
                raise RuntimeError('CPU unexpected decoder bug')
            return strict_json(text)
        with patch('runtime.subscription.strict_json', side_effect=fail_inner), self.assertRaisesRegex(RuntimeError, 'unexpected'):
            self.client.decode(event_text(events(final)), final, self.request)


class InnerJSONRepairChain(TempCase):
    def start(self):
        wf = self.workflow(1)
        card = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1, 'H1'))
        wf.journal.commit('fixture/hypothesis', cards=[card])
        tid = wf.enqueue_role('fixture/screen', 'screen', refs=[card['id']],
                              context={'round': 1, 'slot': 'H1', 'hypothesis_ref': card['id']})
        return wf, tid

    @staticmethod
    def valid_decision():
        return {'action': 'submit', 'content': review('screening', eligible=True,
                hard_rejection_reasons=[], similarity_analysis='CPU test double', evidence_gaps=[])}

    def attempt(self, wf, tid, inner, *, usage=USAGE, stderr='', decoder_error=None, identity_change=False):
        wf.queue.ready()
        task = wf.queue.get(tid)
        execution = tid + '-cpu-' + str(task['attempt'] + 1)
        directory = wf.run_dir / 'jobs' / execution
        path = directory / 'request.json'
        identity = code_identity()
        body = dict(wf.request_body(task), task_id=tid, execution_id=execution,
                    deadline_unix=time.time() + 60, core_sha256=identity['core_sha256'])
        atomic_json(path, body)
        wf.budget.reserve(execution, 'llm', estimated_seconds=60)
        wf.queue.claim(tid, request_path=str(path))
        final = envelope(inner)
        client = transport()
        client.preflight = lambda timeout: {'source': 'test_double'}
        client.command = lambda directory: []
        if decoder_error:
            client.decode = lambda *args: (_ for _ in ()).throw(decoder_error)
        def run(command, working, timeout, stdin):
            (working / 'answer.json').write_text(final, encoding='utf-8')
            return 0, event_text(events(final, usage)), stderr
        identities = [identity, {**identity, 'core_sha256': 'changed'}] if identity_change else [identity, identity]
        with patch('runtime.model_router.client_for', return_value=client), patch('runtime.model_router.cli.run', side_effect=run), \
             patch.object(sys, 'argv', ['task_worker.py', str(path)]), \
             patch('runtime.task_worker.code_identity', side_effect=identities), redirect_stderr(StringIO()):
            task_worker.main()
        receipt = read_json(directory / 'result.json')
        self.assertEqual(receipt['task_id'], tid)
        self.assertEqual(receipt['execution_id'], execution)
        return receipt, directory, final

    def test_actual_worker_receipt_poll_repairs_same_task_and_accounts_failure(self):
        wf, tid = self.start()
        receipt, directory, final = self.attempt(wf, tid, BAD_INNER)
        self.assertEqual(receipt['status'], 'failed')
        self.assertNotIn('response', receipt['data'])
        self.assertEqual(read_json(directory / 'llm_audit.json')['usage'], USAGE)
        self.assertEqual(receipt['data']['usage'], USAGE)
        self.assertEqual(wf.poll(), 1)
        current = wf.queue.get(tid)
        self.assertEqual((current['state'], current['repair_count'], current['retry_count']), ('retry', 1, 0))
        feedback = wf.request_body(current)['spec']['context']['repair_feedback']
        self.assertEqual(feedback['task_id'], tid)
        self.assertEqual(feedback['execution_id'], receipt['execution_id'])
        self.assertEqual(feedback['format_error']['raw_output'], final)
        self.assertEqual(feedback['format_error']['location']['offset'], BAD_INNER.index('\n'))
        settled = [e for e in wf.journal.events if e['type'] == 'budget_settled']
        self.assertEqual(settled[-1]['reported_usage'], USAGE)
        self.attempt(wf, tid, json.dumps(self.valid_decision()))
        self.assertEqual(wf.poll(), 1)
        self.assertEqual(wf.queue.get(tid)['state'], 'completed')
        self.assertEqual(wf.queue.get(tid)['attempt'], 2)
        self.assertEqual(len(wf.journal.find_cards('ReviewDecisionCard')), 1)
        self.assertEqual(wf.poll(), 0)

    def test_format_and_scientific_contract_failures_share_two_repairs(self):
        wf, tid = self.start()
        self.attempt(wf, tid, BAD_INNER)
        wf.poll()
        invalid = self.valid_decision()
        invalid['content'].update(eligible=False, hard_rejection_reasons=[])
        receipt, _, _ = self.attempt(wf, tid, json.dumps(invalid))
        self.assertEqual(receipt['status'], 'completed')  # decoding does not validate the scientific contract
        wf.poll()
        current = wf.queue.get(tid)
        self.assertEqual((current['state'], current['repair_count']), ('retry', 2))
        self.assertIn('拒绝必须有非空理由', current['repair_feedback']['error'])
        self.assertEqual(wf.journal.find_cards('ReviewDecisionCard'), [])
        self.attempt(wf, tid, json.dumps(self.valid_decision()))
        wf.poll()
        self.assertEqual(wf.queue.get(tid)['state'], 'completed')
        self.assertEqual(len(wf.budget.snapshot()['reservations']), 3)

    def test_format_budget_exhaustion_is_terminal_without_transport_retries(self):
        wf, tid = self.start()
        for _ in range(3):
            self.attempt(wf, tid, BAD_INNER)
            wf.poll()
        task = wf.queue.get(tid)
        self.assertEqual((task['state'], task['repair_count'], task['retry_count'], task['attempt']), ('failed', 2, 0, 3))
        self.assertEqual(task['error']['code'], 'contract_error')
        self.assertEqual(len(wf.budget.snapshot()['reservations']), 3)
        self.assertEqual(sum(e['type'] == 'budget_settled' for e in wf.journal.events), 3)
        self.assertEqual(wf.poll(), 0)

    def test_identity_change_and_unknown_errors_do_not_enter_format_repair(self):
        for option, expected in (({'identity_change': True}, 'ContractError'),
                                 ({'decoder_error': RuntimeError('unknown bug')}, 'RuntimeError')):
            with self.subTest(expected=expected):
                wf, tid = self.start()
                receipt, _, _ = self.attempt(wf, tid, BAD_INNER, **option)
                self.assertEqual(receipt['error']['code'], expected)
                wf.poll()
                task = wf.queue.get(tid)
                self.assertEqual(task['state'], 'failed')
                self.assertEqual(task.get('repair_count', 0), 0)
                # A separate temporary workflow for the next subtest.
                self.root = self.root / expected

    def test_missing_usage_remains_unknown_and_authentication_remains_blocked(self):
        wf, tid = self.start()
        receipt, _, _ = self.attempt(wf, tid, BAD_INNER, usage=None)
        self.assertNotIn('data', receipt)
        wf.poll()
        settlement = [e for e in wf.journal.events if e['type'] == 'budget_settled'][-1]
        self.assertIsNone(settlement['reported_usage'])
        receipt, _, _ = self.attempt(wf, tid, BAD_INNER, stderr='401 unauthorized authentication failed')
        self.assertEqual(receipt['error']['code'], 'llm_authentication')
        wf.poll()
        self.assertEqual(wf.queue.get(tid)['state'], 'failed')
        self.assertEqual(wf.transport_blocked['code'], 'llm_authentication')


if __name__ == '__main__':
    unittest.main(verbosity=2)
