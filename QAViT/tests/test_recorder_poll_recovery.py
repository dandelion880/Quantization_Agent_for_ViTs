"""真实回执文件/poll 路径的 CPU 故障注入；不调用模型。"""
from pathlib import Path
import time
from unittest.mock import patch

from tests.test_infrastructure import TempCase
from tests.support import hypothesis, meta
from core.cards import make_card
from core.journal import atomic_json, file_sha256
from core.workflow import Workflow
from runtime.identity import code_identity


class RecorderPollRecovery(TempCase):
    def test_persisted_receipt_survives_precommit_failure_and_restart(self):
        wf = self.workflow(1)
        card = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1, 'H1'))
        wf.journal.commit('fixture/science', cards=[card])
        tid = wf.enqueue_recorder(1, incomplete=True)
        wf.queue.ready()
        task = wf.queue.get(tid)
        execution = tid + '-cpu-receipt'
        directory = wf.run_dir / 'jobs' / execution
        path = directory / 'request.json'
        request = dict(wf.request_body(task), task_id=tid, execution_id=execution,
                       core_sha256=code_identity()['core_sha256'], deadline_unix=time.time() + 120)
        atomic_json(path, request)
        wf.budget.reserve(execution, 'llm', estimated_seconds=120)
        wf.queue.claim(tid, request_path=str(path))
        batch = task['inputs']['context']['recording_batch']
        content = meta(1, batch['judgment_refs'])
        receipt = {'task_id': tid, 'execution_id': execution, 'core_sha256': request['core_sha256'],
                   'request_sha256': file_sha256(path), 'status': 'completed', 'resources': {'wall_seconds': 0},
                   'data': {'response': {'source': 'test_double', 'decision': {'action': 'submit', 'content': content}}}}
        atomic_json(directory / 'result.json', receipt)
        original = wf.journal.commit
        def crash(key, **kwargs):
            if key.startswith('complete/'):
                def fault(stage):
                    if stage == 'cards_written': raise OSError('CPU injected precommit interruption')
                kwargs['fault'] = fault
            return original(key, **kwargs)
        with patch.object(wf.journal, 'commit', side_effect=crash), self.assertRaises(OSError):
            wf.poll()
        self.assertFalse(any(e['type'] == 'recorder_batch_completed' for e in wf.journal.events))
        self.assertEqual(wf.queue.get(tid)['state'], 'running')
        self.assertEqual(wf.memory.recording_batch()['source_refs'], batch['source_refs'])
        wf = Workflow(wf.run_dir)
        self.assertEqual(wf.poll(), 1)
        self.assertEqual(wf.queue.get(tid)['state'], 'completed')
        self.assertEqual(wf.memory.recording_batch()['source_refs'], [])
        self.assertEqual(wf.memory.snapshot()['L2'], {})
        self.assertEqual(wf.poll(), 0)
        self.assertEqual(len(wf.journal.find_cards('MetareviewCard')), 1)
        self.assertEqual(sum(e['type'] == 'recorder_batch_completed' for e in wf.journal.events), 1)
