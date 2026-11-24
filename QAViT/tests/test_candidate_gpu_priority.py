"""Candidate fairness on real OS leases; hardware identity is simulated here."""
from pathlib import Path
import os
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import make_card
from runtime.gpu_resources import ResourceBusy, acquire, placement
from tests.test_adaptive_resources import devices
from tests.test_dual_task_evaluation import payload
from tests.test_infrastructure import TempCase
from tests.support import experiment_payload


class CandidateGpuPriority(TempCase):
    def candidates(self, count=3, plans_only=False, name='research'):
        from core.workflow import Workflow
        from tests.support import validation_config
        wf = Workflow(self.root / name, validation_config())
        plans, requests, trials = [], [], []
        for n in range(count):
            ref = f'candidate-H{n + 1}@1'
            context = {'round': 1, 'slot': f'H{n + 1}', 'hypothesis_ref': ref}
            plans.append(wf.queue.enqueue('plan/' + ref, 'llm', role='generation',
                inputs={'purpose': 'plan_l3', 'context': context}))
            if plans_only and n:
                continue
            requests.append(wf.queue.enqueue('request/' + ref, 'experiment_request',
                inputs={**context, 'draft': {'level': 'L3'}}))
            draft = experiment_payload('L3')
            draft.update(payload())
            draft['resources'].update(cpu_threads=1, estimated_seconds=60)
            spec = make_card('ExperimentCard', wf.journal.run_id, draft)
            wf.journal.commit('spec/' + ref, cards=[spec])
            trials.append(wf.queue.enqueue('trial/' + ref, 'experiment',
                inputs={'experiment_ref': spec['id'], 'parent_task': requests[-1]}))
        wf.queue.promote()
        return wf, plans, requests, trials

    def test_primary_gpu_before_dual_task_for_zero_to_six_cards(self):
        for count, expected in [(0, []), (1, [1]), (2, [1, 1]), (3, [1, 1, 1]),
                                (4, [2, 1, 1]), (5, [2, 2, 1]), (6, [2, 2, 2])]:
            with self.subTest(count=count):
                wf, _, _, trials = self.candidates(name='cards-' + str(count))
                wf.runtime.settings = {'allowed_gpus': list(range(count))}
                allocations = []
                def launch(task, body, **kwargs):
                    wf.queue.claim(task['task_id'])
                    allocations.append(body['gpus'])
                with patch('runtime.gpu_resources.gpu_bindings', return_value=devices(count)), \
                     patch('runtime.gpu_resources.externally_busy', return_value=set()), \
                     patch.object(wf.runtime, '_launch', side_effect=launch):
                    for tid in trials:
                        card = wf.journal.card(wf.queue.get(tid)['inputs']['experiment_ref'])
                        body = placement(card, wf.runtime.settings)
                        try:
                            wf.runtime.launch(wf.queue.get(tid), body, input_versions={})
                        except ResourceBusy:
                            self.assertEqual(wf.queue.get(tid)['attempt'], 0)
                    self.assertEqual([len(g) for g in allocations], expected)
                    flat = [g for allocation in allocations for g in allocation]
                    self.assertEqual(len(flat), len(set(flat)))
                    self.assertEqual(wf.budget.snapshot()['reservations'], {})

    def test_pending_serial_coding_reserves_other_candidates_and_failure_frees_it(self):
        wf, plans, requests, trials = self.candidates(plans_only=True)
        task = wf.queue.get(trials[0])
        self.assertEqual(wf.runtime.unserved_gpu_candidates(task), 2)
        wf.queue.claim(plans[1])
        wf.queue.fail(plans[1], {'code': 'test_failure'})
        self.assertEqual(wf.runtime.unserved_gpu_candidates(task), 1)
        wf.queue.claim(plans[2])
        wf.queue.finish(plans[2])
        self.assertEqual(wf.runtime.unserved_gpu_candidates(task), 1)  # request not yet enqueued
        tid = wf.queue.enqueue('request/last', 'experiment_request', inputs={
            'round': 1, 'hypothesis_ref': 'candidate-H3@1', 'draft': {'level': 'L3'}})
        wf.queue.promote(); wf.queue.claim(tid); wf.queue.fail(tid, {'code': 'test_failure'})
        self.assertEqual(wf.runtime.unserved_gpu_candidates(task), 0)

    def test_failure_and_restore_release_primary_without_changing_remaining_binding(self):
        from core.workflow import Workflow
        wf, _, requests, trials = self.candidates()
        wf.queue.claim(trials[0]); wf.queue.fail(trials[0], {'code': 'test_failure'})
        wf.queue.claim(requests[0]); wf.queue.fail(requests[0], {'code': 'service_failed'})
        # Actual completed plan snapshots do not retain reservations.
        for plan in wf.queue.snapshot(kinds={'llm'}).values():
            wf.queue.claim(plan['task_id']); wf.queue.finish(plan['task_id'])
        wf = Workflow(wf.run_dir)
        self.assertEqual(wf.runtime.unserved_gpu_candidates(wf.queue.get(trials[1])), 1)
        self.assertEqual(wf.queue.get(trials[1])['inputs']['parent_task'], requests[1])

    def test_legacy_requests_follow_original_plan_without_double_reservation(self):
        wf, plans, requests, trials = self.candidates()
        for plan, request in zip(plans, requests):
            # Historical final requests have origin_task, not the new explicit
            # hypothesis_ref/round fields; they still own exactly one candidate.
            wf.queue.change(request, 'ready', patch={'inputs': {
                'origin_task': plan, 'draft': {'level': 'L3'}}})
            wf.queue.claim(plan); wf.queue.finish(plan)
        self.assertEqual(wf.runtime.unserved_gpu_candidates(wf.queue.get(trials[0])), 2)

    def test_external_lease_and_busy_card_count_against_spares(self):
        rows = devices(5)
        # All IDs and UUIDs are synthetic; flock is the real Linux lease primitive.
        with patch('runtime.gpu_resources.externally_busy', return_value={rows[4]['uuid']}):
            _, held = acquire([3], bindings=[rows[3]])
            try:
                selected, fds = acquire(list(range(5)), 2, bindings=rows, reserve=2)
                try:
                    self.assertEqual(selected, [0])  # only 0,1,2 actually free
                finally:
                    for fd in fds: os.close(fd)
                # Probe locks on reserved cards must be released immediately.
                selected, fds = acquire([1, 2], 2, bindings=rows)
                try:
                    self.assertEqual(selected, [1, 2])
                finally:
                    for fd in fds: os.close(fd)
            finally:
                for fd in held: os.close(fd)
