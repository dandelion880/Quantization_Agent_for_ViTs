"""Host task ancestry and asset ownership isolation; CPU only."""
from pathlib import Path
from copy import deepcopy
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, make_card
from core.journal import atomic_json, file_sha256
from tests import test_memory_delivery as delivery
from tests.test_infrastructure import TempCase
from tests.support import review


class BranchACLRegression(TempCase):
    science = delivery.DeliveryRegression.science
    debate_task = delivery.DeliveryRegression.debate_task
    read = delivery.DeliveryRegression.read
    full_card = delivery.DeliveryRegression.full_card
    index = delivery.DeliveryRegression.index
    begin = delivery.DeliveryRegression.begin
    task = delivery.DeliveryRegression.task

    def setUp(self):
        super().setUp()
        worker = patch('runtime.task_runtime.TaskRuntime.launch', side_effect=AssertionError('CPU only'))
        worker.start(); self.addCleanup(worker.stop)

    def test_context_reuses_shared_ancestry_but_new_context_detects_tampering(self):
        wf=self.workflow(1)
        cards=self.science(wf,'shared_proof')
        descendants=[]
        for i in range(12):
            card=make_card('ReviewDecisionCard',wf.journal.run_id,review('interpretation',rationale=str(i)),
                           refs=[cards[1]['id'],*[v['id'] for v in descendants[-3:]]])
            wf.journal.commit('shared-proof/'+str(i),cards=[card]);descendants.append(card)
        task=self.debate_task(wf)
        original=deepcopy(task)
        with patch.object(wf.journal,'card',wraps=wf.journal.card) as reads:
            cached=wf.context(task)
            cached_count=reads.call_count
        original_acl=wf.can_read_card
        def uncached(task,card):
            task.pop('_acl_allowed_refs',None)
            return original_acl(task,card)
        with patch.object(wf,'can_read_card',side_effect=uncached),patch.object(wf.journal,'card',wraps=wf.journal.card) as reads:
            baseline=wf.context(task)
            baseline_count=reads.call_count
        cached.pop('available_seconds');baseline.pop('available_seconds')
        cached.pop('budget_receipt');baseline.pop('budget_receipt')
        self.assertEqual(cached,baseline)
        self.assertLess(cached_count,baseline_count)
        self.assertEqual(task,original)
        broken=deepcopy(cards[1]);broken['payload']['observations']['tampered']=True
        atomic_json(wf.journal.card_path(broken['id']),broken)
        with self.assertRaises(ContractError):wf.context(task)
        with self.assertRaises(ContractError):wf.authorized_assets(task)

    def test_asset_authorization_reuse_keeps_private_branch_denied(self):
        wf=self.workflow(1);self.science(wf,'public_proof')
        own,opposite=self.debate_task(wf),self.debate_task(wf,side='B')
        secret=private_result(self,wf,opposite)
        original=deepcopy(own)
        allowed=wf.authorized_assets(own)
        original_acl=wf.can_read_card
        def uncached(task,card):
            task.pop('_acl_allowed_refs',None)
            return original_acl(task,card)
        with patch.object(wf,'can_read_card',side_effect=uncached):
            baseline=wf.authorized_assets(own)
        self.assertEqual(allowed,baseline)
        self.assertNotIn(str(wf.journal.card_path(secret[1]['id'])),allowed)
        self.assertNotIn(secret[1]['payload']['artifacts'][0]['path'],allowed)
        self.assertEqual(own,original)


def private_result(case, wf, parent):
    tid = wf.queue.enqueue('unrelated-job-key', 'experiment', inputs={'parent_task': parent['task_id']})
    wf.queue.ready(); wf.queue.claim(tid)
    cards = case.science(wf, 'PRIVATE_BRANCH', large=True, commit=False)
    wf.queue.finish(tid, cards=cards, outputs=[c['id'] for c in cards])
    return cards


def test_child_interpret_inherits_parent_position(case):
    wf = case.workflow(1)
    public = case.science(wf, 'public')
    own, opposite = case.debate_task(wf), case.debate_task(wf, side='B')
    secret = private_result(case, wf, opposite)
    rid = wf.queue.enqueue('unrelated-request-key', 'experiment_request', inputs={
        'parent_task': own['task_id'], 'requester': 'judger', 'refs': [wf.brief['id']],
        'draft': public[0]['payload']})
    wf.queue.ready(); wf.queue.claim(rid)
    wf.queue.change(rid, 'running', patch={'experiment_ref': public[0]['id']})
    request = wf.queue.get(rid)
    with patch.object(wf.broker, 'freeze', return_value=public[0]), patch.object(wf.broker, 'cached', return_value=public[1]):
        wf.progress_request(request)
    child = wf.tasks()['unrelated-request-key/interpret']
    case.assertEqual(child['inputs'].get('parent_task'), rid)
    child, context = case.begin(wf, child)
    case.assertNotIn(secret[1]['id'], case.index(wf, child))
    for ref in (secret[1]['id'], secret[3]['id']):
        with case.assertRaises(ContractError): case.read(wf, child, 'read_card', ref=ref)
    with case.assertRaises(ContractError): case.read(wf, child, 'read_asset', asset=secret[1]['payload']['artifacts'][0]['path'])
    case.assertEqual(case.full_card(wf, child, public[1]['id']), public[1])


def test_visible_path_mentions_do_not_grant_private_asset_or_card(case):
    wf = case.workflow(1)
    own, opposite = case.debate_task(wf), case.debate_task(wf, side='B')
    secret = private_result(case, wf, opposite)
    paths = [secret[1]['payload']['artifacts'][0]['path'], str(wf.journal.card_path(secret[1]['id']))]
    leak = make_card('ReviewDecisionCard', wf.journal.run_id, review('interpretation',
        rationale='public text references an unavailable asset', artifacts=[{'path': p} for p in paths]))
    wf.journal.commit('visible-path-mention', cards=[leak])
    own, _ = case.begin(wf, own)
    case.assertEqual(case.full_card(wf, own, leak['id']), leak)
    for path in paths:
        with case.assertRaises(ContractError): case.read(wf, own, 'read_asset', asset=path)


def test_unowned_asset_not_authorized_by_scientific_card(case):
    wf = case.workflow(1)
    path = wf.run_dir/'unowned.txt';path.write_text('UNOWNED PRIVATE FILE')
    leak = make_card('ReviewDecisionCard', wf.journal.run_id, review('interpretation',
        rationale='arbitrary location', artifacts=[{'path': str(path), 'sha256': file_sha256(path)}]))
    wf.journal.commit('unowned', cards=[leak])
    task, _ = case.begin(wf, case.task(wf, 'generate', refs=[wf.brief['id']], round=1, slot='H1', hypothesis_type='existing_method'))
    with case.assertRaises(ContractError): case.read(wf, task, 'read_asset', asset=str(path))

BranchACLRegression.test_child_interpret_inherits_parent_position = test_child_interpret_inherits_parent_position
BranchACLRegression.test_visible_path_mentions_do_not_grant_private_asset_or_card = test_visible_path_mentions_do_not_grant_private_asset_or_card
BranchACLRegression.test_unowned_asset_not_authorized_by_scientific_card = test_unowned_asset_not_authorized_by_scientific_card


def test_production_code_child_and_independent_reference(case):
    wf = case.workflow(1)
    public = case.science(wf, 'public_code')
    payload = public[0]['payload'].copy()
    payload['execution'] = dict(payload['execution'], requires_code=True, interface='quantizer_v1')
    spec = make_card('ExperimentCard', wf.journal.run_id, payload)
    wf.journal.commit('code-spec', cards=[spec])
    own, opposite = case.debate_task(wf), case.debate_task(wf, side='B')
    secret = private_result(case, wf, opposite)
    rid = wf.queue.enqueue('code-request', 'experiment_request', inputs={
        'parent_task': own['task_id'], 'requester': 'judger', 'refs': [wf.brief['id']], 'draft': payload})
    wf.queue.ready(); wf.queue.claim(rid)
    wf.queue.change(rid, 'running', patch={'experiment_ref': spec['id']})
    wf.progress_request(wf.queue.get(rid))
    for purpose in ('code', 'reference'):
        child = wf.tasks()['code-request/' + purpose]
        case.assertEqual(child['inputs']['parent_task'], rid)
        child, context = case.begin(wf, child)
        with case.assertRaises(ContractError): case.read(wf, child, 'read_card', ref=secret[1]['id'])
        with case.assertRaises(ContractError): case.read(wf, child, 'read_asset', asset=secret[1]['payload']['artifacts'][0]['path'])
        case.assertEqual(case.full_card(wf, child, spec['id']), spec)
        if purpose == 'reference': case.assertNotIn('directory', context['memory'])
        wf.queue.finish(child['task_id'])

BranchACLRegression.test_production_code_child_and_independent_reference = test_production_code_child_and_independent_reference
