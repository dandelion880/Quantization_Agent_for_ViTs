"""Recorder checkpoint, immutable procedure packages and snapshot-scoped reads."""
from copy import deepcopy
from unittest.mock import patch

from core.cards import ContractError, fingerprint, make_card
from core.workflow import Workflow
from runtime.identity import code_identity
from tests.support import hypothesis, validation_config
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as delivery


class PublishedSkills(TempCase):
    begin = delivery.DeliveryRegression.begin
    task = delivery.DeliveryRegression.task
    read = delivery.DeliveryRegression.read

    def setUp(self):
        super().setUp()
        root = patch('core.skills.SHARED_ROOT', self.root / 'shared')
        root.start()
        self.addCleanup(root.stop)

    def run_named(self, name):
        cfg = validation_config()
        cfg['budget']['min_free_bytes'] = 0
        return Workflow(self.root / name, cfg)

    def focus(self, w):
        return self.begin(w, self.task(w, 'focus', round=1, refs=[w.brief['id']]))[0]

    def record(self, w, action, name, base=None):
        h = make_card('HypothesisCard', w.journal.run_id, hypothesis(1, 'H1'))
        w.journal.commit('fixture/' + h['id'], cards=[h])
        tid = w.enqueue_recorder(1, incomplete=True, key='record/' + h['id'])
        task, _ = self.begin(w, w.queue.get(tid))
        draft = {'action':action, 'name':name, 'description':'Choose paired controls',
                 'scope':'same parent and image identities', 'counterexamples':'unpaired observations',
                 'procedure':'Match images before comparing arms.', 'rationale':'dependent observations',
                 'sources':[{'ref':h['id'], 'field':'/payload/mechanism'}]}
        if base:
            draft['base_sha256'] = base
        return task, {'scientific_analysis':'bounded comparison', 'generation_guidance':'compare arms',
            'next_questions':[], 'knowledge_updates':[{'status':'hypothesis','text':'check order',
                'scope':{'design':'paired'},'rationale':'controlled', 'evidence_refs':[]}],
            'skill_updates':[{'name':'../bad','sources':[{'ref':'fake'}]}, draft]}

    def test_isolated_publication_crash_replay_cross_run_and_retirement(self):
        a = self.run_named('A')
        old = self.run_named('old')
        first = self.focus(a)
        identity = code_identity()['core_sha256']
        name = 'evaluate-ptq-candidate'
        asset = 'skills/' + name + '/SKILL.md'
        original = self.read(a, first, 'read_asset', asset=asset)
        manual = original['asset'].split('@')[1].split('/')[0]
        task, content = self.record(a, 'revise', name, manual)
        publish = a.skills.publish
        def crash_after_release(*args):
            publish(*args)
            raise OSError('injected crash after shared commit')
        with patch.object(a.skills,'publish',side_effect=crash_after_release):
            a.submit(task,content,'checkpoint')
        card = a.result_card(a.queue.get(task['task_id']))
        self.assertEqual([r['status'] for r in card['payload']['skill_updates']], ['invalid','queued'])
        self.assertTrue(a.memory.snapshot()['L2'])
        self.assertFalse(a.memory.recording_batch()['source_refs'])
        self.assertTrue(list(a.skills.root.glob('releases/*.json')))
        a = Workflow(a.run_dir)
        a.submit(task,content,'replayed checkpoint')
        a.skills.reconcile()
        result = [e for e in a.journal.events if e['type']=='skill_update_result']
        self.assertEqual([e['status'] for e in result], ['published'])
        self.assertEqual(len(list(a.skills.root.glob('releases/*.json'))),1)
        self.assertEqual(self.read(a,first,'read_asset',asset=asset),original)
        self.assertEqual(self.read(old,self.focus(old),'read_asset',asset=asset),original)
        fresh = self.focus(a)
        revised = self.read(a,fresh,'read_asset',asset=asset)
        self.assertIn('Match images',revised['text'])
        self.assertEqual(self.read(a,fresh,'read_asset',**original['version_read'])['text'],original['text'])
        b = self.run_named('B')
        use = self.focus(b)
        self.assertEqual(self.read(b,use,'read_asset',asset=asset),revised)
        src = self.read(b,use,'read_asset',asset='skills/'+name+'/references/origin.json')['text']
        self.assertIn(card['content_sha256'],src)
        self.assertIn(fingerprint(hypothesis(1,'H1')['mechanism']),src)
        with self.assertRaises(ContractError):
            self.read(b,use,'read_asset',asset=str(a.run_dir/'research_run.json'))
        retire, data = self.record(b,'retire',name,result[0]['version'])
        b.submit(retire,data,'retire')
        self.assertNotIn(asset,b.skills.catalog(b.journal.sequence,'paired'))
        self.assertEqual(self.read(b,use,'read_asset',asset=asset),revised)
        exact=b.skills.read(revised['asset'],b.journal.sequence)
        self.assertEqual(exact[1],revised['text'])
        self.assertEqual(code_identity()['core_sha256'],identity)

    def test_same_name_conflict_and_bounded_discovery(self):
        a, b = self.run_named('A'), self.run_named('B')
        for w in (a,b):
            t,c = self.record(w,'create','paired-design')
            w.submit(t,c,'concurrent independent proposals')
        self.assertEqual([e['status'] for e in b.journal.events if e['type']=='skill_update_result'],['conflict'])
        for index in range(20):
            name='unrelated-'+str(index)
            version=a.skills.package(name,{'SKILL.md':f'---\nname: {name}\ndescription: unrelated\n---\n'+'large'*2000})
            a.journal.commit(name, events=[{'type':'skill_update_result','status':'published','name':name,'version':version}])
        cat=a.skills.catalog(a.journal.sequence,'paired controls')
        self.assertLessEqual(cat.count('/SKILL.md'),7)
        self.assertIn('paired-design',cat)
        self.assertNotIn('large',cat)
        self.assertIn('unrelated-19',a.skills.read('skills/INDEX.md',a.journal.sequence)[1])
