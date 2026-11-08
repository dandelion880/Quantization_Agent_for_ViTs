"""Optional scientific knowledge through the existing context/read/submit path.

Scripted CPU integration only; real subscription checks live separately in EXP.
"""
from pathlib import Path
import re
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cards import ContractError, canonical, make_card
from core.journal import file_sha256
from core.workflow import Workflow
from runtime.identity import code_identity
from runtime.paths import QAVIT
from tests.test_infrastructure import TempCase
from tests import test_memory_delivery as delivery
from tests.support import hypothesis


class ScientificSkills(TempCase):
    task = delivery.DeliveryRegression.task
    begin = delivery.DeliveryRegression.begin
    tools = delivery.DeliveryRegression.tools
    read = delivery.DeliveryRegression.read
    science = delivery.DeliveryRegression.science

    def generation(self, wf):
        return self.task(wf, 'generate', refs=[wf.brief['id']], round=1,
                         slot='H3', hypothesis_type='new_algorithm')

    def test_catalog_metadata_links_identity_and_all_assets_readable(self):
        wf = self.workflow(1)
        task, context = self.begin(wf, self.generation(wf))
        root = QAVIT / 'docs/skills'
        skills = sorted(root.glob('*/SKILL.md'))
        self.assertEqual(len(skills), 4)
        identity = code_identity()['core_files']
        catalog = context['skill_catalog']
        self.assertLess(len(catalog), 2400)
        for path in skills:
            body = path.read_text()
            self.assertIn('name: ' + path.parent.name, body)
            self.assertIn('description: >-', body)
            self.assertIn(path.parent.name + '/SKILL.md', catalog)
            # A body or reference is never injected by catalog discovery alone.
            self.assertNotIn(body, canonical(context))
        for path in root.rglob('*.md'):
            asset = 'skills/' + str(path.relative_to(root))
            result = self.read(wf, task, 'read_asset', asset=asset)
            self.assertEqual(result['text'], path.read_text())
            self.assertEqual(result['sha256'], file_sha256(path))
            self.assertIsNone(result['next_offset'])
            self.assertIn(str(path.relative_to(QAVIT)), identity)
            for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
                if '://' not in target:
                    self.assertTrue((path.parent / target).is_file(), (path, target))

    def test_discovery_is_for_research_roles_and_not_engineering_or_supervisor(self):
        wf = self.workflow(1)
        parent = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1, 'H1'))
        wf.journal.commit('parent', cards=[parent])
        cases = [self.generation(wf), self.task(wf, 'screen', round=1),
                 self.task(wf, 'evolve', refs=[parent['id']], round=1, parent_ref=parent['id'])]
        recorder = wf.enqueue_recorder(1)
        cases.append(wf.queue.get(recorder))
        for task in cases:
            self.assertIn('skill_catalog', wf.request_body(task)['spec']['context'])
        experiment = self.science(wf, 'code-scope')[0]
        for task in [self.task(wf, 'supervise'),
                     self.task(wf, 'code', refs=[experiment['id']], experiment_ref=experiment['id'])]:
            self.assertNotIn('skill_catalog', wf.request_body(task)['spec']['context'])

    def test_index_and_repeated_reads_are_deduplicated_across_continuation(self):
        wf = self.workflow(1)
        task, _ = self.begin(wf, self.generation(wf))
        asset = 'skills/diagnose-ptq-failure/SKILL.md'
        task, context = self.tools(wf, task, [('read_asset', {'asset': x})
                                  for x in ('skills/INDEX.md', asset, asset)])
        self.assertEqual(canonical(context).count('# QAViT Scientific Skill Catalog'), 1)
        self.assertEqual(canonical(context).count('# 定位量化失效'), 1)
        reference = 'skills/diagnose-ptq-failure/references/quantization-error-map.md'
        self.assertNotIn('# W / A / 交互误差定位', canonical(context))
        wf = Workflow(wf.run_dir)
        task, context = self.tools(wf, wf.queue.get(task['task_id']), [('read_asset', {'asset': reference})])
        self.assertEqual(canonical(context).count('# 定位量化失效'), 1)
        self.assertEqual(canonical(context).count('# W / A / 交互误差定位'), 1)
        wf.submit(task, hypothesis(1, 'H3'), 'scripted CPU')
        self.assertEqual(wf.queue.get(task['task_id'])['state'], 'completed')
        self.assertFalse(wf.journal.find_cards('EvidenceCard'))

    def test_absent_skill_tree_does_not_block_original_h3_path(self):
        wf = self.workflow(1)
        read_bytes, rglob = Path.read_bytes, Path.rglob
        root = QAVIT / 'docs/skills'
        def without_catalog(path):
            if path == root / 'INDEX.md':
                raise FileNotFoundError(path)
            return read_bytes(path)
        def without_skills(path, pattern):
            return (p for p in rglob(path, pattern) if not p.is_relative_to(root))
        with patch.object(Path, 'read_bytes', without_catalog), patch.object(Path, 'rglob', without_skills):
            task, context = self.begin(wf, self.generation(wf))
            self.assertNotIn('skill_catalog', context)
            self.assertFalse(any('/docs/skills/' in p for p in wf.authorized_assets(task)))
            wf.submit(task, hypothesis(1, 'H3'), 'scripted CPU')
        self.assertEqual(wf.queue.get(task['task_id'])['state'], 'completed')
        self.assertEqual(len(wf.tasks()), 1)

    def test_failed_selection_returns_to_researcher_without_a_new_gate(self):
        wf = self.workflow(1)
        task, _ = self.begin(wf, self.generation(wf))
        wf.receive_tools(task, {'tool_calls': [{'id': 'missing', 'name': 'read_asset',
            'arguments': {'asset': 'skills/unavailable/SKILL.md'}}]}, 'scripted CPU')
        with patch.object(wf.runtime, 'launch', side_effect=AssertionError('no LLM in unit test')):
            wf.dispatch()
        task, context = self.begin(wf, wf.queue.get(task['task_id']))
        self.assertIn('ContractError', canonical(context['tool_results']))
        wf.submit(task, hypothesis(1, 'H3'), 'scripted CPU')
        self.assertEqual(wf.queue.get(task['task_id'])['state'], 'completed')
        self.assertFalse(any(t['kind'] == 'experiment_request' for t in wf.tasks().values()))

    def test_existing_asset_hash_and_path_boundaries_still_apply(self):
        wf = self.workflow(1)
        task, _ = self.begin(wf, self.generation(wf))
        asset = QAVIT / 'docs/skills/assess-evidence-transfer/SKILL.md'
        from core.journal import read_json, atomic_json
        path = Path(task['request_path'])
        body = read_json(path)
        body['read_permissions'][str(asset)] = '0' * 64
        atomic_json(path, body)
        with self.assertRaisesRegex(ContractError, '资产内容已变化'):
            self.read(wf, task, 'read_asset', asset=str(asset))
        with self.assertRaises(ContractError):
            self.read(wf, task, 'read_asset', asset='AGENTS.md')
