"""Resource discovery, legacy citations and numerical snapshot compatibility (CPU)."""
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cards import ContractError, canonical, make_card
from core.journal import atomic_json, file_sha256, read_json
from core.workflow import Workflow
from runtime.identity import code_identity
from runtime.paths import QAVIT, ROOT
from tests import test_memory_delivery as delivery
from tests.audit_knowledge import audit, generated_catalog
from tests.test_infrastructure import TempCase
from tests.support import experiment_payload, hypothesis


class KnowledgeResources(TempCase):
    task = delivery.DeliveryRegression.task
    begin = delivery.DeliveryRegression.begin
    read = delivery.DeliveryRegression.read
    tools = delivery.DeliveryRegression.tools
    science = delivery.DeliveryRegression.science
    debate_task = delivery.DeliveryRegression.debate_task

    def generation(self, wf):
        return self.task(wf, 'generate', refs=[wf.brief['id']], round=1,
                         slot='H3', hypothesis_type='new_algorithm')

    def test_navigation_catalog_and_source_identity_consistency(self):
        self.assertEqual(audit()['errors'], [])
        self.assertEqual(generated_catalog(), (QAVIT / 'docs/skills/INDEX.md').read_text())
        self.assertIn('knowledge/INDEX.md', code_identity()['core_files'])
        # Only the new structure clause was added; all pre-existing AGENTS rules remain.
        text = (ROOT / 'AGENTS.md').read_text()
        self.assertEqual(text.count('新增或修改文件按职责组织：'), 1)
        self.assertIn('合理技术例外可按实际依赖保留原位', text)

    def test_all_seven_roles_discover_and_read_without_full_knowledge_injection(self):
        from tools.safeguards import write_artifact
        wf = self.workflow(1)
        parent = make_card('HypothesisCard', wf.journal.run_id, hypothesis(1, 'H1'))
        payload = experiment_payload()
        payload['execution'].update(requires_code=True, interface='quantizer_v1')
        spec = make_card('ExperimentCard', wf.journal.run_id, payload)
        wf.journal.commit('fixtures', cards=[parent, spec])
        candidate = write_artifact(wf.run_dir, spec, {'interface': 'quantizer_v1',
            'source': 'def fit(x, parameters):\n    return parameters\ndef apply(x, state, parameters):\n    return x\n'}, 'coding')
        tasks = [self.generation(wf), self.task(wf, 'screen', round=1),
                 self.task(wf, 'evolve', refs=[parent['id']], round=1, parent_ref=parent['id']),
                 self.task(wf, 'supervise'),
                 self.task(wf, 'code', refs=[spec['id']], experiment_ref=spec['id']),
                 self.task(wf, 'safeguard', refs=[spec['id']], experiment_ref=spec['id'], candidate=candidate),
                 wf.queue.get(wf.enqueue_recorder(1))]
        self.assertEqual(len({t['role'] for t in tasks}), 7)
        assets = ['knowledge/INDEX.md', 'skills/adapt-quantization-method/SKILL.md',
                  'methods/gptaq.md', 'solutions/seeds.json']
        for task in tasks:
            with self.subTest(role=task['role']):
                task, context = self.begin(wf, task)
                self.assertEqual(context['knowledge_index']['read_asset']['asset'], 'QAViT/knowledge/INDEX.md')
                for name in assets:
                    path = wf.asset_path(name)
                    self.assertNotIn(path.read_text(), canonical(context))
                    result = self.read(wf, task, 'read_asset', asset=name)
                    self.assertEqual(result['text'], path.read_text())
                    self.assertEqual(result['sha256'], file_sha256(path))
                wf.queue.finish(task['task_id'], result={'fixture': 'read-only role check complete'})

    def test_legacy_document_citations_survive_read_continue_restore_and_submit(self):
        wf = self.workflow(1)
        task, _ = self.begin(wf, self.generation(wf))
        assets = ['skills/diagnose-ptq-failure/SKILL.md', 'methods/gptaq.md']
        task, context = self.tools(wf, task, [('read_asset', {'asset': name}) for name in assets])
        handle = next(k for k, v in context['memory']['source_handles'].items() if v == wf.brief['id'])
        value = hypothesis(1, 'H3')
        value['sources'] = [{'ref': name, 'status': '已交付正文', 'supports': '定义；不是测量'} for name in assets]
        value['sources'] += [{'document': 'solutions/seeds.json', 'status': '只见目录；待测'},
                             {'source': '本卡理论推导', 'status': '待验证'}, {'ref': handle},
                             {'ref': 'C999999', 'supports': 'invalid optional ref'}]
        wf = Workflow(wf.run_dir)
        task = wf.queue.get(task['task_id'])
        wf.submit(task, value, 'scripted CPU')
        card = wf.result_card(wf.queue.get(task['task_id']))
        sources = card['payload']['sources']
        self.assertEqual([s['document'] for s in sources[:2]], [str(wf.asset_path(a)) for a in assets])
        self.assertEqual(sources[0]['supports'], value['sources'][0]['supports'])
        self.assertEqual(sources[-1]['ref'], wf.brief['id'])
        self.assertEqual(card['payload']['submission_issues'][0]['submission']['ref'], 'C999999')
        self.assertEqual(len(card['payload']['submission_issues']), 1)
        self.assertEqual(wf.scientific_refs(card['payload'], task), {wf.brief['id']})
        self.assertFalse(wf.journal.find_cards('EvidenceCard'))
        self.assertFalse(any(t['kind'] == 'experiment_request' for t in wf.tasks().values()))

    def test_document_compatibility_does_not_launder_private_refs_or_paths(self):
        from tests.test_branch_acl import private_result
        wf = self.workflow(1)
        self.science(wf, 'public')
        own, opposite = self.debate_task(wf), self.debate_task(wf, side='B')
        secret = private_result(self, wf, opposite)
        private_path = secret[1]['payload']['artifacts'][0]['path']
        for ref in (secret[1]['id'], private_path, str(ROOT / 'AGENTS.md'), 'methods/missing.md'):
            with self.assertRaises(ContractError):
                wf.resolve_card_refs(own, {'sources': [{'ref': ref}]})
        # Free document mentions cannot grant access, even after publication.
        value = hypothesis(1, 'H2')
        value['sources'] = [{'document': private_path}]
        mention = make_card('HypothesisCard', wf.journal.run_id, value)
        wf.journal.commit('untrusted_document_mention', cards=[mention])
        own, _ = self.begin(wf, own)
        with self.assertRaises(ContractError):
            self.read(wf, own, 'read_asset', asset=private_path)
        with self.assertRaises(ContractError):
            wf.resolve_card_refs(own, {'evidence_refs': ['methods/gptaq.md']})

    def test_missing_navigation_and_catalogs_do_not_block_original_research(self):
        wf = self.workflow(1)
        read_text, read_bytes, is_file = Path.read_text, Path.read_bytes, Path.is_file
        def no_method_index(path, *args, **kwargs):
            if path == QAVIT / 'docs/methods/index.json':
                raise FileNotFoundError(path)
            return read_text(path, *args, **kwargs)
        def no_skill_index(path):
            if path == QAVIT / 'docs/skills/INDEX.md':
                raise FileNotFoundError(path)
            return read_bytes(path)
        def no_navigation(path):
            return False if path.is_relative_to(QAVIT / 'knowledge') else is_file(path)
        with patch.object(Path, 'read_text', no_method_index), patch.object(Path, 'read_bytes', no_skill_index), \
                patch.object(Path, 'is_file', no_navigation):
            task, context = self.begin(wf, self.generation(wf))
            for key in ('knowledge_index', 'skill_catalog', 'method_catalog'):
                self.assertNotIn(key, context)
            found = self.read(wf, task, 'retrieve_memory', query={})
            self.assertEqual(found['method_documents']['entries'], [])
            wf.submit(task, hypothesis(1, 'H3'), 'scripted CPU without catalogs')
        self.assertEqual(wf.queue.get(task['task_id'])['state'], 'completed')

    def test_new_navigation_uses_frozen_hash_and_page_dedup_after_restore(self):
        wf = self.workflow(1)
        task, _ = self.begin(wf, self.generation(wf))
        task, context = self.tools(wf, task, [('read_asset', {'asset': 'knowledge/INDEX.md'})] * 2)
        self.assertEqual(canonical(context).count('# 科研资源导航与维护规则'), 1)
        wf = Workflow(wf.run_dir)
        task = wf.queue.get(task['task_id'])
        request = read_json(task['request_path'])
        request['read_permissions'][str(QAVIT / 'knowledge/INDEX.md')] = '0' * 64
        atomic_json(task['request_path'], request)
        with self.assertRaisesRegex(ContractError, '资产内容已变化'):
            self.read(wf, task, 'read_asset', asset='knowledge/INDEX.md')

    def test_numeric_method_library_query_read_and_snapshot_compatibility(self):
        from runtime.configuration import config_check
        from tests.support import validation_config
        from tools.experiments import resolve_execution
        from tools.experiment_worker import backend_and_data
        from tools.profiling import profile_host
        from ptq_agent.execution.method_library import MethodLibrary
        from ptq_agent.schemas import ToolCall, PTQError
        config = config_check(validation_config())
        execution = resolve_execution(experiment_payload()['execution'], config['research_brief'])
        backend, cfg, _ = backend_and_data(execution)
        self.assertTrue(cfg['loop']['require_method_documents'])  # Accepted legacy configuration.
        host = profile_host(backend, cfg, self.root / 'numeric')
        result = host.registry.invoke('profiling', ToolCall('q', 'query_method_library', {'target': 'all', 'query': ''}))
        self.assertEqual(result.status, 'success')
        self.assertEqual(len(result.data['methods']), 21)
        read = host.registry.invoke('profiling', ToolCall('r', 'read_method_document', {'methods': ['gptaq', 'smoothquant']}))
        self.assertEqual(read.status, 'success')
        self.assertEqual(len(read.data['documents']), 2)
        saved = read_json(host.run_dir / 'method_library.json')
        restored = MethodLibrary(self.root / 'missing_source_tree', snapshot=saved)
        self.assertEqual(restored.read('gptaq'), host.method_library.read('gptaq'))
        saved['documents']['gptaq']['content'] += 'tampered'
        with self.assertRaises(PTQError):
            MethodLibrary(snapshot=saved)
