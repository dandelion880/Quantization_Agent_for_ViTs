"""Engineering regressions from observed request shapes. No accuracy claims."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from runtime.paths import QAVIT
from agents.roles import TOOLS
from agents.source_workspace import edit, revision
from core.cards import ContractError, validate_schema
from core.journal import file_sha256
from tools.runtime_interface import execute_custom, interface_fixture
from tools.safeguards import write_artifact, verify_artifact


class BreakpointContracts(unittest.TestCase):
    def test_decode_order_and_diagnostic_receipt(self):
        payload = {'level': 'L2', 'execution': {'service': 'custom_experiment',
            'resolved_choices': {'training': False}, 'calibration': {'split':'calibration','offset':1,'count':1},
            'parameters': {'diagnostic_view': {'split':'search','offset':2,'count':2,'ordering_seed':7}}}}
        backend, cfg, data = interface_fixture({}, {'seed':7,'shape':[1,5,8],'sites':['blocks.0.mlp.fc2'],
                                                    'counts':{'calibration':3,'search':5,'final':1}})
        loaded = []
        decode = backend.data_adapter.load_image
        def tracked(row):
            loaded.append(row['sample_id'])
            return decode(row)
        backend.data_adapter.load_image = tracked
        def diagnostic(_, ctx):
            view = ctx['data']
            self.assertEqual(view.identity['diagnostic_view']['ordered_pool_ids'], data.ids('search'))
            self.assertEqual(view.identity['split_ids']['search'], data.ids('search')[2:4])
            with self.assertRaisesRegex(ContractError,'split'):
                next(view.batches('final'))
            for split in ('calibration','search'):
                self.assertEqual([sid for _,sid in view.batches(split)], view.ids(split))
            return {'observations':{'decoded':len(loaded)}}
        with TemporaryDirectory() as tmp:
            args = (payload,backend,cfg,data,Path(tmp))
            result = execute_custom(SimpleNamespace(execute=diagnostic), *args)
            self.assertEqual(loaded, data.ids('calibration')[1:2] + data.ids('search')[2:4])
            self.assertEqual(result['sample_selection']['search'], loaded[1:])
            with self.assertRaisesRegex(ContractError,'实际读取范围'):
                execute_custom(SimpleNamespace(execute=lambda *_:{'observations':{'reported':loaded}}), *args)
            for field, value in [('offset',5),('ordering_seed',99)]:
                selection = payload['execution']['parameters']['diagnostic_view']
                old, selection[field] = selection[field], value
                with self.assertRaises(ContractError):
                    execute_custom(SimpleNamespace(execute=diagnostic), *args)
                selection[field] = old
            self.assertEqual(len(loaded),3)

    def test_authorized_edit_retry_restore_submit(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / 'parent.py'
            src.write_bytes(b'def execute(p,c):\r\n    return 0\r\n')
            acl = {str(src):file_sha256(src)}
            args = {'asset':str(src),'old':'return 0','new':'return 1'}
            with self.assertRaisesRegex(ContractError,'base'):
                validate_schema(args, TOOLS['edit_source']['parameters'], 'edit_source')
            with self.assertRaisesRegex(ContractError,'pinned file sha256'):
                edit(root,{**args,'base':'0'*64},acl)
            with self.assertRaisesRegex(ContractError,'authorized'):
                edit(root,{**args,'base':acl[str(src)]},{})
            one = edit(root,{**args,'base':acl[str(src)]},acl)
            next_edit = {'base':one['revision'],'old':'return 1','new':'return 2'}
            two = edit(root,next_edit,acl)
            self.assertEqual(two, edit(root,next_edit,acl))
            experiment = {'id':'request','content_sha256':'hash','payload':{'execution':{'interface':'experiment_v1'}}}
            artifact = write_artifact(root,experiment,{'source':revision(root,two['revision'])},'coding')
            verify_artifact(artifact,experiment)
            self.assertEqual(src.read_bytes(), b'def execute(p,c):\r\n    return 0\r\n')

    def test_drain_ignores_queued_coding_but_charges_real_overlap(self):
        from core.workflow import Workflow
        from unittest.mock import Mock
        wf = Mock()
        tasks = {'numeric':{'kind':'experiment','state':'running'},'coding':{'kind':'llm','state':'ready'}}
        wf.queue.snapshot.return_value = tasks
        Workflow.update_numeric_wait(wf)
        wf.budget.numeric_wait.assert_called_with(False)
        Workflow.update_numeric_wait(wf, draining=True)
        wf.budget.numeric_wait.assert_called_with(True)
        tasks['coding']['state'] = 'running'
        Workflow.update_numeric_wait(wf, draining=True)
        wf.budget.numeric_wait.assert_called_with(False)

    def test_stopped_old_segment_is_settled_when_solution_registration_needs_retry(self):
        import time
        from unittest.mock import patch
        from core.workflow import Workflow
        from runtime.run_segments import run_managed
        from tests.support import validation_config
        config = validation_config()
        config['budget'].update(clock_mode='active_segments', min_free_bytes=0)
        with TemporaryDirectory() as tmp:
            wf = Workflow(Path(tmp)/'branch', config)
            from core.journal import atomic_json
            atomic_json(wf.run_dir/'summary.json', {'finalization_errors':[{'step':'prior_monitor'}]})
            wf.budget.begin_segment(started_unix=time.time()-1)
            error = [{'step':'consider_l3','round':4,'code':'OSError','message':'temporarily unavailable'}]
            with patch('runtime.run_segments.collect_final_receipts', side_effect=[error, []]), \
                    patch.object(wf, 'scope_target_reached', return_value=True):
                report = run_managed(wf, started_unix=time.time())
            self.assertIsNone(report['budget']['active_segment'])
            self.assertTrue(report['state_saved'])
            self.assertTrue(any(e.get('type')=='run_status' and e.get('error')==error for e in wf.journal.events))
            stops = [e for e in wf.journal.events if e['type']=='budget_segment_stopped']
            self.assertTrue(stops[0]['recovered_without_stop'])
            self.assertEqual(len(stops), 2)

    def test_handoff_read_exact_input_and_deny_sibling(self):
        from tests.support import validation_config
        from core.workflow import Workflow
        from runtime.paths import ROOT
        with TemporaryDirectory(dir=ROOT/'EXP/inputs') as tmp, TemporaryDirectory() as work:
            baseline = Path(tmp)/'receipt.json'
            baseline.write_text('{"baseline":1}')
            sibling = Path(tmp)/'sibling.json'
            sibling.write_text('{}')
            config = validation_config()
            config['research_brief']['research_handoff'] = {'baseline':{'path':str(baseline),'sha256':file_sha256(baseline)}}
            wf = Workflow(Path(work)/'branch',config)
            tid = wf.enqueue_role('focus','focus',refs=[wf.brief['id']],context={'round':1})
            def read(path):
                return wf.host_tool({'inputs':{'parent_task':tid,'name':'read_asset','arguments':{'asset':str(path)}}})
            self.assertEqual(read(baseline)['text'],baseline.read_text())
            with self.assertRaises(ContractError):
                read(sibling)
            baseline.write_text('{}')
            with self.assertRaisesRegex(ContractError,'变化'):
                read(baseline)

    def test_nested_custom_export_cold_reload_and_next_local_forward(self):
        import torch
        from core.journal import atomic_json
        from core.cards import fingerprint
        from tests.test_calibration_search import fixture
        from ptq_agent.execution.model import RTNExecutor
        from ptq_agent.execution.evaluation import model_digest
        from tools.experiments import asset
        from tools.frozen import reload_complete
        from tools.experiment_worker import data_view
        backend,cfg,data,plan = fixture()
        identity = {**backend.identity(),'config':{'device':'cpu'},'agent_code_sha256':'test-numeric'}
        backend.identity = lambda: dict(identity)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root/'model'
            directory.mkdir()
            exporter = RTNExecutor(backend,cfg,data)
            model,w,a,audit = exporter.prepare(plan,{})
            exporter.export(model,plan,w,a,directory/'parent',audit)
            atomic_json(directory/'manifest.json',{'format':'wrapped','plan':plan})
            source = root/'implementation.py'
            source.write_text("from ptq_agent.execution.model import RTNExecutor\n"
                "class Wrapped(RTNExecutor):\n    def load(self, path):\n        return super().load(path/'parent')\n"
                "def make_executor(backend,cfg,data):\n    return Wrapped(backend,cfg,data)\n")
            archive = root/'snapshot.json'
            atomic_json(archive,{'files':{'implementation.py':source.read_text()}})
            segmentation = root/'segmentation.json'
            atomic_json(segmentation,{'backbone_hashes':[model_digest(backend.model),model_digest(model)]})
            binding = {'model_dir':str(directory),'model_ref':'fixture','model_identity':dict(identity),
                       'evidence_ref':'fixture','verified_segmentation':asset(segmentation),
                       'assets':[asset(p) for p in directory.rglob('*') if p.is_file()],
                       'source_snapshot':{**asset(archive),'field':'files'},
                       'source_version':fingerprint({'implementation.py':file_sha256(source)}),
                       'candidate':{'path':str(source),'source_sha256':file_sha256(source)}}
            cold,_ = reload_complete(binding,backend,cfg,data)
            x,_ = next(data_view(data,{'split':'search','count':1,'offset':1}).batches('search'))
            with torch.inference_mode():
                self.assertTrue(torch.equal(backend.forward_model(cold,x),backend.forward_model(model,x)))
            with self.assertRaisesRegex(ContractError,'identity differs'):
                reload_complete({**binding,'model_identity':{**identity,'fixture':'other'}},backend,cfg,data)
            (directory/'parent'/'extra').touch()
            with self.assertRaisesRegex(ContractError,'changed or incomplete'):
                reload_complete(binding,backend,cfg,data)
