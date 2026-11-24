from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from agents.source_workspace import edit, locate, revision
from core.cards import ContractError
from tools.development import stage_protocol, ClassificationDev
from runtime.coding_native import delivered, interrupted_usage
from types import SimpleNamespace


class DevelopmentView(unittest.TestCase):
    def test_frozen_request_separates_protocols(self):
        rows = [{'kind':'classification','split':'dev','count':4},{'kind':'ade_frozen','split':'ade_dev','count':2}]
        brief = {'evaluation_protocol':{'full_dataset':True},'execution_guidance':{
                 'development':{'manifest':{'sha256':'v'},'evaluations':rows}}}
        spec = {'service':'quantize_evaluate','evaluations':rows}
        self.assertFalse(stage_protocol(brief,spec,'L3')['full_dataset'])
        self.assertTrue(brief['evaluation_protocol']['full_dataset'])
        with self.assertRaises(ContractError):
            stage_protocol(brief,spec,'L2')
        with self.assertRaises(ContractError):
            stage_protocol(brief,{**spec,'evaluations':rows[:1]},'L3')

    def test_development_rejects_calibration_content_at_another_path(self):
        adapter = SimpleNamespace(class_to_idx={'c':0},PREPROCESS={},records=lambda:[{'file_sha256':'collision'}])
        manifest = {'class_to_idx':adapter.class_to_idx,'preprocessing':{},'classification':[
            {'file_sha256':'collision','label':0,'dataset_split':'train','sample_id':'train/c/x.jpg','path':'/train/c/x.jpg'}]}
        with self.assertRaises(ContractError):
            ClassificationDev(SimpleNamespace(data_adapter=adapter),manifest,'v')

    def test_revision_is_scoped_and_idempotent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = edit(root,{'old':'','new':'def f():\n    return 2\n'}, {})
            args = {'base':initial['revision'],'old':'return 2','new':'return 3'}
            result = edit(root,args,{})
            self.assertEqual(edit(root,args,{}),result)
            self.assertEqual(locate(revision(root,result['revision']),{'symbol':'f'}),0)
            with self.assertRaises(ContractError):
                revision(root/'another-task',result['revision'])
            with self.assertRaises(ContractError):
                edit(root,{**args,'asset':'/outside.py'},{})

    def test_delivery_and_interrupted_model_turns(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'implementation.py').write_text('pass')
            self.assertEqual(delivered(root),'pass')
            log = root/'events.jsonl'
            log.write_text('{"type":"turn.completed","usage":{"input_tokens":15}}\n'
                           '{"type":"turn.completed","usage":{"input_tokens":7}}\n{"type":')
            self.assertEqual(interrupted_usage(log)['completed_turn_usage'],{'input_tokens':22})

    def test_prepare_exports_and_independently_reloads_in_restricted_domain(self):
        from copy import deepcopy
        from unittest.mock import patch
        from tests.test_execution_contracts import full_fixture
        from tools.experiments import preflight_experiment
        from tools.experiment_worker import backend_and_data, evaluate_services
        brief, draft, plan = full_fixture()
        payload = preflight_experiment(draft, brief, executable=True)
        spec = payload['execution']
        backend, cfg, data = backend_and_data(spec)
        spec.update(result_protocol={'development_view':{'sha256':'probe'}}, evaluations=[])
        def build(local, cfg, visible, parameters):
            self.assertTrue(local.rows)
            self.assertEqual({r['sample_id'] for r in local.rows}, set(data.ids('calibration')))
            self.assertNotIn('development_view', visible['execution']['result_protocol'])
            return deepcopy(plan)
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for row in data.rows['calibration']:
                image = directory / row['sample_id']
                image.touch()
                row['path'] = str(image)
            with patch('tools.experiment_worker.module_from_approval',return_value=SimpleNamespace(build_plan=build)):
                value = evaluate_services({'payload':payload},backend,cfg,data,directory,None)
            self.assertTrue((directory/'model/manifest.json').is_file())
            self.assertTrue(value['observations']['reload']['passed'])

    def test_native_events_sum_all_reported_turns_and_legacy_rejects_commands(self):
        import json
        from tests.test_inner_json_repair import transport, envelope
        from ptq_agent.schemas import LLMRequest, PTQError
        request = LLMRequest(role='coding',system_prompt='fixture',messages=[],tools=[],context={},
                             response_schema={'type':'object','additionalProperties':True})
        answer = envelope('{"action":"submit","content":{"source_file":"implementation.py"}}')
        events = [{'type':'item.completed','item':{'type':'command_execution','exit_code':0}},
                  {'type':'turn.completed','usage':{'input_tokens':9,'output_tokens':3}},
                  {'type':'item.completed','item':{'type':'file_change'}},
                  {'type':'item.completed','item':{'type':'agent_message','text':answer}},
                  {'type':'turn.completed','usage':{'input_tokens':7,'output_tokens':4}}]
        raw = '\n'.join(map(json.dumps,events))
        response, details = transport().decode(raw, answer, request, native=True)
        self.assertEqual(response.usage,{'input_tokens':16,'output_tokens':7})
        self.assertFalse(details['tool_host_disabled_verified'])
        with self.assertRaises(PTQError):
            transport().decode(raw,answer,request)
