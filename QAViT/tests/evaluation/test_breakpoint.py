"""Controlled replay isolation checks. These tests never call a model."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.evaluation import breakpoint as bp
from tools.evaluation.common import digest, fingerprint, write_json


class BreakpointIsolation(unittest.TestCase):
    def test_read_only_start_and_divergent_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = {'execution_id':'t-a1','kind':'llm','input_versions':{'v':'pinned'},
                       'read_permissions':{},'spec':{'role':'generation','purpose':'generate',
                       'context':{'task':{},'read_view':{'sequence':4}},'settings':{},'messages':[]}}
            obj = root/'sources/input'
            write_json(obj, request)
            sha = digest(obj.read_bytes())
            manifest = {'source_files':[{'path':'jobs/t-a1/request.json','object':'sources/input','sha256':sha}]}
            manifest['manifest_sha256'] = fingerprint(manifest)
            write_json(root/'manifest.json',manifest)
            case = {'execution_id':'t-a1','request_sha256':sha,'manifest_sha256':manifest['manifest_sha256'],
                    'read_view':{'sequence':4},'input_versions':{'v':'pinned'},'acceptance':'ASSESSOR_ONLY'}
            actor = bp.start_request(root,case)
            write_json(root/'later.json',{'answer':'FUTURE_ONLY'})
            def different(spec, *_):
                self.assertNotIn('ASSESSOR_ONLY', str(spec))
                self.assertNotIn('FUTURE_ONLY', str(spec))
                return {'response':{'usage':{},'tool_calls':[{'name':'retrieve_memory','id':'new','arguments':{}}]}}
            with patch.object(bp,'call_once',side_effect=different):
                result = bp.replay(actor,root/'independent',2)
            self.assertEqual(result['status'],'unmeasured')
            self.assertFalse((root/'independent/steps/1/delivered.json').exists())
            self.assertEqual(digest(obj.read_bytes()), sha)
            for key in ('input_versions','read_view','request_sha256'):
                with self.assertRaises(ValueError):
                    bp.start_request(root,{**case,key:'wrong'})
