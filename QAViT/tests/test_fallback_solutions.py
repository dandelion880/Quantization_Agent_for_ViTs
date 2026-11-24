"""B retention and strict frozen-source binding regressions."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError, fingerprint
from core.journal import atomic_json, file_sha256
from core.solutions import SolutionLibrary
from tests.test_infrastructure import TempCase
from tools.frozen import _source_files, _candidate_module, bind_solutions, historical_source, _materialize_source


class FallbackTests(TempCase):
    def promote(self, wf, number, top1, miou, valid=True):
        evidence = {'id': 'ev'+str(number), 'payload': {'measurement_source':'real', 'status':'completed',
            'provenance': {'model_ref': 'model'+str(number)}, 'observations': {'evaluations': {
            'classification': {'full_dataset': True, 'metrics': {'top1':top1}},
            'segmentation': {'full_dataset':valid, 'metrics': {'mIoU':miou}}}}, 'artifacts': []}}
        wf.solutions.consider_round(number, evidence, {'id':'exp','payload':{'level':'L3','execution':{}}},
            {'id':'selection','payload':{'selected_slot':'H2','selected_hypothesis_ref':'h@2'}}, validator=lambda *a,**k:None)

    def test_pending_seeds_are_never_B(self):
        wf=self.workflow(1)
        self.assertEqual(wf.solutions.fallback()['status'], 'pending_no_verified_result')
        self.assertTrue(all(r['measurement_origin']=='unmeasured_seed' for r in wf.solutions.catalog()))

    def test_misfiled_top3_does_not_supply_a_foreign_model_or_protocol_B(self):
        wf = self.workflow(1)
        library = wf.solutions
        path = self.root / 'misfiled_top3.json'
        for hash_in_bucket, hash_in_row in [('another-model', library.key),
                                              (library.key, 'another-protocol')]:
            atomic_json(path, {'protocol_hash': hash_in_bucket,
                               'entries': [{'protocol_hash': hash_in_row}]})
            with self.assertRaises(ContractError):
                library._bound_entries(path)
        self.assertEqual(library.fallback()['status'], 'pending_no_verified_result')

    def test_B_survives_failure_lower_result_and_restart(self):
        wf=self.workflow(1)
        self.promote(wf,1,80,40)
        original=wf.solutions.fallback()
        self.promote(wf,2,99,80,False)
        self.promote(wf,3,81,30)
        self.assertEqual(wf.solutions.fallback(),original)
        restored=SolutionLibrary(wf.journal,wf.brief['payload'],shared=False)
        self.assertEqual(restored.fallback(),original)
        self.assertFalse(original['competes_in_preselection'])

    def test_valid_promotion_and_duplicate_round_idempotent(self):
        wf=self.workflow(1)
        self.promote(wf,1,80,40);self.promote(wf,2,79,41)
        self.assertEqual(wf.solutions.fallback()['candidate']['model_ref'],'model2')
        sequence=wf.journal.sequence
        self.promote(wf,2,100,100)
        self.assertEqual(wf.journal.sequence,sequence)
        self.assertEqual(wf.solutions.fallback()['candidate']['metrics']['mIoU'],41)

    def test_measured_top3_does_not_discard_unmeasured_starts(self):
        wf=self.workflow(1)
        for i in range(1,5):self.promote(wf,i,80,i)
        self.assertEqual(len(wf.solutions.snapshot()),3)
        pending=[r for r in wf.solutions.catalog() if r['status']=='pending_evaluation']
        self.assertEqual(len(pending),3)
        self.assertTrue(all(Path(r['configuration_asset']).is_file() for r in pending))

    def test_historical_B_scores_not_current_measurements(self):
        wf=self.workflow(1);self.promote(wf,1,80,40)
        rows=wf.solutions.snapshot()
        rows[0]['run_dir']='/original/run'
        wf.journal.commit('history',events=[{'type':'solution_library','entries':rows}])
        b=wf.solutions.fallback()['candidate']
        self.assertEqual(b['measurement_origin'],'historical')
        self.assertFalse(b['new_measurement'])
        incumbent={'model_ref':'local','evidence_refs':['local-ev'],'top1':82,'mIoU':42}
        self.assertEqual(wf.solutions.fallback(incumbent)['candidate']['model_ref'],'local')

    def test_seed_or_unknown_cannot_bind_as_verified_package(self):
        wf=self.workflow(1)
        for ref in ('gptaq_w4a4','not-a-solution'):
            with self.assertRaises(ContractError):bind_solutions(wf.journal,[ref],wf.brief['payload'])

    def test_snapshot_hash_and_declared_source_version_both_enforced(self):
        path=self.root/'snapshot.json'
        source={'tools/ptq_agent/__init__.py':'# original\n'}
        atomic_json(path,{'events':[{'source_snapshot':source}]})
        binding={'source_snapshot':{'path':str(path),'sha256':file_sha256(path),'field':'events.0.source_snapshot'},
                 'source_version':fingerprint({k:hashlib.sha256(v.encode()).hexdigest() for k,v in source.items()})}
        self.assertEqual(_source_files(binding),source)
        changed=deepcopy(binding);changed['source_version']='bad'
        with self.assertRaises(ContractError):_source_files(changed)
        atomic_json(path,{'events':[{'source_snapshot':{'other':'changed'}}]})
        with self.assertRaises(ContractError):_source_files(binding)

    def test_legacy_source_backfill_uses_committed_experiment_era(self):
        from tests.support import experiment_payload
        wf=self.workflow(1)
        old=wf.broker.request(experiment_payload(),requester='generation')
        wf.journal.commit('old-exp',cards=[old])
        first=wf.solutions.source_record(wf.journal.sequence)
        changed={'later.py':'# later source\n'}
        version=fingerprint({k:hashlib.sha256(v.encode()).hexdigest() for k,v in changed.items()})
        wf.journal.commit('intervention',events=[{'type':'implementation_intervention',
            'code_identity':{'core_sha256':version},'source_snapshot':changed}])
        new=wf.broker.request(experiment_payload(),requester='generation')
        wf.journal.commit('new-exp',cards=[new])
        base={'run_dir':str(wf.journal.root),'configuration':wf.brief['payload']}
        before={str(p):file_sha256(p) for p in wf.journal.root.rglob('*.json')}
        self.assertEqual(historical_source({**base,'experiment_ref':old['id']}),first)
        latest=historical_source({**base,'experiment_ref':new['id']})
        self.assertEqual(_source_files(latest),changed)
        with self.assertRaises(ContractError):
            historical_source({**base,'experiment_ref':old['id'],'source_version':version})
        self.assertEqual(before,{str(p):file_sha256(p) for p in wf.journal.root.rglob('*.json')})

    def test_host_prepares_source_cache_worker_never_writes_it(self):
        from unittest.mock import patch
        source={'tools/ptq_agent/__init__.py':'# source\n','tools/vision_models.py':'# loader\n'}
        path=self.root/'snapshot.json';atomic_json(path,{'source_snapshot':source})
        binding={'source_snapshot':{'path':str(path),'sha256':file_sha256(path),'field':'source_snapshot'},
            'source_version':fingerprint({k:hashlib.sha256(v.encode()).hexdigest() for k,v in source.items()}),
            'source_cache':str(self.root/'host-cache')}
        with self.assertRaises(ContractError):_materialize_source(binding,create=False)
        root=_materialize_source(binding)
        with patch.object(Path,'write_text',side_effect=AssertionError('worker wrote cache')):
            self.assertEqual(_materialize_source(binding,create=False),root)
        (root/'vision_models.py').write_text('# changed')
        with self.assertRaises(ContractError):_materialize_source(binding,create=False)

    def test_custom_executor_imports_use_matching_private_source(self):
        import types
        namespace='qavit_history_test_binding'
        private=types.ModuleType(namespace);private.marker='historical'
        sys.modules[namespace]=private
        path=self.root/'custom.py';path.write_text('from ptq_agent import marker\ndef make_executor(*args): return marker\n')
        candidate={'path':str(path),'source_sha256':file_sha256(path)}
        try:
            module=_candidate_module(candidate,namespace)
            self.assertEqual(module.make_executor(),'historical')
            self.assertEqual(file_sha256(path),candidate['source_sha256'])
        finally:sys.modules.pop(namespace,None)

if __name__=='__main__':unittest.main()
