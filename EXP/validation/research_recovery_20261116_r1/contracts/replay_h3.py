"""CPU component replay of original H3; explicit reduced data, no formal Evidence."""
from pathlib import Path
import sys, json, time, importlib.util, traceback
from copy import deepcopy
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'QAViT'))
import runtime.paths
import torch
from core.journal import Journal, read_json, atomic_json, file_sha256
from core.solutions import SolutionLibrary
from tools.frozen import bind_solutions, load_frozen
from tools.experiments import normalize_source_refs
from tools.runtime_interface import execute_custom
from tools.experiment_worker import backend_and_data
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'EXP/runs/deitb_static_w4a4_gonly_20261115_r2'
start = time.time()
torch.set_num_threads(4)
result = {'scope': 'real CPU component replay; cal2/search1; no formal or full H3 claims', 'cuda_available': torch.cuda.is_available()}
try:
    original = Journal(RUN, read_json(RUN / 'research_run.json')['run_id'])
    brief = original.find_cards('ResearchBrief')[0]['payload']
    request = original.card('ExperimentCard-04be4a545bf24c4eb8ae88fff237ac27@1')
    payload = normalize_source_refs(request['payload'])
    target = 'c1c5ac43b4957e8afca2e8c4caca8f3cd1d3cc75eb72e0c8529bf2d00b6a27db'
    row = next(r for r in SolutionLibrary(original, brief).available() if r['id'] == target)
    journal = Journal(OUT / 'replay_journal', 'h3-cpu-replay')
    # New engineering journal cites original immutable rows; never writes original run.
    atomic_json(journal.root / 'research_run.json', {'run_id': journal.run_id})
    journal.commit('source', events=[{'type': 'solution_library', 'entries': [row]}])
    binding = bind_solutions(journal, [target], brief)[0]
    # Preserve identity-bearing model/data configuration, migrate only device.
    evidence = original.card(row['evidence_ref'])
    execution = deepcopy(original.card(evidence['payload']['experiment_ref'])['payload']['execution'])
    execution['backend'].setdefault('backend_overrides', {})['device'] = 'cpu'
    backend, cfg, data = backend_and_data(execution)
    original_code = RUN / 'code/606389ce29b8c0c3c7b9e7c4ae6b1cbf01b2905ae5f5275a857b2806a4163437/implementation.py'
    source = original_code.read_text()
    # Only the explicitly recorded sample budget changes, not mechanism functions.
    source = source.replace("[:32]", "[:2]").replace("[:64]", "[:1]")
    source = source.replace('len(calibration_ids) == 32', 'len(calibration_ids) == 2').replace('len(set(calibration_ids)) == 32', 'len(set(calibration_ids)) == 2')
    source = source.replace('len(search_ids) == 64', 'len(search_ids) == 1').replace('len(set(search_ids)) == 64', 'len(set(search_ids)) == 1')
    source = source.replace('exactly 32 unique calibration', 'exactly 2 unique calibration').replace('exactly 64 unique search', 'exactly 1 unique search')
    source = source.replace("'calibration_images': 32", "'calibration_images': 2")
    code = OUT / 'replay_implementation.py'; code.write_text(source)
    spec = importlib.util.spec_from_file_location('h3_cpu_replay', code)
    plugin = importlib.util.module_from_spec(spec); spec.loader.exec_module(plugin)
    payload['execution'].update(frozen_sources=[binding], calibration={'split': 'calibration', 'count': 2, 'offset': 0})
    payload['execution']['resolved_choices'] = {}
    result.update(original_request_ref=request['id'], original_code_sha256=file_sha256(original_code), replay_code_sha256=file_sha256(code),
                  normalized_evidence_refs=payload['evidence_refs'], source_binding={k:v for k,v in binding.items() if k not in ('assets','source_snapshot')})
    # Demonstrate original H3 missing binding -> request correction without running numerical code.
    from runtime.failures import failure_details
    from tools.runtime_interface import select_frozen_source
    try: select_frozen_source([], plugin.FROZEN_EVIDENCE)
    except Exception as exc: result['original_missing_binding'] = failure_details(exc, {})
    import tools.frozen as frozen_module
    original_candidate_module = frozen_module._candidate_module
    expected_audit = read_json(Path(binding['model_dir']) / 'manifest.json')['audit']
    def capture_loader(candidate, namespace=None):
        module = original_candidate_module(candidate, namespace)
        factory = module.make_executor
        def construct(*args):
            executor = factory(*args)
            audit = executor.audit
            def capture_audit(*values):
                value = audit(*values)
                differences = []
                def walk(a, b, path=''):
                    if isinstance(a, dict) and isinstance(b, dict):
                        for k in a.keys() | b.keys(): walk(a.get(k), b.get(k), path + '/' + k)
                    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
                        for i, (x,y) in enumerate(zip(a,b)): walk(x,y,path+'/'+str(i))
                    elif a != b: differences.append({'path': path, 'loaded': a, 'exported': b})
                walk(value, expected_audit)
                atomic_json(OUT / 'load_audit_differences.json', differences)
                return value
            executor.audit = capture_audit
            return executor
        module.make_executor = construct
        return module
    frozen_module._candidate_module = capture_loader
    measured = execute_custom(plugin, payload, backend, cfg, data, OUT)
    atomic_json(OUT / 'h3_measurement.json', measured)
    result.update(status='completed', elapsed_seconds=time.time()-start, execution_facts=measured['execution_facts'],
                  search_paired_comparisons=measured['observations']['search_paired_comparisons'],
                  path_checks=measured['observations']['path_checks'],
                  lambda_one_control=measured['observations']['lambda_one_control'])
except Exception as exc:
    result.update(status='failed', error={'code':type(exc).__name__, 'message':str(exc)}, elapsed_seconds=time.time()-start)
    traceback.print_exc()
finally:
    atomic_json(OUT / 'h3_replay_result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k in ('scope','status','elapsed_seconds','error')}, ensure_ascii=False), flush=True)
