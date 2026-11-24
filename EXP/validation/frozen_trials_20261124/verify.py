"""Run the production regressions used in this iteration, retaining errors and exact code identity."""
from pathlib import Path
import subprocess
import sys
import time
import re
from core.journal import atomic_json, file_sha256
from runtime.identity import code_identity

OUT = Path(__file__).resolve().parent
MODULES = """frozen_trials calibration_search frozen_propagation frozen_helper_service execution_contracts
execution_binding_recovery dual_task_evaluation prepared_grids plan_execution_consistency
generation_loop gonly inspirations multi_candidate_l3 fallback_solutions branch_acl
context_cost read_delivery_boundaries memory_delivery scientific_skills evidence_handoffs
feedback_delivery knowledge_resources structure numeric_imports experiment_preflight""".split()

if __name__ == '__main__':
    cmd = [sys.executable, '-B', '-m', 'unittest', *(f'tests.test_{name}' for name in MODULES)]
    log = OUT / (sys.argv[1] if len(sys.argv) > 1 else 'verified_contracts.log')
    start = time.monotonic()
    core = code_identity()['core_sha256']
    with log.open('x') as stream:
        job = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
    text = log.read_text()
    counts = re.search(r'^Ran (\d+) tests? in ([\d.]+)s$', text, re.M)
    result = {'command': cmd, 'before_core': core, 'after_core': code_identity()['core_sha256'],
              'returncode': job.returncode, 'test_count': int(counts[1]) if counts else None,
              'seconds': time.monotonic() - start, 'log': log.name, 'log_sha256': file_sha256(log),
              'failures': re.findall(r'^(?:ERROR|FAIL): (.*)$', text, re.M)}
    atomic_json(OUT / ('verification.json' if job.returncode == 0 else log.stem + '_failure.json'), result)
    sys.exit(job.returncode)
