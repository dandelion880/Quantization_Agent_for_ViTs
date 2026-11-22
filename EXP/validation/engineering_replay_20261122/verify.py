"""Run an individual CPU component or the union, without starting a research workflow."""
import argparse
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'QAViT'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

GROUPS = {
    'data': ['breakpoint_contracts', 'execution_contracts', 'development_view', 'experiment_inputs', 'reference_optimization'],
    'frozen': ['breakpoint_contracts', 'frozen_helper_service', 'frozen_propagation', 'dual_task_evaluation', 'source_recovery'],
    'source': ['breakpoint_contracts', 'code_asset_ownership', 'code_repair', 'execution_binding_recovery', 'stale_safeguard_recovery'],
    'knowledge': ['branch_acl', 'read_delivery_boundaries', 'knowledge_resources', 'memory_delivery', 'research_topics',
                  'scientific_skills', 'recorder_B_context', 'recorder_reference_handoff', 'inspirations'],
    'generation': ['gonly', 'research_upgrade', 'candidate_revision', 'multi_candidate_l3', 'minimal_fixes', 'feedback_delivery'],
    'runtime': ['breakpoint_contracts', 'multi_candidate_l3', 'step_recovery', 'launch_intent_recovery',
                'numeric_parallel_budget', 'recorder_poll_recovery', 'request_continuation'],
    'evaluation': ['evaluation.test_eval_l0', 'evaluation.test_gonly_rules', 'evaluation.test_breakpoint'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('components', nargs='+', choices=['all', *GROUPS])
    args = parser.parse_args()
    groups = GROUPS if 'all' in args.components else args.components
    modules = dict.fromkeys(name for group in groups for name in GROUPS[group])
    start = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        'tests.' + (name if '.' in name else 'test_' + name) for name in modules])
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(f"components={','.join(groups)} modules={len(modules)} tests={result.testsRun} "
          f"errors={len(result.errors)} failures={len(result.failures)} skips={len(result.skipped)} "
          f"elapsed={time.monotonic()-start:.2f}s", flush=True)
    return int(not result.wasSuccessful())


if __name__ == '__main__':
    sys.exit(main())
