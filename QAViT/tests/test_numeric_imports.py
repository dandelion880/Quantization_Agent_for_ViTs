"""Useful numerical standard libraries and preserved structural/worker boundaries."""
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import ContractError
from tools.safeguards import inspect_source
from tests.test_infrastructure import TempCase
from tests import test_single_implementation as single

SOURCE = '''from __future__ import annotations
from contextlib import nullcontext
from operator import add
from bisect import bisect_left
import heapq, fractions, decimal, enum, array, struct, json, hashlib, time

def fit(x, parameters):
    with nullcontext():
        return parameters

def apply(x, state, parameters):
    offset = bisect_left([0.0, 1.0], 0.0)
    return add(x, float(offset))
'''


class NumericImports(TempCase):
    begin = single.SingleImplementation.begin
    start = single.SingleImplementation.start
    progress = single.SingleImplementation.progress
    coding = single.SingleImplementation.coding
    approve = single.SingleImplementation.approve

    def test_public_numeric_helpers_across_all_generated_interfaces(self):
        for interface, suffix in [('quantizer_v1', ''), ('executor_v1', '\ndef build_plan(b,c,p): return {}\n'),
                                  ('experiment_v1', '\ndef execute(p,c): return {}\n')]:
            with self.subTest(interface=interface):
                self.assertEqual(inspect_source(SOURCE + suffix, interface)['syntax'], 'passed')
        namespace = {}
        exec(SOURCE, namespace)
        self.assertEqual(namespace['apply'](3, {}, {}), 3.0)

    def test_errors_identify_module_or_operation_without_misclassifying_dependencies(self):
        for module in ('os', 'subprocess', 'socket', 'pathlib', 'ctypes', 'importlib', 'shutil'):
            with self.subTest(module=module), self.assertRaisesRegex(ContractError, module):
                inspect_source('import ' + module + '\ndef execute(p,c): return {}', 'experiment_v1')
        with self.assertRaisesRegex(ContractError, 'fraudulent_dependency.*尚未声明'):
            inspect_source('import fraudulent_dependency\ndef execute(p,c): return {}', 'experiment_v1')
        for statement in ('open("x")', 'eval("1")', 'exec("x=1")', '__import__("os")'):
            with self.subTest(statement=statement), self.assertRaisesRegex(ContractError, '不允许调用'):
                inspect_source('def execute(p,c):\n    return ' + statement, 'experiment_v1')
        for statement in ('from contextlib import os', 'import contextlib as c\nx = c.os',
                          'from .math import sqrt'):
            with self.subTest(statement=statement), self.assertRaises(ContractError):
                inspect_source(statement + '\ndef execute(p,c): return {}', 'experiment_v1')

    def test_regex_compile_is_not_confused_with_python_code_compilation(self):
        for source in ('from re import compile', 'from re import compile as compile_pattern'):
            function = 'compile_pattern' if ' as ' in source else 'compile'
            source += '\ndef execute(p,c): return ' + function + '(r"[0-9]+").fullmatch("42") is not None'
            inspect_source(source, 'experiment_v1')
            namespace = {}
            exec(source, namespace)
            self.assertTrue(namespace['execute']({}, {}))
        for source in ('def execute(p,c): return compile("x=1", "x", "exec")',
                       'from torch import compile\ndef execute(p,c): return {}',
                       'from re import compile\ndef execute(compile,c): return compile("x=1")'):
            with self.subTest(source=source), self.assertRaises(ContractError):
                inspect_source(source, 'experiment_v1')

    def test_helpers_work_in_an_approved_real_cpu_worker(self):
        # Roles scripted; execution, tiny model operations and Landlock are real.
        wf, researcher, request = self.start()
        request = self.approve(wf, self.coding(wf, request, SOURCE))
        trial = wf.tasks()[request['key'] + '/execute']
        wf.dispatch()
        deadline = time.monotonic() + 45
        while wf.queue.get(trial['task_id'])['state'] == 'running' and time.monotonic() < deadline:
            time.sleep(.1)
            wf.poll()
        done = wf.queue.get(trial['task_id'])
        self.assertEqual(done['state'], 'completed', done.get('error'))
        evidence = wf.journal.card(done['outputs'][0])
        self.assertEqual(evidence['payload']['status'], 'completed')
        self.assertEqual(evidence['payload']['measurement_source'], 'synthetic_computation')
        self.assertEqual(self.progress(wf, request)['result']['evidence_ref'], evidence['id'])

    def test_real_landlock_still_blocks_writes_outside_job(self):
        # Install the irreversible kernel restriction in a separate disposable process.
        from runtime.paths import ROOT
        script = '''import sys
from pathlib import Path
from runtime.guards import protect_files
job, outside = map(Path, sys.argv[1:])
protect_files(job)
(job / "allowed.txt").write_text("ok")
try:
    outside.write_text("unexpected")
except PermissionError:
    print("outside_write_blocked")
else:
    raise AssertionError("outside write unexpectedly permitted")
'''
        # TempCase is under the explicitly writable EXP/cache/tmp, so use a separate
        # output root and protected target outside that tree for this kernel check.
        import tempfile
        with tempfile.TemporaryDirectory(dir=ROOT / 'EXP/validation', prefix='landlock-check-') as temporary:
            directory = Path(temporary)
            job = directory / 'job'; job.mkdir()
            outside = directory / 'outside.txt'; outside.write_text('preserved')
            result = subprocess.run([sys.executable, '-B', '-c', script, str(job), str(outside)],
                                    cwd=ROOT / 'QAViT', capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('outside_write_blocked', result.stdout)
            self.assertEqual(outside.read_text(), 'preserved')
