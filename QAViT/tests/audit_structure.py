"""离线结构审计：语法、导入、动态资源、资产定位和源码快照；不执行模型。"""
import argparse
import ast
import importlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.paths import ROOT, QAVIT
from runtime.identity import code_identity, code_snapshot
from core.journal import atomic_json, file_sha256, read_json


def audit():
    files = [p for p in QAVIT.rglob('*.py') if '.runtime_pkgs' not in p.parts and '__pycache__' not in p.parts]
    errors, imports, modules = [], [], []
    for path in sorted(files):
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        compile(tree, str(path), 'exec')  # 不生成 pyc。
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ''
            if node.level:
                target = path.parent
                for _ in range(node.level - 1):
                    target = target.parent
                if module:
                    target = target.joinpath(*module.split('.'))
            elif module.split('.')[0] in ('core', 'runtime', 'agents', 'tools', 'ptq_agent', 'tests'):
                target = (QAVIT / 'tools' if module.startswith('ptq_agent') else QAVIT).joinpath(*module.split('.'))
            else:
                continue
            imports.append({'file': str(path.relative_to(QAVIT)), 'line': node.lineno, 'import': ast.unparse(node)})
            if not target.with_suffix('.py').is_file() and not target.is_dir():
                errors.append(imports[-1])
        # 官方源码有其自身 hubconf 的相对模块约定，按资产清单逐文件校验；不构造大型模型。
        if 'model_sources' in path.parts:
            continue
        name = str(path.relative_to(QAVIT / 'tools' if 'ptq_agent' in path.parts else QAVIT).with_suffix('')).replace('/', '.')
        if name.endswith('.__init__'):
            name = name[:-9]
        try:
            importlib.import_module(name)
            modules.append(name)
        except Exception as exc:
            errors.append({'module': name, 'error': str(exc)})
    from ptq_agent.execution.task_programs import SOURCE_FILES
    dynamic = [QAVIT / 'runtime/task_worker.py',
               QAVIT / 'tools/experiment_worker.py', QAVIT / 'tools/vision_models.py',
               QAVIT / 'tools/ptq_agent/execution/program_worker.py', QAVIT / 'tests/fixtures.py']
    dynamic += [QAVIT / 'tools/ptq_agent' / p for p in SOURCE_FILES]
    dynamic += list((QAVIT / 'agents/prompts').rglob('*.md'))
    for path in dynamic:
        if not path.is_file():
            errors.append({'missing_dynamic_resource': str(path)})
    from agents.tasks import TASKS
    from tools.experiments import SERVICES
    from tools.safeguards import INTERFACES
    from runtime.configuration import config_check
    from tests.support import validation_config
    config = config_check(validation_config())
    factory, function = config['research_brief']['model']['adapter']['factory'].split(':')
    spec = importlib.util.spec_from_file_location('qavit_audited_factory', ROOT / factory)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    assert callable(getattr(fixture, function))
    from ptq_agent.configuration import load_config
    numerical = load_config(QAVIT / 'configs/deitb_backend.json')
    assets = []
    for key in ('model_path', 'data_path', 'asset_manifest'):
        path = Path(numerical['backend'][key])
        assets.append({'kind': key, 'path': str(path), 'exists': path.is_file(),
                       'bytes': path.stat().st_size if path.is_file() else None})
        if not path.is_file():
            errors.append({'missing_asset': str(path)})
    data = Path(numerical['backend']['data_path'])
    if data.is_file() and file_sha256(data) != numerical['backend']['data_sha256']:
        errors.append({'asset_hash_mismatch': str(data)})
    manifest_path = Path(numerical['backend']['asset_manifest'])
    for model in read_json(manifest_path)['models']:
        for relative, expected in model['code_source']['source_file_sha256'].items():
            path = (manifest_path.parent / relative).resolve()
            if not path.is_file() or file_sha256(path) != expected:
                errors.append({'model_source_hash_mismatch': str(path)})
    ade = read_json(QAVIT / 'configs/ade_evaluation.json')
    head = read_json(QAVIT / 'configs' / ade['head_receipt'])
    for name, path in [('ade_dataset', Path(ade['data_root'])), ('ade_head', Path(head['checkpoint']))]:
        assets.append({'kind': name, 'path': str(path), 'exists': path.exists()})
        if not path.exists():
            errors.append({'missing_asset': str(path)})
    identity = code_identity()
    snapshot = code_snapshot(identity)
    assert 'runtime/task_worker.py' in snapshot and 'agents/prompts/tasks/generate.md' in snapshot
    assert 'tools/model_sources/deit/models.py' in snapshot
    return {'status': 'passed' if not errors else 'failed', 'errors': errors,
            'syntax_files': len(files), 'imported_modules': modules, 'local_imports': imports,
            'dynamic_resources': [str(p.relative_to(QAVIT)) for p in dynamic], 'assets': assets,
            'purposes': list(TASKS), 'services': list(SERVICES), 'code_interfaces': list(INTERFACES),
            'source_identity': identity, 'snapshot_file_count': len(snapshot),
            'limits': ['No model instantiated; checkpoint tensor loading not verified.',
                       'No real LLM/GPU/full evaluation/network requests.',
                       'Dataset/head existence and model constructor hashes checked; images not evaluated.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if ROOT / 'EXP' not in output.parents or output.exists():
        parser.error('输出必须为 EXP 下的新文件')
    result = audit()
    atomic_json(output, result)
    print(json.dumps({k: result[k] for k in ('status', 'errors', 'syntax_files', 'snapshot_file_count')}, ensure_ascii=False))
    sys.exit(0 if result['status'] == 'passed' else 1)
