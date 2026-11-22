"""Thin request adapters over the existing frozen numerical helpers.

Loading, source verification, observed reads and Evidence remain host owned.
The search changes only this job's loaded instance, never the cited package.
"""
from copy import deepcopy
from core.cards import ContractError

HELPER_NAMES = ('profile_frozen_propagation', 'search_frozen_scales')


def normalize_helper(execution):
    from core.execution_contract import merge
    name, args = execution.get('tool'), execution.get('arguments')
    if name not in HELPER_NAMES or not isinstance(args, dict):
        raise ContractError('frozen_helper requires tool and arguments: ' + ', '.join(HELPER_NAMES))
    allowed = ({'blocks', 'interventions', 'token_layout', 'classification_output'}
               if name == HELPER_NAMES[0] else
               {'candidates', 'metric', 'block', 'reference', 'context', 'token_scope', 'cache_mib'})
    if set(args) - allowed - {'split', 'count', 'offset'}:
        raise ContractError('unsupported frozen_helper arguments: ' + str(sorted(set(args) - allowed - {'split', 'count', 'offset'})))
    if execution.get('parameters') or execution.get('evaluations'):
        raise ContractError('frozen_helper consumes arguments and returns diagnostics, not formal evaluations')
    if 'source_ref' in execution and (not isinstance(execution['source_ref'], str) or not execution['source_ref'].strip()):
        raise ContractError('source_ref must be a nonempty Evidence/solution/model identifier')
    split = args.get('split', execution.get('evaluation', execution.get('calibration', {})).get('split', 'calibration'))
    if split not in ('calibration', 'search') or (name == HELPER_NAMES[1] and split != 'calibration'):
        raise ContractError('frozen propagation accepts calibration/search; scale selection accepts calibration only')
    key = 'calibration' if split == 'calibration' else 'evaluation'
    other = 'evaluation' if key == 'calibration' else 'calibration'
    if other in execution:
        raise ContractError('frozen_helper reads one explicit view per request')
    selection = execution.setdefault(key, {})
    merge(selection, {'split': split, **{k: args[k] for k in ('count', 'offset') if k in args}}, 'execution.' + key)
    if split == 'search' and 'count' not in selection:
        raise ContractError('search diagnostics need an explicit count (null means complete search view)')
    if 'count' in selection and selection['count'] is not None and (type(selection['count']) is not int or selection['count'] <= 0):
        raise ContractError('frozen_helper count must be positive or null')
    if type(selection.get('offset', 0)) is not int or selection.get('offset', 0) < 0:
        raise ContractError('frozen_helper offset must be nonnegative')
    args['split'] = split


class FrozenHelper:
    @staticmethod
    def execute(payload, context):
        spec = payload['execution']
        model, source = context['load_frozen'](spec.get('source_ref'))
        args = deepcopy(spec['arguments'])
        split = args.pop('split', 'calibration')
        for key in ('count', 'offset'):
            args.pop(key, None)  # The host has already restricted the data view.
        if spec['tool'] == 'profile_frozen_propagation':
            from ptq_agent.execution.propagation import profile_frozen_propagation
            report = profile_frozen_propagation(context['backend'], model, context['data'],
                split=split, source={k: v for k, v in source.items() if k != 'manifest'}, **args)
        else:
            from ptq_agent.execution.calibration_search import search_frozen_scales
            report = search_frozen_scales(context['backend'], model, context['data'], **args)
            report['source'] = {k: v for k, v in source.items() if k != 'manifest'}
        return {'observations': {'tool': spec['tool'], 'result': report,
            'state_scope': 'job-local loaded instance; cited package unchanged',
            'selected_package_exported': False}}


def frozen_helper_service(experiment, backend, cfg, data, directory):
    from tools.runtime_interface import execute_custom
    spec = experiment['payload']['execution']
    result = execute_custom(FrozenHelper, experiment['payload'], backend, cfg, data, directory)
    result['execution_facts']['measurement_kind'] = ('frozen_propagation' if spec['tool'] == HELPER_NAMES[0]
                                                     else 'calibration_scale_selection')
    report = result['observations']['result']
    result['execution_facts']['parameters'] = {'tool': spec['tool'], 'arguments': deepcopy(spec['arguments']),
                                              'source': deepcopy(report['source'])}
    result['execution_facts']['sites'] = (list(report['boundaries']) if spec['tool'] == HELPER_NAMES[0]
        else sorted({site for row in report['candidates'] for site in row['scales']}))
    return result
