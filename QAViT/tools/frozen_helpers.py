"""Thin request adapters over the existing frozen numerical helpers.

Loading, source verification, observed reads and Evidence remain host owned.
Scientific decisions and publication of a new child belong to separate calls.
"""
from copy import deepcopy
from core.cards import ContractError

HELPER_NAMES = ('profile_frozen_propagation', 'search_frozen_scales', 'score_frozen_candidates')


def normalize_helper(execution):
    from core.execution_contract import merge
    name, args = execution.get('tool'), execution.get('arguments')
    if name not in HELPER_NAMES or not isinstance(args, dict):
        raise ContractError('frozen_helper requires tool and arguments: ' + ', '.join(HELPER_NAMES))
    allowed = {
        'profile_frozen_propagation': {'blocks', 'interventions', 'token_layout', 'classification_output', 'view', 'output'},
        'search_frozen_scales': {'candidates', 'metric', 'block', 'reference', 'context', 'token_scope', 'cache_mib'},
        'score_frozen_candidates': {'candidates', 'observations_spec', 'objectives'},
    }[name]
    if set(args) - allowed - {'split', 'count', 'offset'}:
        raise ContractError('unsupported frozen_helper arguments: ' + str(sorted(set(args) - allowed - {'split', 'count', 'offset'})))
    if execution.get('parameters') or execution.get('evaluations'):
        raise ContractError('frozen_helper consumes arguments and returns diagnostics, not formal evaluations')
    if 'source_ref' in execution and (not isinstance(execution['source_ref'], str) or not execution['source_ref'].strip()):
        raise ContractError('source_ref must be a nonempty Evidence/solution/model identifier')
    split = args.get('split', execution.get('evaluation', execution.get('calibration', {})).get('split', 'calibration'))
    if split not in ('calibration', 'search') or (name == HELPER_NAMES[1] and split != 'calibration'):
        raise ContractError('frozen propagation accepts calibration/search; scale selection accepts calibration only')
    if name == 'score_frozen_candidates':
        from ptq_agent.execution.frozen_trials import _validate
        from ptq_agent.schemas import PTQError
        try:
            _validate(args.get('observations_spec', []))
        except (KeyError, TypeError, PTQError) as exc:
            raise ContractError('invalid observations_spec: ' + str(exc)) from exc
        if split == 'search' and args.get('objectives'):
            raise ContractError('use calibration for selection; confirm declared candidates on search')
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
        elif spec['tool'] == 'score_frozen_candidates':
            from ptq_agent.execution.frozen_trials import score_frozen_candidates
            report = score_frozen_candidates(context['backend'], model, context['data'], split=split,
                source_ref=source['model_ref'], input_shapes=source['manifest'].get('input_shapes'), **args)
            report['source'] = {k: v for k, v in source.items() if k != 'manifest'}
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
    kind = {'profile_frozen_propagation': 'frozen_propagation', 'search_frozen_scales': 'calibration_scale_selection',
            'score_frozen_candidates': 'paired_frozen_candidate_scores'}
    result['execution_facts']['measurement_kind'] = kind[spec['tool']]
    report = result['observations']['result']
    result['execution_facts']['parameters'] = {'tool': spec['tool'], 'arguments': deepcopy(spec['arguments']),
                                              'source': deepcopy(report['source'])}
    result['execution_facts']['sites'] = (list(report['boundaries']) if spec['tool'] == HELPER_NAMES[0]
        else sorted({site for row in report['candidates'] for site in row.get('receipt', row)['scales']}))
    return result
