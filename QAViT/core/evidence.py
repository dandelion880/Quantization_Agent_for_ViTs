"""Lossless sources, small objective views, and field-level evidence selection.

Only arithmetic and recorded labels are interpreted here. Metric desirability,
causality, significance, and scientific support belong to the researcher.
"""
from collections import Counter
from copy import deepcopy
from fnmatch import fnmatchcase
import json
import math
import hashlib
from pathlib import Path
import re

from core.cards import canonical, fingerprint


ARM_LABELS = ('arm', 'arm_id', 'arm_name', 'variant', 'variant_id', 'method', 'condition', 'treatment')
ARM_COLLECTIONS = {'arms', 'arm_results', 'variants', 'treatments'}

LABELS = {'site', 'site_id', 'module', 'name', 'token', 'tokens', 'token_type',
          'category', 'split', 'metric', *ARM_LABELS}


def arm_labels(value):
    """Recorded identities only; structured labels retain an exact content digest."""
    return {k: ({'sha256': fingerprint(value[k])} if isinstance(value[k], (dict, list)) else value[k])
            for k in ARM_LABELS if isinstance(value, dict) and k in value}


def pointer(path, key):
    return path + '/' + str(key).replace('~', '~0').replace('/', '~1')


def leaves(value, field='', labels=None, metric=''):
    """Yield every scalar, preserving its JSON pointer and enclosing row labels."""
    labels = labels or {}
    if isinstance(value, dict):
        labels = {**labels, **{k: v for k, v in value.items()
                              if k in LABELS and isinstance(v, (str, int, float, bool))},
                  **arm_labels(value)}
        for key, child in value.items():
            # Module-keyed tables are rows too; keep their exact site in the label and
            # pointer while grouping the same metric across indexed modules.
            site_key = field.rsplit('/', 1)[-1] not in ARM_COLLECTIONS and re.search(r'(?:^|\.)\d+(?:\.|::|$)', key)
            yield from leaves(child, pointer(field, key), {**labels, 'site': key} if site_key else labels,
                              pointer(metric, '*site*' if site_key else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from leaves(child, pointer(field, index), labels, pointer(metric, '*'))
    else:
        yield {'field': field, 'metric': metric, 'value': value, 'labels': labels}


def selected_leaves(value, field, select):
    """All filters are conjunctive; glob patterns are explicit, never fuzzy inference."""
    for row in leaves(value, field):
        path = row['field']
        segments = [s.replace('~1', '/').replace('~0', '~') for s in path.split('/')[1:]]
        labels = row['labels']
        if 'site' in select and not any(fnmatchcase(str(v), select['site']) for v in
                [*segments, *(labels[k] for k in ('site', 'site_id', 'module', 'name') if k in labels)]):
            continue
        if 'token_type' in select and not any(fnmatchcase(str(v), select['token_type']) for v in
                [*segments, *(labels[k] for k in ('token', 'tokens', 'token_type') if k in labels)]):
            continue
        if 'metric' in select and not any(fnmatchcase(m, select['metric']) for m in
                (segments[-1] if segments else '', row['metric'].lstrip('/'), str(labels.get('metric', '')))):
            continue
        if 'text' in select and select['text'].casefold() not in canonical(row).casefold():
            continue
        if 'minimum' in select or 'maximum' in select:
            if type(row['value']) not in (int, float):
                continue
            if row['value'] < select.get('minimum', -math.inf) or row['value'] > select.get('maximum', math.inf):
                continue
        yield row


def collection_summary(value, ref, field, *, max_characters=6000, focus=None):
    """Streaming statistics over ALL entries, including tail extrema and sign conflicts."""
    numbers, categories = {}, {}
    def visit(node, path, metric):
        # Summary arithmetic needs no row labels or per-leaf dictionaries. Keep the
        # same depth-first order (including floating-point recurrence and tie order).
        if isinstance(node, dict):
            for key, child in node.items():
                site_key = path.rsplit('/', 1)[-1] not in ARM_COLLECTIONS and re.search(r'(?:^|\.)\d+(?:\.|::|$)', key)
                visit(child, pointer(path, key), pointer(metric, '*site*' if site_key else key))
            return
        if isinstance(node, list):
            child_metric = pointer(metric, '*')
            for index, child in enumerate(node):
                labels = arm_labels(child)
                # Arms are distinct measurements, never exchangeable repetitions.
                # An explicitly arm-keyed collection needs no guessed identity: its
                # exact index remains distinct even when a producer omitted labels.
                group_metric = (pointer(metric, index) if path.rsplit('/', 1)[-1] in ARM_COLLECTIONS
                                else child_metric + ('[' + canonical(labels) + ']' if labels else ''))
                visit(child, pointer(path, index), group_metric)
            return
        scalar = node
        if type(scalar) in (int, float):
            group = numbers.get(metric)
            if group is None:
                group = numbers[metric] = {'count': 0, 'mean': 0.0, 'negative': 0, 'zero': 0,
                    'minimum': {'value': scalar, 'field': path},
                    'maximum': {'value': scalar, 'field': path}}
            group['count'] += 1
            # Weighted recurrence avoids overflowing the sum of large finite values.
            n = group['count']
            group['mean'] = group['mean'] * ((n - 1) / n) + scalar / n
            group['negative'] += scalar < 0
            group['zero'] += scalar == 0
            if scalar < group['minimum']['value']:
                group['minimum'] = {'value': scalar, 'field': path}
            if scalar > group['maximum']['value']:
                group['maximum'] = {'value': scalar, 'field': path}
        else:
            group = categories.get(metric)
            if group is None:
                group = categories[metric] = {'count': 0, 'values': Counter(), 'all_values_shown': True}
            group['count'] += 1
            key = canonical(scalar)
            if len(key) <= 200 and (key in group['values'] or len(group['values']) < 8):
                group['values'][key] += 1
            else:
                group['all_values_shown'] = False
    visit(value, field, '')
    # A bounded categorical preview is explicitly partial; exact values are queryable.
    for group in categories.values():
        group['values'] = dict(group['values'])
    # A summary is also a presentation, not an exemption from context budgeting.
    # Rank complete metric groups, never truncate rows or average unlike metrics.
    focus = focus or {}
    def priority(item):
        metric, group = item
        requested = (('metric' in focus and any(fnmatchcase(m, focus['metric']) for m in (metric.lstrip('/'), metric.rsplit('/', 1)[-1])))
                     or ('text' in focus and focus['text'].casefold() in metric.casefold()))
        conflict = group.get('negative', 0) > 0 and group.get('maximum', {}).get('value', 0) > 0
        return (not requested, not conflict, metric)
    result = {'read_card': {'ref': ref, 'field': field}, 'sha256': fingerprint(value),
              'items': len(value), 'numeric_summary': {}, 'categorical_summary': {},
              'summary_basis': 'all_entries; unweighted_descriptive_mean; immutable_source'}
    available = max(600, max_characters - len(canonical(result)) - 650)
    for name, groups in (('numeric_summary', numbers), ('categorical_summary', categories)):
        for metric, group in sorted(groups.items(), key=priority):
            cost = len(canonical({metric: group}))
            if cost <= available:
                result[name][metric] = group
                available -= cost
    omitted = len(numbers) + len(categories) - len(result['numeric_summary']) - len(result['categorical_summary'])
    if omitted:
        result['display_coverage'] = {
            'numeric_metrics_total': len(numbers), 'categorical_metrics_total': len(categories),
            'omitted_metric_groups': omitted,
            'mixed_sign_metrics_total': sum(g['negative'] > 0 and g['maximum']['value'] > 0 for g in numbers.values()),
            'omitted_mixed_sign_metrics': sum(g['negative'] > 0 and g['maximum']['value'] > 0
                for key, g in numbers.items() if key not in result['numeric_summary']),
            'selection': 'explicit metric/text focus, then mixed-sign groups, then metric name; no scientific ranking',
            'expand': {'read_card': {'ref': ref, 'field': field, 'select': {'metric': '*'}}},
            'meaning': 'Displayed groups describe all their rows; undisplayed metrics may contain contrary evidence. '
                       'Use metric/site/token filters or exact field reads; no scientific conclusion is inferred.'}
    return result


CONTEXT_FIELDS = {'summary', 'status', 'conditions', 'comparison', 'scope', 'limitations',
                  'limitation', 'warnings', 'warning', 'counterevidence', 'contradictions',
                  'conclusion', 'explanation', 'meaning', 'notes', 'failure', 'error',
                  'controls', 'metric_definitions', 'scope_limitations', 'whole_network',
                  'paired_classification_counts', 'paired_sign_counts'}


def prose_view(value, ref, field, allowance):
    """Retain complete short prose; long prose has a complete opening paragraph and source."""
    if not isinstance(value, str) or len(value) <= allowance:
        return deepcopy(value)
    paragraphs = value.split('\n\n')
    opening = paragraphs[0] if len(paragraphs[0]) <= min(1200, allowance) else None
    return {'read_card': {'ref': ref, 'field': field}, 'sha256': fingerprint(value),
            'characters': len(value), 'complete': False,
            **({'opening_paragraph': opening} if opening else {}),
            'meaning': 'Long scientific text remains in the source; opening paragraph is not a summary.'}


def _bounded_json_size(value, limit):
    """Exact below the threshold; stop encoding once the display decision is known.

    Input is already validated JSON from the immutable card. Full source hashing in
    collection_summary remains unchanged; this only avoids serializing large trees
    repeatedly to decide whether they fit in a small display.
    """
    size = 0
    for chunk in json.JSONEncoder(ensure_ascii=False, sort_keys=True,
                                 separators=(',', ':'), allow_nan=False).iterencode(value):
        size += len(chunk)
        if size > limit:
            break
    return size


def project_observations(value, ref, field='/payload/observations', *, max_characters=6000, focus=None):
    """Budget deterministic metric views while leaving immutable data precisely readable.

    Scalar results and small scientific summaries stay intact. Wide metric dictionaries
    and large arrays receive bounded groups with coverage and contrary-sign notices.
    The budget governs display only and is never a submission requirement.
    """
    if not isinstance(value, (dict, list)):
        return prose_view(value, ref, field, max(12000, max_characters))
    if isinstance(value, list) and (field.rsplit('/', 1)[-1].endswith('sample_ids') or field.rsplit('/', 1)[-1] == 'split_ids'):
        return sample_list_view(value, ref, field)
    if isinstance(value, dict) and 'read_card' not in value:
        # Compact identity lists before deciding whether scientific metrics fit.
        compact = compact_sample_lists(value, ref, field)
        if compact != value and _bounded_json_size(compact, max_characters) <= max_characters:
            return compact
    size = _bounded_json_size(value, max(4000, max_characters))
    if size <= min(4000, max_characters):
        return deepcopy(value)

    def bounded_table():
        # Keep recorded scientific context outside the metric preview; it must not
        # disappear among categorical frequencies or become an inferred conclusion.
        context = {k: (prose_view(v, ref, pointer(field, k), max(700, max_characters // 3))
                       if not isinstance(v, (dict, list)) else
                       project_observations(v, ref, pointer(field, k),
                           max_characters=max(900, max_characters // 3), focus=focus))
                   for k, v in value.items() if k in CONTEXT_FIELDS or
                       (len(value) <= 24 and not isinstance(v, (dict, list)))} if isinstance(value, dict) else {}
        room = max(900, max_characters - len(canonical(context)))
        result = collection_summary(value, ref, field, max_characters=room, focus=focus)
        if context:
            result['recorded_context'] = context
        return result

    large_table = (isinstance(value, list) or len(value) > 24 or
        sum(bool(re.search(r'(?:^|\.)\d+(?:\.|::|$)', k)) for k in value) >= 8)
    if large_table and size > max(4000, max_characters):
        return bounded_table()
    if isinstance(value, dict):
        large = [k for k, v in value.items() if isinstance(v, (dict, list)) and _bounded_json_size(v, 4000) > 4000]
        child_budget = max(900, max_characters // max(1, len(large)))
        result = {k: project_observations(v, ref, pointer(field, k),
                    max_characters=child_budget if k in large else max_characters, focus=focus)
                  for k, v in value.items()}
        # Many individually small tables can still overflow the combined view.
        # Re-project the original as complete metric groups rather than slicing it.
        if len(canonical(result)) > max_characters * 2:
            return bounded_table()
        return result
    return [project_observations(v, ref, pointer(field, i), max_characters=max_characters, focus=focus)
            for i, v in enumerate(value)]


def sample_list_view(value, ref, field):
    """Exact ordered identity digest; IDs are not categorical scientific observations."""
    if len(canonical(value)) <= 1000:
        return deepcopy(value)
    return {'read_card': {'ref': ref, 'field': field}, 'sha256': fingerprint(value),
            'items': len(value), 'ordered': True, 'complete': False,
            'meaning': 'Exact ordered sample IDs remain in the immutable source; no sampled preview.'}


def compact_sample_lists(value, ref, field='', sample=False):
    if isinstance(value, list):
        if sample:
            return sample_list_view(value, ref, field)
        return [compact_sample_lists(v, ref, pointer(field, i)) for i, v in enumerate(value)]
    if isinstance(value, dict):
        if 'read_card' in value:
            return deepcopy(value)
        return {k: compact_sample_lists(v, ref, pointer(field, k),
                    k.endswith('sample_ids') or field.rsplit('/', 1)[-1] in ('sample_selection', 'split_ids'))
                for k, v in value.items()}
    return deepcopy(value)


def evidence_facts(card, root=None):
    """Small shared handoff for Generation, Recorder and Memory, with immutable sources.

    Legacy ADE paired metrics may be recovered only from an already bound, hash-checked
    artifact in this run. It is a view of historical measurement, never a new Evidence.
    No metric is inferred from prose or from a different execution/arm.
    """
    payload, ref = card['payload'], card['id']
    provenance = payload.get('provenance', {})
    observations = payload.get('observations', {})
    result = {'status': payload.get('status'), 'measurement_source': payload.get('measurement_source'),
              'execution_id': payload.get('execution_id'), 'experiment_ref': payload.get('experiment_ref'),
              'protocol_hash': provenance.get('protocol_hash'),
              'source': {'read_card': {'ref': ref}, 'sha256': card.get('content_sha256')},
              'identity': {k: deepcopy(provenance[k]) for k in
                  ('baseline_state_sha256', 'model_ref', 'model_unchanged') if k in provenance},
              'evaluations': {}}
    trial = observations.get('result', {})
    if (observations.get('tool') == 'profile_frozen_propagation' and isinstance(trial, dict)
            and all(isinstance(trial.get(k), dict) for k in ('source','boundaries','interventions','semantics'))
            and {'baseline_output_error','numeric_state_sha256','split','sample_count'} <= trial.keys()):
        args = provenance.get('execution_facts', {}).get('parameters', {}).get('arguments', {})
        blocks = args.get('blocks', [])
        endpoint = blocks[-1] if blocks else None
        def token_nmse(error):
            return {group:row['relative_mse'] for group,row in error.get('outputs', {}).items()
                    if 'relative_mse' in row}
        def table(name, body):
            handle = {'ref':ref,'field':'/payload/observations/result/' + name}
            return body if len(canonical(body)) <= 4000 else {'read_card':handle,'count':len(body)}
        boundary = {name:{key:token_nmse(row[key]) for key in ('input_total','local_added','upstream_mapped','output_total')
                          if key in row} for name,row in trial['boundaries'].items()}
        arms = {}
        for name, row in trial['interventions'].items():
            arms[name] = {'intervention':row['intervention'], 'output_nmse':token_nmse(row['output_error']),
                          'last_requested_block_nmse':token_nmse(row['boundaries'].get(endpoint, {}).get('error_to_fp', {}))}
        baseline = trial['boundaries'].get(endpoint, {}).get('output_total', {})
        result['propagation'] = {'parent_model_ref':trial['source'].get('model_ref'),
            'state':trial['numeric_state_sha256'],'split':trial['split'],'originals':trial['sample_count'],
            'view':trial.get('view','classification'),'endpoint':trial.get('endpoint',args.get('classification_output')),
            'last_requested_block':endpoint,'baseline_at_last_block_nmse':token_nmse(baseline),
            'boundary_nmse':table('boundaries',boundary),'arms':table('interventions',arms),
            'classification':table('classification_diagnostic', {name:{k:r[k] for k in ('teacher_kl','top1','correct','samples') if k in r}
                              for name,r in trial.get('classification_diagnostic',{}).items()}),
            'semantics':trial['semantics'],'cross_term':{'read_card':{'ref':ref,'field':'/payload/observations/result/decomposition_cross_term'}},
            'read_card':{'ref':ref,'field':'/payload/observations/result'}}
    if (isinstance(trial, dict) and trial.get('api') == 'frozen_candidate_scores/1'
            and all(isinstance(trial.get(k), list) for k in ('sample_ids','views','candidates'))
            and all(isinstance(r, dict) and {'id','scores','objectives'} <= r.keys() for r in trial['candidates'])):
        field = '/payload/observations/result'
        result['frozen_trials'] = {k: deepcopy(trial[k]) for k in ('parent_state_sha256','parent_model_ref',
            'split','reduction','winners','parameters_installed') if k in trial}
        result['frozen_trials'].update(originals=len(trial['sample_ids']), view_units=len(trial['views']),
            candidates=len(trial['candidates']), read_card={'ref':ref,'field':field},
            scores_source={'read_card': {'ref':ref,'field':field+'/candidates'}})
        pool = [{k:r[k] for k in ('id','scores','objectives')} for r in trial['candidates']]
        if len(canonical(pool)) <= 2600:
            result['frozen_trials']['pool'] = pool
    if isinstance(observations.get('refinement'), dict):
        r = observations['refinement']
        result['refinement'] = {k:r[k] for k in ('parent_model_ref','parent_state_sha256','child_state_sha256',
            'initialization','selection_source','candidate_sha256') if k in r}
        result['refinement']['read_card'] = {'ref':ref, 'field':'/payload/observations/refinement'}
    for key in ('model_identity', 'data_identity'):
        if key in provenance:
            original = provenance[key]
            result['identity'][key] = {'sha256': fingerprint(original),
                'read_card': {'ref': ref, 'field': '/payload/provenance/' + key}}
    scope = payload.get('scope', {})
    result['conditions'] = {k: compact_sample_lists(scope[k], ref, '/payload/scope/' + k)
        for k in ('model', 'model_identity_sha256', 'measurement_kind', 'weight_bits', 'activation_bits', 'data') if k in scope}
    for key in ('reload', 'status', 'failure', 'error', 'counterevidence', 'limitations', 'warnings'):
        if key in observations:
            result[key] = project_observations(observations[key], ref, '/payload/observations/' + key)
        elif key in payload:
            result[key] = deepcopy(payload[key])
    for key in ('controls', 'metric_definitions', 'limitations', 'scope_limitations', 'counterevidence',
                'paired_classification_counts', 'paired_sign_counts'):
        if key in observations:
            result[key] = project_observations(observations[key], ref, pointer('/payload/observations', key))
    if isinstance(observations.get('arms'), dict):
        result['arm_results'] = {}
        for name, arm in observations['arms'].items():
            if not isinstance(arm, dict):
                continue
            arm_path = pointer('/payload/observations/arms', name)
            result['arm_results'][name] = {
                key: project_observations(value, ref, pointer(arm_path, key), max_characters=1600)
                for key, value in arm.items() if key in (*ARM_LABELS, 'whole_network', 'metrics', 'summary', 'status', 'error')
                or key.startswith('paired_difference_vs_')}
    elif isinstance(observations.get('arms'), list):
        result['arm_results'] = project_observations(observations['arms'], ref, '/payload/observations/arms')
    if 'arms' in observations:
        result['arm_source'] = {'read_card': {'ref': ref, 'field': '/payload/observations/arms'},
                                'meaning': 'Distinct arms within one execution; not independent repetitions.'}
    evaluations = observations.get('evaluations', {})
    if not isinstance(evaluations, dict):
        result['evaluation_source'] = {'read_card': {'ref': ref, 'field': '/payload/observations/evaluations'},
                                      'meaning': 'Legacy non-keyed evaluations require exact reading; arm ownership not inferred.'}
        return result
    for name, row in evaluations.items():
        if not isinstance(row, dict):
            continue
        field = pointer('/payload/observations/evaluations', name)
        metrics = deepcopy(row.get('metrics', {}))
        sources = {'read_card': {'ref': ref, 'field': field + '/metrics'}}
        legacy_source = None
        if name == 'segmentation' and not any(k.startswith('baseline_') for k in metrics) and root:
            for receipt in payload.get('artifacts', []):
                path = Path(receipt.get('path', ''))
                if path.name != 'segmentation.json':
                    continue
                try:
                    path = path.resolve()
                    if not path.is_relative_to(Path(root).resolve()):
                        continue
                    raw = path.read_bytes()
                    if hashlib.sha256(raw).hexdigest() != receipt.get('sha256'):
                        raise ValueError('artifact hash mismatch')
                    original = json.loads(raw)
                    quant = original.get('metrics', {}).get('quantized', original.get('metrics', {}).get('quant'))
                    fp = original.get('metrics', {}).get('float', original.get('metrics', {}).get('fp'))
                    if (original.get('sample_ids') != row.get('sample_ids') or not isinstance(fp, dict)
                            or not isinstance(quant, dict) or any(quant.get(k) != metrics[k] for k in ('mIoU', 'mAcc', 'aAcc') if k in metrics)):
                        raise ValueError('artifact scope/metric mismatch')
                    metrics.update({'baseline_' + k: fp[k] for k in ('mIoU', 'mAcc', 'aAcc') if k in fp})
                    legacy_source = {**receipt, 'json_pointer': '/metrics', 'basis': 'historical_artifact_projection; not_new_measurement'}
                except (OSError, ValueError, TypeError) as exc:
                    result.setdefault('source_issues', []).append({'path': str(path), 'reason': str(exc)})
                break
        pairs = {}
        for metric in (('top1', 'top5') if name == 'classification' else ('mIoU', 'mAcc', 'aAcc') if name == 'segmentation' else ()):
            q, fp = metrics.get(metric), metrics.get('baseline_' + metric)
            if q is None and fp is None:
                continue
            numeric = type(q) in (int, float) and type(fp) in (int, float)
            pairs[metric] = {'FP': fp, 'Q': q, 'unit': 'percent',
                'delta_Q_minus_FP': q - fp if numeric else None, 'delta_unit': 'percentage_points',
                'retention_percent': 100*q/fp if numeric and fp else None,
                'pair_status': 'paired' if numeric else 'missing_FP_or_Q'}
        result['evaluations'][name] = {**{k: deepcopy(row[k]) for k in
            ('split', 'sample_count', 'full_dataset', 'complete_view', 'purpose', 'development_sha256', 'metric_units', 'status', 'error') if k in row},
            'sample_ids': sample_list_view(row['sample_ids'], ref, field + '/sample_ids') if 'sample_ids' in row else None,
            'metrics': metrics, 'paired_metrics': pairs, 'source': sources,
            **({'legacy_paired_source': legacy_source} if legacy_source else {})}
    return result
