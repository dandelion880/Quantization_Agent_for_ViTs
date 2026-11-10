"""Lossless sources, small objective views, and field-level evidence selection.

Only arithmetic and recorded labels are interpreted here. Metric desirability,
causality, significance, and scientific support belong to the researcher.
"""
from collections import Counter
from copy import deepcopy
from fnmatch import fnmatchcase
import json
import math
import re

from core.cards import canonical, fingerprint


LABELS = {'site', 'site_id', 'module', 'name', 'token', 'tokens', 'token_type',
          'variant', 'arm', 'category', 'split', 'metric'}


def pointer(path, key):
    return path + '/' + str(key).replace('~', '~0').replace('/', '~1')


def leaves(value, field='', labels=None, metric=''):
    """Yield every scalar, preserving its JSON pointer and enclosing row labels."""
    labels = labels or {}
    if isinstance(value, dict):
        labels = {**labels, **{k: v for k, v in value.items()
                              if k in LABELS and isinstance(v, (str, int, float, bool))}}
        for key, child in value.items():
            # Module-keyed tables are rows too; keep their exact site in the label and
            # pointer while grouping the same metric across indexed modules.
            site_key = re.search(r'(?:^|\.)\d+(?:\.|::|$)', key)
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
                site_key = re.search(r'(?:^|\.)\d+(?:\.|::|$)', key)
                visit(child, pointer(path, key), pointer(metric, '*site*' if site_key else key))
            return
        if isinstance(node, list):
            child_metric = pointer(metric, '*')
            for index, child in enumerate(node):
                visit(child, pointer(path, index), child_metric)
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
                  'conclusion', 'explanation', 'meaning', 'notes', 'failure', 'error'}


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
