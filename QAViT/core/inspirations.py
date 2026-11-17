"""Run-local idea library, independent of complete solutions and empirical rank.

Journal is the sole store. Entries contain short notes and references; recovery
projects existing immutable requests, code artifacts and results, never copies a
complete solution or schedules an experiment. All reads honor a commit cutoff.
"""
from copy import deepcopy
from core.cards import ContractError, TEXT, STRINGS, object_schema, fingerprint


UPDATE_SCHEMA = object_schema({
    'inspiration_id': TEXT, 'title': TEXT, 'idea': TEXT, 'basis': TEXT,
    'progress': TEXT, 'open_questions': STRINGS, 'next_step': TEXT,
    'status': {'enum': ['active', 'paused']}, 'reason': TEXT,
    'source_refs': STRINGS,
    'absorbed_parts': {'type': 'array', 'items': object_schema({
        'part': TEXT, 'target_ref': TEXT}, ['part', 'target_ref'])},
}, [])
UPDATE_SCHEMA['additionalProperties'] = False


class InspirationLibrary:
    def __init__(self, journal, solutions):
        self.journal, self.solutions = journal, solutions
        self._source_bindings = {}

    def events(self, sequence=None):
        return [e for e in self.journal.events if sequence is None or e['commit_sequence'] <= sequence]

    def snapshot(self, sequence=None):
        rows = {}
        for event in self.events(sequence):
            if event['type'] == 'inspiration_updated':
                rows[event['entry']['id']] = deepcopy(event['entry'])
        return rows

    def active(self, sequence=None):
        return next((r for r in self.snapshot(sequence).values() if r['status'] == 'active'), None)

    def bound_id(self, task, sequence=None):
        context = task['inputs'].get('context', {})
        # L3 must retain its exact candidate even if Generation switches directions.
        if context.get('hypothesis_ref'):
            for event in reversed(self.events(sequence)):
                if event['type'] == 'inspiration_candidate' and event['hypothesis_ref'] == context['hypothesis_ref']:
                    return event['inspiration_id']
        for event in reversed(self.events(sequence)):
            if event['type'] == 'inspiration_task' and event['task_id'] == task['task_id']:
                return event['inspiration_id']
        return context.get('inspiration_id')

    def update_events(self, args, task, *, key, sequence=None):
        rows, current = self.snapshot(sequence), self.snapshot()
        identifier = args.get('inspiration_id') or 'idea-' + fingerprint(key)[:20]
        old = rows.get(identifier)
        if args.get('inspiration_id') and not old:
            raise ContractError('未知或不可见的灵感 ID')
        if old != current.get(identifier):
            raise ContractError('灵感已有新修订，请读取当前进展后更新')
        recorder = task['role'] == 'recorder'
        if recorder and (not old or set(args) - {'inspiration_id', 'progress', 'open_questions', 'next_step',
                                                 'source_refs', 'absorbed_parts'}):
            raise ContractError('Recorder 只更新已有灵感进展与引用；继续/暂停/切换由 Generation 决定')
        if not old and not all(args.get(k) for k in ('title', 'idea', 'basis', 'next_step')):
            raise ContractError('新灵感需要 title、idea、basis、next_step；无需完整方案或实测')
        row = deepcopy(old) if old else {'id': identifier, 'revision': 0, 'status': 'paused',
            'progress': '尚未实施', 'open_questions': [], 'source_refs': [], 'absorbed_parts': []}
        if 'status' in args and args['status'] != row['status'] and not args.get('reason'):
            raise ContractError('激活/暂停/恢复/切换需要具体 reason')
        for field in ('title', 'idea', 'basis', 'progress', 'open_questions', 'next_step', 'status', 'reason'):
            if field in args:
                row[field] = deepcopy(args[field])
        row['source_refs'] = list(dict.fromkeys(row['source_refs'] + args.get('source_refs', [])))
        for part in args.get('absorbed_parts', []):
            if part not in row['absorbed_parts']:
                row['absorbed_parts'].append(deepcopy(part))
        row.update(revision=row['revision'] + 1, updated_by=task['role'], update_task_ref=task['task_card'])
        events = []
        if row['status'] == 'active':
            for other in current.values():
                if other['id'] != identifier and other['status'] == 'active':
                    if recorder or rows.get(other['id']) != other:
                        raise ContractError('活跃灵感已变化，请重新读取后决定切换')
                    events.append({'type': 'inspiration_updated', 'entry': {
                        **other, 'revision': other['revision'] + 1, 'status': 'paused',
                        'reason': 'Generation 切换至 ' + identifier + ': ' + args.get('reason', ''),
                        'updated_by': task['role'], 'update_task_ref': task['task_card']}})
        events.append({'type': 'inspiration_updated', 'entry': row})
        c = task['inputs'].get('context', {})
        if not recorder and row['status'] == 'active' and c.get('slot') == 'H3' and task['inputs'].get('purpose') == 'generate':
            events.append({'type': 'inspiration_task', 'task_id': task['task_id'], 'inspiration_id': identifier})
        return identifier, events

    def candidate_events(self, task, card):
        """Compatibility: an H3 submission itself can establish its first idea."""
        p = card['payload']
        if p['slot'] != 'H3':
            return []
        identifier = p.get('inspiration_id') or self.bound_id(task)
        events = []
        if not identifier:
            active = self.active()
            identifier = active['id'] if active else None
        if not identifier:
            identifier, events = self.update_events({
                'title': p['title'], 'idea': p['mechanism'], 'basis': p['research_question'],
                'progress': '已提交候选；实验状态见关联请求',
                'next_step': 'Generation 根据关联结果与未完成事项决定下一步',
                'status': 'active', 'reason': 'Generation 提交首个 H3 候选'}, task, key=card['id'])
        events += [{'type': 'inspiration_task', 'task_id': task['task_id'], 'inspiration_id': identifier},
                   {'type': 'inspiration_candidate', 'inspiration_id': identifier, 'hypothesis_ref': card['id']}]
        return events

    def source_binding(self, sequence):
        epoch = max((e['commit_sequence'] for e in self.events(sequence)
                     if e['type'] == 'implementation_intervention'), default=0)
        if epoch not in self._source_bindings:
            self._source_bindings[epoch] = self.solutions.source_record(epoch)
        return deepcopy(self._source_bindings[epoch])

    def recovery(self, identifier, sequence=None):
        rows = self.snapshot(sequence)
        if identifier not in rows:
            raise ContractError('未知或不可见的灵感 ID')
        events = self.events(sequence)
        tasks, owners, hypotheses = {}, set(), []
        for event in events:
            if event['type'] == 'task_snapshot':
                tasks[event['task']['task_id']] = event['task']
            elif event['type'] == 'inspiration_task' and event['inspiration_id'] == identifier:
                owners.add(event['task_id'])
            elif event['type'] == 'inspiration_candidate' and event['inspiration_id'] == identifier:
                hypotheses.append(event['hypothesis_ref'])
        requests, history = [], []
        for request in tasks.values():
            if request['kind'] != 'experiment_request':
                continue
            inputs = request['inputs']
            bound = inputs.get('inspiration_id')
            if bound != identifier and not (bound is None and inputs.get('parent_task') in owners):
                continue
            ref = request.get('experiment_ref') or request.get('request_ref')
            requests.append({'task_ref': request['task_card'], 'state': request['state'],
                'request_ref': request.get('request_ref'), 'experiment_ref': request.get('experiment_ref'),
                'evidence_refs': [r for r in request['outputs'] if self.journal.cards[r]['kind'] == 'EvidenceCard'],
                'error': request.get('error'), 'research_note': inputs.get('research_note'),
                'host_source': self.source_binding(self.journal.cards[ref]['sequence']) if ref else None,
                'configuration': {'ref': ref, 'field': '/payload'} if ref else {
                    'ref': request['task_card'], 'field': '/payload/inputs/draft'}})
            children = [t for t in tasks.values() if t['inputs'].get('parent_task') == request['task_id']]
            requests[-1]['evidence_refs'] = list(dict.fromkeys(requests[-1]['evidence_refs'] + [
                ref for child in children if child['kind'] == 'experiment'
                for ref in child['outputs'] if self.journal.cards[ref]['kind'] == 'EvidenceCard']))
            for code in children:
                candidate = (code.get('result') or {}).get('artifact')
                if code['inputs'].get('purpose') != 'code' or not candidate:
                    continue
                trials = [t for t in children if t['kind'] == 'experiment'
                          and (t['inputs'].get('approval') or {}).get('candidate', {}).get('id') == candidate['id']]
                history.append({'candidate': deepcopy(candidate),
                    'request_ref': candidate['experiment_ref'],
                    'experiment_ref': trials[-1]['inputs']['experiment_ref'] if trials else candidate['experiment_ref'],
                    'execution_refs': list(dict.fromkeys(t['inputs']['experiment_ref'] for t in trials)),
                    'source_task_ref': code['task_card'], 'execution_task_refs': [t['task_card'] for t in trials],
                    'configuration': {'ref': candidate['experiment_ref'], 'field': '/payload'},
                    'host_source': self.source_binding(self.journal.cards[candidate['experiment_ref']]['sequence']),
                    'evidence_refs': list(dict.fromkeys(r for t in trials for r in t['outputs']
                                                       if self.journal.cards[r]['kind'] == 'EvidenceCard')),
                    'failures': [{'error': t['error']} for t in trials if t.get('error')],
                    'inspiration_id': identifier,
                    'meaning': 'Exact prior source; read and incrementally revise it. Scores do not terminate this idea.'})
        return {**rows[identifier], 'hypothesis_refs': list(dict.fromkeys(hypotheses)),
                'requests': requests, 'implementation_history': history,
                'policy': 'References restore original code/config/results; saving notes never schedules experiments.'}

    def source_task_refs(self, sequence=None):
        key = (self.journal.sequence, sequence)
        if getattr(self, '_source_refs_key', None) != key:
            refs = set()
            for identifier in self.snapshot(sequence):
                recovery = self.recovery(identifier, sequence)
                refs.update(r['task_ref'] for r in recovery['requests'])
                for row in recovery['implementation_history']:
                    refs.update([row['source_task_ref'], *row['execution_task_refs']])
            self._source_refs_key, self._source_refs = key, refs
        return set(self._source_refs)

