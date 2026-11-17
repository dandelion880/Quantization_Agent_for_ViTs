"""Generation-owned research directions and method identities over the Journal.

No scientific score, experiment gate or duplicate solution store. Topic revisions,
method lineage and exact candidate versions are separate append-only identities.
"""
from copy import deepcopy
from core.cards import ContractError, TEXT, STRINGS, object_schema, fingerprint, make_card, validate_card
from core.journal import read_json

DECISION_SCHEMA = object_schema({
    'decision': {'enum': ['establish', 'continue', 'narrow', 'expand', 'pause', 'resume', 'switch']},
    'topic_id': TEXT, 'question': TEXT, 'mutable_dimensions': {**STRINGS, 'minItems': 1},
    'held_conditions': STRINGS, 'objective': TEXT, 'rationale': TEXT,
    'next_step': TEXT, 'reconsider_when': TEXT, 'previous_reconsider_when': TEXT,
    'unresolved_questions': STRINGS, 'inspiration_ids': STRINGS, 'method_ids': STRINGS,
    'source_refs': STRINGS, 'completed_action_ids': STRINGS, 'deferred_action_ids': STRINGS,
}, ['decision', 'question', 'mutable_dimensions', 'held_conditions', 'objective', 'rationale',
    'next_step', 'reconsider_when'])
REVISION_SCHEMA = deepcopy(DECISION_SCHEMA)
REVISION_SCHEMA['properties'].update(affected_slots={'type':'array', 'items':{'enum':['H1','H2','H3']},
                                                     'minItems':1,'uniqueItems':True}, impact=TEXT)
REVISION_SCHEMA['required'] += ['affected_slots', 'impact']



class ResearchTopics:
    def __init__(self, journal, inspirations, solutions):
        self.journal, self.inspirations, self.solutions = journal, inspirations, solutions

    def events(self, sequence=None):
        return [e for e in self.journal.events if sequence is None or e['commit_sequence'] <= sequence]

    def decision_card(self, task, content, refs):
        """Replay the same decision even after cards_written but before commit."""
        card=make_card('ReviewDecisionCard',self.journal.run_id,content,author='generation',refs=sorted(refs))
        card['logical_id']='ReviewDecisionCard-'+fingerprint(['research_direction',task['task_id']])[:32]
        card['id']=card['logical_id']+'@1'
        card['content_sha256']=fingerprint({k:v for k,v in card.items() if k!='content_sha256'})
        path=self.journal.card_path(card['id'])
        if path.exists() and card['id'] not in self.journal.cards:
            pending=validate_card(read_json(path))
            stable=lambda value:{k:v for k,v in value.items() if k not in ('created_at','content_sha256')}
            if stable(pending)!=stable(card):
                raise ContractError('未发布主题决定与原请求不一致，不能借恢复替换正文')
            card=pending
        return card

    def versions(self, sequence=None):
        return {e['entry']['ref']: deepcopy(e['entry']) for e in self.events(sequence) if e['type']=='topic_version'}

    def snapshot(self, sequence=None):
        return {r['id']:r for r in self.versions(sequence).values()}

    def active(self, sequence=None):
        return next((r for r in self.snapshot(sequence).values() if r['status']=='active'), None)

    def round_ref(self, number, sequence=None):
        return next((e['topic_ref'] for e in reversed(self.events(sequence))
                     if e['type']=='topic_round_binding' and e['round']==number), None)

    def binding(self, task, sequence=None):
        c=task['inputs'].get('context',{})
        # Published candidates, including old candidates after a topic switch,
        # always keep their original topic during L3 planning and execution.
        if c.get('hypothesis_ref'):
            return {k:v for k,v in self.journal.card(c['hypothesis_ref'])['payload'].items()
                    if k in ('topic_ref','method_id')}
        ref=c.get('topic_ref') or self.round_ref(c.get('round'),sequence)
        for e in self.events(sequence):
            if (e['type']=='topic_scope_binding' and e['round']==c.get('round')
                    and c.get('slot') in e['affected_slots']):
                ref=e['topic_ref']
        return {'topic_ref':ref} if ref else {}

    def decision_events(self, task, card, sequence, *, midround=False):
        p=card['payload']; rows=self.snapshot(sequence); current=self.snapshot()
        active=self.active(sequence)
        if active != self.active():
            raise ContractError('主题已变化；恢复当前状态后再决定')
        identifier=p.get('topic_id')
        old=rows.get(identifier) if identifier else active
        action=p['decision']
        if identifier and (old is None or old != current.get(identifier)):
            raise ContractError('主题不存在、不可见或已被修订')
        if action=='establish' and rows:
            raise ContractError('已有主题；请继续、切换或恢复，不重建初始主题')
        if action in ('continue','narrow','expand','pause') and (not old or old['status']!='active'):
            raise ContractError('该决定需要已有活跃主题')
        if action=='resume' and (not old or old['status']!='paused'):
            raise ContractError('恢复须指向保留的暂停主题')
        if action=='switch' and not identifier:
            old=None
        if action=='switch' and old and old['status']=='active':
            raise ContractError('同一主题请继续或缩放；切换须选择其他主题')
        identifier=old['id'] if old else 'topic-'+fingerprint(card['id'])[:20]
        for idea in p.get('inspiration_ids',[]):
            if idea not in self.inspirations.snapshot(sequence):
                raise ContractError('关联灵感须存在于当前读取快照')
        methods=self.methods(sequence)
        if any(mid not in methods for mid in p.get('method_ids',[])):
            raise ContractError('关联方法须使用已有谱系身份')
        actions=deepcopy(old.get('actions',[]) if old else [])
        known={a['id']:a for a in actions}
        if set(p.get('completed_action_ids',[])) & set(p.get('deferred_action_ids',[])):
            raise ContractError('同一研究动作不能同时完成和暂存')
        for field,status in (('completed_action_ids','completed'),('deferred_action_ids','deferred')):
            for aid in p.get(field,[]):
                if aid not in known:
                    raise ContractError('未完成动作 ID 不属于当前主题')
                known[aid].update(status=status, decision_ref=card['id'])
        continuing=next((a for a in actions if a['description']==p['next_step'] and a['status']=='open'),None)
        if continuing:
            continuing.update(reconsider_when=p['reconsider_when'], decision_ref=card['id'])
        else:
            actions.append({'id':'action-'+fingerprint([identifier,card['id']])[:16], 'description':p['next_step'],
                            'reconsider_when':p['reconsider_when'], 'status':'open', 'decision_ref':card['id']})
        row={**deepcopy(old or {}), **{k:deepcopy(p[k]) for k in
             ('question','mutable_dimensions','held_conditions','objective','rationale','next_step','reconsider_when')},
             'id':identifier, 'version':old['version']+1 if old else 1,
             'status':'paused' if action=='pause' else 'active', 'decision':action,
             'decision_ref':card['id'], 'actions':actions}
        for field in ('inspiration_ids','method_ids','source_refs'):
            # Omission never drops retained associations or unresolved work.
            row[field]=list(dict.fromkeys((old or {}).get(field,[])+p.get(field,[])))
        row['unresolved_questions']=deepcopy(p.get('unresolved_questions',(old or {}).get('unresolved_questions',[])))
        row['ref']=identifier+'@'+str(row['version'])
        events=[]
        if active and active['id']!=identifier and row['status']=='active':
            paused={**active,'version':active['version']+1,'status':'paused','decision':'switch',
                    'decision_ref':card['id'], 'rationale':p['rationale'],
                    'reconsider_when':p.get('previous_reconsider_when',active['reconsider_when'])}
            paused['ref']=paused['id']+'@'+str(paused['version'])
            events.append({'type':'topic_version','entry':paused})
        events.append({'type':'topic_version','entry':row})
        if midround:
            events.append({'type':'topic_scope_binding','round':task['inputs']['context']['round'],
                'topic_ref':row['ref'],'affected_slots':p['affected_slots'],'impact':p['impact'],
                'decision_ref':card['id']})
        elif row['status']=='active':
            events.append({'type':'topic_round_binding','round':task['inputs']['context']['round'],
                           'topic_ref':row['ref'],'decision_ref':card['id']})
        return row,events

    def methods(self, sequence=None):
        rows={}
        for e in self.events(sequence):
            if e['type']=='method_candidate':
                row=rows.setdefault(e['method_id'],{'id':e['method_id'],'name':e['name'],
                    'parent_method_ids':e['parent_method_ids'],'candidate_refs':[], 'inspiration_ids':[], 'topic_refs':[]})
                row['candidate_refs'].append(e['candidate_ref'])
                for field,key in (('inspiration_ids','inspiration_id'),('topic_refs','topic_ref')):
                    if e.get(key) and e[key] not in row[field]: row[field].append(e[key])
        return deepcopy(rows)

    def bind_method(self, task, content, sequence):
        body=deepcopy(content); declaration=body.get('method_lineage',{})
        mid=body.get('method_id')
        known=self.methods(sequence)
        if mid and mid not in known:
            raise ContractError('method_id 必须是已有谱系；新算法省略 ID，由宿主分配')
        parents=declaration.get('parent_method_ids',[])
        solutions={'solution:'+r['id'] for r in self.solutions.available(sequence)}
        if any(p not in known and p not in solutions for p in parents):
            raise ContractError('方法父身份须是已有 method_id 或 solution:<完整方案ID>')
        if mid:
            if parents and parents!=known[mid]['parent_method_ids']:
                raise ContractError('已有谱系父身份不能改写；派生新方法须省略 method_id')
        else:
            mid='method-'+fingerprint([task['task_id'],body])[:20]
        body['method_id']=mid
        return body

    def candidate_event(self, task, card, inspiration_id=None):
        p=card['payload']; known=self.methods().get(p['method_id'])
        declaration=p.get('method_lineage',{})
        parents=known['parent_method_ids'] if known else declaration.get('parent_method_ids',[])
        base=p.get('solution_derivation',{}).get('base_solution_id')
        if not known and base and 'solution:'+base not in parents: parents=[*parents,'solution:'+base]
        return {'type':'method_candidate','method_id':p['method_id'],
            'name':known['name'] if known else declaration.get('name',p['title']),
            'parent_method_ids':parents,'change':declaration.get('change',p.get('solution_derivation',{}).get('changes')),
            'candidate_ref':card['id'],'topic_ref':p.get('topic_ref'), 'inspiration_id':p.get('inspiration_id') or inspiration_id,
            'generation_task_id':task['task_id'] if task['inputs']['purpose']=='generate' and card['version']==1 else None}

    def recovery(self, ref, sequence=None):
        versions=self.versions(sequence); rows=self.snapshot(sequence)
        row=versions.get(ref) or rows.get(ref)
        if row is None: raise ContractError('主题不存在或不属于当前读取快照')
        # Exact version boundaries avoid reading later decisions through an old ref.
        refs={r['ref'] for r in versions.values() if r['id']==row['id'] and r['version']<=row['version']}
        tasks={e['task']['task_id']:e['task'] for e in self.events(sequence) if e['type']=='task_snapshot'}
        requests={t['task_id'] for t in tasks.values() if t['kind']=='experiment_request'
                  and t['inputs'].get('topic_ref') in refs}
        candidates=[e for e in self.events(sequence) if e['type']=='method_candidate' and e.get('topic_ref') in refs]
        raw,code=self.inspirations.request_history(requests,sequence)
        notes=[e for e in self.events(sequence) if e['type']=='topic_note' and e['note']['topic_ref'] in refs]
        return {**row,'candidate_refs':[e['candidate_ref'] for e in candidates],
            'method_ids':list(dict.fromkeys(row['method_ids']+[e['method_id'] for e in candidates])),
            'inspiration_ids':list(dict.fromkeys(row['inspiration_ids']+[e['inspiration_id'] for e in candidates if e.get('inspiration_id')])),
            'requests':raw,'implementation_history':code,'recorder_notes':notes[-3:]}

    def method_recovery(self, identifier, sequence=None):
        row=self.methods(sequence).get(identifier)
        if not row: raise ContractError('未知或不可见的方法谱系')
        events=self.events(sequence)
        owners={e.get('generation_task_id') for e in events if e['type']=='method_candidate' and e['method_id']==identifier}-{None}
        tasks={e['task']['task_id']:e['task'] for e in events if e['type']=='task_snapshot'}
        requests={t['task_id'] for t in tasks.values() if t['kind']=='experiment_request' and
            (t['inputs'].get('method_id')==identifier or t['inputs'].get('hypothesis_ref') in row['candidate_refs']
             or (not t['inputs'].get('method_id') and t['inputs'].get('parent_task') in owners))}
        raw,code=self.inspirations.request_history(requests,sequence)
        return {**row,'requests':raw,'implementation_history':code}

    def retained_recoveries(self, sequence=None):
        """Include method sources from legacy rounds without a topic or idea."""
        for row in self.snapshot(sequence).values():
            yield self.recovery(row['ref'],sequence)
        for identifier in self.methods(sequence):
            yield self.method_recovery(identifier,sequence)

    def source_task_refs(self, sequence=None):
        refs=set()
        for recovery in self.retained_recoveries(sequence):
            refs.update(r['task_ref'] for r in recovery['requests'])
            for item in recovery['implementation_history']:
                refs.update([item['source_task_ref'],*item['execution_task_refs']])
        return refs
