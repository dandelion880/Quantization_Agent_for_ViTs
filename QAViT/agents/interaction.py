"""角色上下文、宿主工具与正文提交。共享 Workflow 的 Journal/队列，不持有第二份状态。

这是 Workflow 的职责分区，不独立实例化；状态和生命周期统一由 Workflow 拥有。
"""
from agents.tasks import TASKS
from copy import deepcopy
from pathlib import Path
from importlib.util import find_spec
import re
from runtime.paths import ROOT
from core.cards import ContractError, canonical, fingerprint, make_card, validate_payload, validate_schema, TEXT, PAYLOAD_SCHEMAS
from core.journal import file_sha256, read_json, atomic_json
from core.debate import ranking_table, validate_verdict
from agents.roles import tools_for, submission_schema
from tools.experiments import SERVICES
from tools.data_receipt import data_receipt
from tools.safeguards import write_artifact, NUMERIC_IMPORTS, PUBLIC_INTERFACE_FILES



class RoleInteraction:
    DIRECT_CHARACTERS = 7000
    PAGE_CHARACTERS = 12000

    def read_view(self, task):
        """工具延用发起调用的提交快照；下一次模型调用才刷新。"""
        if task.get('_read_view') is not None:
            return task['_read_view']
        if task.get('request_path'):
            request = read_json(task['request_path'])
            return {**request['spec']['context']['read_view'], 'assets': request.get('read_permissions', {})}
        return {'sequence': self.journal.sequence, 'memory_version': self.journal.memory_version}

    def visible_task_snapshots(self, sequence):
        tasks = {}
        for event in self.journal.events:
            if event['commit_sequence'] <= sequence and event['type'] == 'task_snapshot':
                tasks[event['task']['task_id']] = event['task']
        return tasks

    def bound_code_sources(self, task):
        """Only these host-assigned handoffs authorize an otherwise unregistered code source."""
        purpose, context = task['inputs']['purpose'], task['inputs']['context']
        if purpose == 'safeguard':
            values = [context.get('candidate'), context.get('reference')]
        elif purpose == 'code':
            values = [context.get('revision_feedback', {}).get('previous_candidate')]
        else:
            values = []
        return [v for v in values if isinstance(v, dict) and
                {'id', 'path', 'source_sha256', 'experiment_ref', 'experiment_sha256'} <= v.keys()]

    def position_branch(self, task, tasks):
        """Only host-persisted parent_task links carry the current exchange boundary."""
        current, seen = tasks.get(task['task_id'], task), set()
        while current and current['task_id'] not in seen:
            seen.add(current['task_id'])
            if current['inputs'].get('purpose') == 'position':
                return current
            current = tasks.get(current['inputs'].get('parent_task'))
        return None

    def can_read_card(self, task, card):
        sequence = self.read_view(task)['sequence']
        if self.journal.cards.get(card['id'], {}).get('sequence', sequence + 1) > sequence:
            return False
        purpose = task['inputs']['purpose']
        if purpose == 'reference':
            return card['id'] in task['inputs']['refs'] and card['kind'] in {'ResearchBrief', 'ExperimentCard'}
        if card['kind'] == 'TaskCard':
            body = card['payload']
            entitled = (body['task_id'] == task['task_id'] or body['inputs'].get('parent_task') == task['task_id']
                    or card['id'] in task['inputs']['refs']
                    or card['id'] == task['inputs']['context'].get('revision_feedback', {}).get('execution_task_ref')
                    or (body['inputs'].get('purpose') in ('code', 'reference')
                        and (body.get('result') or {}).get('artifact') in self.bound_code_sources(task))
                    or (body['kind'] == 'literature' and body['state'] == 'completed'))
            if not entitled:
                return False  # 无读取资格的工程记录无需再遍历其历史版本与科研出处。
        if task.get('_acl_tasks_sequence') != sequence:
            task['_acl_tasks_sequence'] = sequence
            task['_acl_tasks'] = self.visible_task_snapshots(sequence)
        tasks = task['_acl_tasks']
        branch = self.position_branch(task, tasks)
        if branch:
            context = branch['inputs']['context']
            cache_key = (sequence, branch['task_id'])
            if task.get('_branch_acl_key') != cache_key:
                roots = {t['task_id'] for t in tasks.values() if t['inputs'].get('purpose') == 'position'
                         and all(t['inputs']['context'].get(k) == context.get(k) for k in ('round', 'pair', 'exchange'))
                         and t['inputs']['context'].get('side') != context.get('side')}
                owners = {t['task_id']: self.position_branch(t, tasks) for t in tasks.values()}
                private = {tid: root['task_id'] for tid, root in owners.items() if root and root['task_id'] in roots}
                created = {}
                hidden_refs = set()
                for event in self.journal.events:
                    if event['commit_sequence'] > sequence or event['type'] != 'task_snapshot':
                        continue
                    value = event['task']
                    if value['task_id'] in roots:
                        created.setdefault(value['task_id'], event['commit_sequence'])
                    root = private.get(value['task_id'])
                    if root:
                        for ref in [value['task_card'], *value['outputs'], value.get('experiment_ref')]:
                            # Cache reuse cannot privatize evidence published before the branch.
                            if ref and self.journal.cards[ref]['sequence'] >= created[root]:
                                hidden_refs.add(ref)
                task['_branch_acl_key'], task['_hidden_refs'] = cache_key, hidden_refs
                if '_acl_allowed_refs' in task:
                    task['_acl_allowed_refs'].clear()
            proven = task.get('_acl_allowed_refs', set())
            todo, seen = [card['id']], set()
            while todo:
                ref = todo.pop()
                if ref in seen or ref in proven:
                    continue
                seen.add(ref)
                value = card if ref == card['id'] else self.journal.card(ref)
                p = value['payload']
                other_position = (value['kind'] == 'ReviewDecisionCard' and p.get('phase') == 'position'
                    and all(p.get(k) == context.get(k) for k in ('round', 'pair', 'exchange'))
                    and p.get('side') != context.get('side'))
                if value['id'] in task['_hidden_refs'] or other_position:
                    return False
                todo.extend(r for r in value['refs'] if r in self.journal.cards and r not in seen)
            proven.update(seen)
        return True

    def require_visible(self, task, ref):
        if isinstance(ref, str) and re.fullmatch(r'C[1-9][0-9]*', ref):
            version = int(ref[1:])
            ref = next((r for r, v in self.journal.card_versions.items() if v == version), ref)
        card = self.journal.card(ref)
        if not self.can_read_card(task, card):
            raise ContractError('提交快照/权限拒绝读取：独立参考规格隔离及同回合立场隔离适用于所有入口')
        return card

    def bind_submission(self, task, content):
        """固定身份和短引用在单一边界编译；错误身份、未来与私有来源仍拒绝。"""
        content = deepcopy(content)
        purpose, context = task['inputs']['purpose'], task['inputs']['context']
        bindings = {}
        if purpose in ('generate', 'evolve'):
            bindings = {k: context[k] for k in ('round', 'slot', 'hypothesis_type')}
            if purpose == 'evolve':
                bindings['parent_refs'] = [context['parent_ref']]
            content.setdefault('feedback_used', [])
            for index, prediction in enumerate(content.get('predictions', [])):
                prediction.setdefault('id', 'p' + str(index + 1))
            metas = [r for r in task['inputs']['refs'] if self.journal.cards[r]['kind'] == 'MetareviewCard']
            if len(metas) == 1:
                for row in content['feedback_used']:
                    row.setdefault('metareview_ref', metas[0])
        elif TASKS[purpose]['output'] == 'ReviewDecisionCard':
            bindings['phase'] = TASKS[purpose]['phase']
            bindings.update({k: context[k] for k in ('round', 'pair', 'exchange') if k in context})
            if purpose == 'position':
                bindings['side'] = context['side']
            content.setdefault('evidence_refs', [])
            content.setdefault('uncertainties', [])
        content = self.resolve_card_refs(task, content)
        for key, value in bindings.items():
            if key in content and content[key] != value:
                raise ContractError('提交与宿主固定身份冲突: ' + key)
            content[key] = value
        if purpose == 'record':
            content = self.memory.prepare_metareview(content, context['recording_batch'], context['round'])
        return content

    def resolve_card_refs(self, task, content):
        """提交与实验请求共用明确的卡片字段；不猜科学参数或自由文本身份。"""
        card_ref_keys = {'ref', 'evidence_ref', 'judgment_ref', 'metareview_ref', 'revision_ref',
                         'parent_ref', 'hypothesis_ref', 'experiment_ref', 'source_ref', 'trial_evidence_ref',
                         'selected_hypothesis_ref', 'parent_selection_ref', 'selection_ref', 'supersedes'}
        def resolve(value, key=''):
            if isinstance(value, dict):
                return {k: resolve(v, k) for k, v in value.items()}
            if isinstance(value, list):
                return [resolve(v, key[:-1] if key.endswith('_refs') else key) for v in value]
            if isinstance(value, str) and key in card_ref_keys:
                return self.require_visible(task, value)['id']
            return value
        return resolve(content)

    @staticmethod
    def card_field(card, field):
        value = card
        if field:
            if not field.startswith('/'):
                raise ContractError('read_card.field 必须是 JSON pointer，例如 /payload/observations')
            for part in field[1:].split('/'):
                key = part.replace('~1', '/').replace('~0', '~')
                if isinstance(value, dict) and key in value:
                    value = value[key]
                elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
                    value = value[int(key)]
                else:
                    shape = (f'当前对象键（前20项）={canonical(list(value)[:20])}' if isinstance(value, dict)
                             else f'当前数组长度={len(value)}，请使用有效数字下标' if isinstance(value, list)
                             else f'当前值类型={type(value).__name__}，没有子字段')
                    raise ContractError(f'read_card.field 不存在: {field}；{shape}。'
                                        '路径从卡片根开始，科研字段通常在 /payload/... 下。')
        return value

    def result_values(self, task):
        """原始工具结果保存在已提交 TaskCard；不依赖被截短的队列展示行。"""
        rows = task.get('tool_results', [])
        if not rows:
            return
        tasks = self.queue.snapshot()
        for row in rows:
            if row['task'] not in tasks:
                raise ContractError('未知任务: ' + row['task'])
            child = tasks[row['task']]
            yield row, child, child.get('result')

    def read_ranges(self, task, ref, field, digest):
        """校验不可变原文与页面；完整祖先覆盖子字段，缺页绝不视作完整。"""
        target = self.journal.card(ref)
        if fingerprint(self.card_field(target, field)) != digest:
            return []
        groups, exact = {}, []
        for row, child, result in self.result_values(task):
            if (child['state'] != 'completed' or child['kind'] != 'host_tool'
                    or child['inputs']['name'] not in ('read_card', 'read_asset') or not isinstance(result, dict)):
                continue
            if child['inputs']['name'] == 'read_asset':
                path = self.journal.card_path(ref)
                if result.get('asset') != str(path) or result.get('sha256') != file_sha256(path):
                    continue
                text = path.read_text()
                start, part = result.get('offset'), result.get('text')
                if (isinstance(start, int) and start >= 0 and isinstance(part, str)
                        and result.get('total_characters') == len(text) and text[start:start + len(part)] == part):
                    groups.setdefault((ref, 'asset-root', len(text)), []).append((start, start + len(part)))
                continue
            source = result.get('ref', result.get('id'))
            if source not in self.journal.cards:
                continue
            # 只有同一原文或续调 TaskCard 的稳定输入可能覆盖目标；不反复加载无关证据。
            if source != ref and not (field.startswith('/payload/inputs/context/') and target['kind'] == 'TaskCard'):
                continue
            original = self.journal.card(source)
            # 续调 TaskCard 会新增版本；仅内容相同的固定输入字段可以延续覆盖。
            stable_input = False
            if (field.startswith('/payload/inputs/context/') and original['kind'] == 'TaskCard'
                    and target['kind'] == 'TaskCard' and original['logical_id'] == target['logical_id']):
                try:
                    stable_input = fingerprint(self.card_field(original, field)) == digest
                except ContractError:
                    # 旧版本尚无这个字段，不构成覆盖；不把正常续调变成程序错误。
                    pass
            if source != ref and not stable_input:
                continue
            if result.get('id') == source:
                if result == original:
                    return [(0, float('inf'))]
                continue
            ancestor = result.get('field', '')
            if ancestor != field and not (ancestor == '' or field.startswith(ancestor + '/')):
                continue
            text = canonical(self.card_field(original, ancestor))
            start, part = result.get('offset'), result.get('text')
            if (not isinstance(start, int) or start < 0 or not isinstance(part, str)
                    or result.get('sha256') != fingerprint(self.card_field(original, ancestor))
                    or result.get('total_characters') != len(text) or text[start:start + len(part)] != part):
                continue
            span = (start, start + len(part))
            if ancestor == field:
                exact.append(span)
            groups.setdefault((source, ancestor, len(text)), []).append(span)
        for (_, ancestor, length), spans in groups.items():
            stop = 0
            for start, end in sorted(spans):
                if start > stop: break
                stop = max(stop, end)
            if stop >= length:
                return [(0, float('inf'))]
        return exact

    def present_card(self, card):
        """不做科学摘要；仅移除工程封套，重型身份保留精确字段入口。"""
        if card['kind'] == 'TaskCard':
            body = card['payload']
            status = {k: deepcopy(body[k]) for k in ('task_id', 'key', 'kind', 'state', 'error', 'outputs') if k in body}
            result = body.get('result')
            if isinstance(result, dict):
                status['result'] = {k: deepcopy(result[k]) for k in ('status', 'reason', 'message') if k in result}
                for key, value in list(status['result'].items()):
                    if len(canonical(value)) > 1200:
                        status['result'][key] = {'preview': canonical(value)[:1000], 'truncated': True,
                            'read_card': {'ref': card['id'], 'field': '/payload/result/' + key}}
            if len(canonical(status.get('error'))) > 1600:
                status['error'] = {'preview': canonical(status['error'])[:1200], 'truncated': True,
                                   'read_card': {'ref': card['id'], 'field': '/payload/error'}}
            return {'id': card['id'], 'kind': card['kind'], 'version': card['version'],
                    'payload': status,
                    'read_card': {'ref': card['id']}, 'presentation': 'engineering status; full immutable source on demand'}
        value = deepcopy(card)
        if card['kind'] == 'ExperimentCard':
            execution = value['payload']['execution']
            research_contract = execution.get('research_quantization_contract')
            if (research_contract is not None and self.brief['id'] in card['refs']
                    and research_contract == self.brief['payload'].get('quantization_contract')):
                execution['research_quantization_contract'] = {
                    'read_card': {'ref': self.brief['id'], 'field': '/payload/quantization_contract'},
                    'sha256': fingerprint(research_contract)}
            if 'measurement_implementation' in execution:
                execution['measurement_implementation'] = {
                    'read_card': {'ref': card['id'], 'field': '/payload/execution/measurement_implementation'},
                    'sha256': fingerprint(execution['measurement_implementation'])}
        if card['kind'] == 'EvidenceCard':
            def numeric_container(item):
                # Tables and named scalar statistics stay in the scientific view.
                # Only dense numeric arrays, or dictionaries composed of them, move to exact references.
                if isinstance(item, list) and item:
                    return all((isinstance(v, (int, float)) and not isinstance(v, bool))
                               or (isinstance(v, list) and numeric_container(v)) for v in item)
                return isinstance(item, dict) and bool(item) and all(numeric_container(v) for v in item.values())

            repeated = {}
            def observations(item, field):
                if numeric_container(item) and len(canonical(item)) > self.DIRECT_CHARACTERS:
                    return {'read_card': {'ref': card['id'], 'field': field}, 'sha256': fingerprint(item)}
                # Keep the first complete value; identical replay counters need not recur 100 times.
                # Canonical content is the key, so this is exact reuse, never a scientific summary.
                if isinstance(item, (dict, list)):
                    text = canonical(item)
                    if len(text) > 1000:
                        if text in repeated:
                            return {'read_card': {'ref': card['id'], 'field': repeated[text]},
                                    'sha256': fingerprint(item)}
                        repeated[text] = field
                if isinstance(item, dict):
                    return {k: observations(v, field + '/' + k.replace('~', '~0').replace('/', '~1'))
                            for k, v in item.items()}
                if isinstance(item, list):
                    return [observations(v, field + '/' + str(i)) for i, v in enumerate(item)]
                return item

            value['payload']['observations'] = observations(card['payload']['observations'], '/payload/observations')
            provenance = value['payload']['provenance']
            for key in ('data_identity', 'model_identity'):
                if key in provenance:
                    provenance[key] = {'read_card': {'ref': card['id'], 'field': '/payload/provenance/' + key},
                                       'sha256': fingerprint(provenance[key])}
            # 冻结实验是必需交付对象；仅折叠已核对相同的执行正文，保留异常/旧记录原文。
            if 'resolved_execution' in provenance:
                experiment = self.journal.card(card['payload']['experiment_ref'])
                if (provenance.get('experiment_sha256') == experiment['content_sha256']
                        and provenance['resolved_execution'] == experiment['payload']['execution']):
                    provenance['resolved_execution'] = {
                        'read_card': {'ref': experiment['id'], 'field': '/payload/execution'},
                        'sha256': fingerprint(provenance['resolved_execution'])}
        return value

    def required_refs(self, task):
        purpose, c = task['inputs']['purpose'], task['inputs']['context']
        refs = list(task['inputs']['refs'])
        if purpose == 'generate':
            refs = refs if c.get('previous_review_pending') else [self.brief['id']]
        elif purpose == 'record':
            refs = []
            for ref in c['recording_batch']['scientific_refs']:
                card = self.journal.card(ref)
                if (card['kind'] in ('HypothesisCard', 'EvidenceCard') or
                    (card['kind'] == 'ReviewDecisionCard' and card['payload']['phase'] in
                     ('verdict', 'ranking', 'interpretation'))):
                    refs.append(ref)
        elif purpose == 'evolve':
            refs = [r for r in refs if self.journal.card(r)['kind'] != 'MetareviewCard']
        elif purpose == 'rank':
            refs = [*c.get('candidate_refs', {}).values(),
                    *[m['verdict_ref'] for m in self.selection_matches(c).values()]]
        elif purpose in ('position', 'arbitrate'):
            # 当前比较对象、当前/最近争点；更早交锋仍在分页目录中。
            refs = [r for r in refs if self.journal.card(r)['kind'] != 'ReviewDecisionCard'
                    or self.journal.card(r)['payload'].get('phase') == 'screening'
                    or self.journal.card(r)['payload'].get('exchange', 0) >= c.get('exchange', 1) - 1]
        elif purpose == 'supervise':
            refs = [r for r in refs if self.journal.card(r)['kind'] == 'TaskCard']
        for row, child, result in self.result_values(task):
            if child['kind'] == 'experiment_request':
                refs.extend(child['outputs'])
        # 明确的出处依赖，不扫描全库推断科学相关性。
        # 实验审批/解释可能继承 Generation 的反馈引用；引用不等于要求重灌完整复盘。
        # 必需反馈字段由 reminders 单独交付，完整历史正文仍可按引用读取。
        selected = list(dict.fromkeys(r for r in refs if self.journal.card(r)['kind'] != 'MetareviewCard'))
        for ref in list(selected):
            card = self.journal.card(ref)
            if card['kind'] == 'EvidenceCard':
                selected.append(card['payload']['experiment_ref'])
        return list(dict.fromkeys(selected))

    @staticmethod
    def prior_basis_pairs(payload):
        """结构身份优先；兼容文本只去掉明确的外围标点，不修补或猜测身份。"""
        structured, texts = [], []
        def collect(value):
            if isinstance(value, dict):
                if 'claim_id' in value and 'revision_ref' in value:
                    structured.append((value['claim_id'], value['revision_ref']))
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)
            elif isinstance(value, str):
                texts.append(value)
        collect(payload)
        if structured:
            return structured
        result = []
        label = re.compile(r'''(?<![\w])(?P<key>claim_id|revision_ref)[`"']?\s*[:=：]\s*''')
        token = re.compile(r'''[([{<（]*\s*(?:(?P<quote>[`"'])(?P<quoted>.*?)(?P=quote)|(?P<bare>[^\s,，;；。"'`()（）\[\]{}<>]+))''')
        for text in texts:
            labels = []
            for match in label.finditer(text):
                value = token.match(text, match.end())
                identity = None if value is None else value['quoted'] if value['quote'] else value['bare']
                # 裸值末尾的单个英文句号是句法分隔；引号内及 ID 内部字符保持原样。
                if value and not value['quote'] and identity.endswith('.') and not identity.endswith('..'):
                    identity = identity[:-1]
                labels.append((match['key'], identity))
            index = 0
            while index + 1 < len(labels):
                left, right = labels[index:index + 2]
                if left[0] == right[0]:
                    index += 1
                    continue
                pair = dict((left, right))
                result.append((pair['claim_id'], pair['revision_ref']))
                index += 2
        return result

    def prior_basis_field(self, task, claim_id, revision):
        """提交交接与 Recorder 共用固定身份/版本/权限核验。"""
        if not isinstance(claim_id, str) or not claim_id.strip() or not isinstance(revision, str) or not revision.strip():
            raise ContractError('旧依据 claim_id/revision_ref 必须为非空完整身份')
        prior = self.require_visible(task, revision)
        if prior['kind'] != 'MetareviewCard':
            raise ContractError('旧依据 revision_ref 必须指向 MetareviewCard')
        matches = [i for i, row in enumerate(prior['payload']['knowledge_updates']) if row['claim_id'] == claim_id]
        if len(matches) != 1:
            raise ContractError('旧依据 claim_id 必须在指定 revision_ref 中唯一存在: ' + claim_id)
        return '/payload/knowledge_updates/' + str(matches[0])

    def recorder_prior_basis(self, task):
        """仅解析本批直接引用中已明确成对标注的旧知识身份，不推断科学相关性。"""
        for source_ref in self.required_refs(task):
            source = self.require_visible(task, source_ref)
            # 工程日志会保存工具说明、引用示例及被拒草稿；仍须审查，但不是已发表科研主张。
            if source['kind'] == 'TaskCard':
                continue
            for claim_id, revision in self.prior_basis_pairs(source['payload']):
                if revision not in source['refs']:
                    raise ContractError('Recorder 旧依据 revision_ref 必须属于本批来源卡片的直接 refs: ' + str(revision))
                yield revision, self.prior_basis_field(task, claim_id, revision), source_ref, claim_id

    def deliver_bound_feedback(self, task, memory, state, latest, deliver):
        """只用已记录的对象引用绑定反馈；全局最新、文本长度和写入顺序不构成必读理由。"""
        if latest:
            memory['reminders'] = {'metareview_ref': latest, 'fields': {},
                'read_card': {'ref': latest}, 'required': False,
                'policy': '全局复盘入口；仅明确关联本对象或分配给本任务的反馈为必读。'}
        purpose = task['inputs']['purpose']
        if purpose == 'reference':
            return
        # purpose 已选定的处理对象与出处分别绑定：共享 Evidence 不把另一方案变成当前对象。
        automatic = purpose not in ('record', 'supervise')
        selected = set(self.required_refs(task)) if automatic else set()
        current = {r for r in selected if self.journal.card(r)['kind'] in
                   ('HypothesisCard', 'ExperimentCard', 'EvidenceCard')}
        bound = {r for r in selected if self.journal.card(r)['kind'] not in
                 ('ResearchBrief', 'MetareviewCard', 'TaskCard')}
        provenance, todo = set(), list(bound)
        while todo:
            ref = todo.pop()
            if ref in provenance:
                continue
            card = self.require_visible(task, ref)
            provenance.add(ref)
            for source in card['refs']:
                if self.journal.card(source)['kind'] in ('ExperimentCard', 'EvidenceCard'):
                    todo.append(source)
        bound.update(provenance)
        hypotheses = {r for r in current if self.journal.card(r)['kind'] == 'HypothesisCard'}
        experiments = {r for r in provenance if self.journal.card(r)['kind'] == 'ExperimentCard'}
        evidence = {r for r in provenance if self.journal.card(r)['kind'] == 'EvidenceCard'}
        feedback = memory.setdefault('bound_feedback', [])
        seen = set()
        def field(ref, pointer, reason):
            if any(r == ref and (pointer == p or pointer.startswith(p + '/')) for r, p in seen):
                return
            descendants = {(r, p) for r, p in seen if r == ref and p.startswith(pointer + '/')}
            if descendants:
                feedback[:] = [e for e in feedback if (e['read_card']['ref'], e['read_card']['field']) not in descendants]
                for key in ('required_reads', 'field_manifest'):
                    memory[key] = [e for e in memory.get(key, []) if (e['ref'], e['field']) not in descendants]
                seen.difference_update(descendants)
            seen.add((ref, pointer))
            card = self.require_visible(task, ref)
            value = self.card_field(card, pointer)
            text, digest = canonical(value), fingerprint(value)
            entry = {'read_card': {'ref': ref, 'field': pointer}, 'reason': reason, 'sha256': digest}
            if len(text) <= self.DIRECT_CHARACTERS:
                entry['value'] = deepcopy(value)
                memory.setdefault('field_manifest', []).append({'ref': ref, 'field': pointer, 'sha256': digest})
            else:
                stop = 0
                for start, end in sorted(self.read_ranges(task, ref, pointer, digest)):
                    if start > stop:
                        break
                    stop = max(stop, end)
                memory['required_reads'].append({'ref': ref, 'field': pointer, 'sha256': digest,
                    'total_characters': len(text), 'next_offset': int(stop) if stop < len(text) else None,
                    'reason': reason})
                entry['required'] = True
            feedback.append(entry)
        if purpose == 'record':
            for ref, pointer, source, claim_id in self.recorder_prior_basis(task):
                field(ref, pointer, '本批 ' + source + ' 明确关联的旧知识 ' + claim_id + '；核对依据，不增加本批 source_refs')
        if purpose == 'generate':
            for ref in task['inputs']['refs']:
                card = self.require_visible(task, ref)
                if card['kind'] == 'MetareviewCard':
                    for key in ('generation_guidance', 'next_questions', 'recurring_failures', 'blind_spots'):
                        field(ref, '/payload/' + key, '宿主为本轮 Generation 绑定的前轮反馈；其余复盘按需读取')
        # 较早交锋的整卡保持按需；已记录的限制字段不按条数/年龄删除。
        # 不推断未结构化文字是否已解决，rationale 和明确问题/条件字段保守保留原文。
        demoted = set(task['inputs']['refs']) - selected if automatic else set()
        for card in self.journal.find_cards('ReviewDecisionCard') if automatic else ():
            if not self.can_read_card(task, card):
                continue
            ref, payload = card['id'], card['payload']
            subjects = {r for r in card['refs'] if self.journal.card(r)['kind'] == 'HypothesisCard'}
            spec_refs = {r for r in card['refs'] if self.journal.card(r)['kind'] == 'ExperimentCard'}
            if subjects:
                linked = bool(subjects & hypotheses or (not hypotheses and spec_refs & current))
                reason = 'current_subject_or_explicit_experiment'
            elif spec_refs:
                linked = bool(spec_refs & experiments)
                reason = 'current_experiment'
            else:
                linked = bool(set(card['refs']) & evidence)
                reason = 'ambiguous_subject; evidence-only feedback conservatively retained'
            if ref not in selected and ref not in demoted and not linked:
                continue
            bound.add(ref)
            if ref in selected:
                continue  # 明确处理对象已经交付，不再派生重复字段。
            old_exchange = (purpose in ('position', 'arbitrate')
                and payload.get('phase') in ('position', 'verdict')
                and payload.get('exchange', 0) < task['inputs']['context'].get('exchange', 1) - 1)
            binding = {'ref': ref, 'version': card['version'], 'read_card': {'ref': ref}, 'reason': reason}
            if ref in demoted or old_exchange:
                binding.update(required=False, body='optional history; exact issue/condition fields retained')
                for key in ('rationale', 'uncertainties', 'objections', 'responses_to_prior', 'next_questions',
                            'evidence_gaps', 'hard_rejection_reasons', 'limitations', 'counterevidence',
                            'conditions', 'applicability', 'unresolved_questions', 'scope'):
                    if key in payload and payload[key]:
                        field(ref, '/payload/' + key, '当前对象较早交锋的已记录问题/条件；未结构化解决状态保守保留')
            else:
                deliver(ref)
                binding['body'] = 'current bound feedback; full source'
            memory.setdefault('feedback_bindings', []).append(binding)
        for ref in state['L3']:
            card = self.journal.card(ref)
            if not self.can_read_card(task, card):
                continue
            for i, row in enumerate(card['payload']['judgment_audit']):
                if row['judgment_ref'] in bound:
                    field(ref, '/payload/judgment_audit/' + str(i), '当前对象明确关联的科研判断审计')
        for claim in state['L2'].values():
            if not set(claim['evidence_refs']) & bound:
                continue
            card = self.journal.card(claim['revision_ref'])
            if not self.can_read_card(task, card):
                continue
            index = next(i for i, row in enumerate(card['payload']['knowledge_updates']) if row['claim_id'] == claim['claim_id'])
            # conditional 是带适用条件的知识，不自动贴作反证；完整正文保留 scope/rationale。
            field(card['id'], '/payload/knowledge_updates/' + str(index), '当前对象引用证据所关联的知识及适用条件')
        context = task['inputs']['context']
        assigned = [r for key in ('feedback_refs', 'supervisor_feedback_refs') for r in context.get(key, [])]
        assigned += [r for r in task['inputs']['refs'] if self.journal.card(r)['kind'] == 'TaskCard'
                     and self.journal.card(r)['payload']['inputs'].get('purpose') == 'supervise']
        for ref in dict.fromkeys(assigned):
            card = self.require_visible(task, ref)
            pointer = '/payload/result' if card['kind'] == 'TaskCard' else '/payload'
            field(ref, pointer, '宿主为本任务明确分配的反馈')

    def context(self, task):
        task = deepcopy(task)
        # Reuse verified shared ancestry only during this fixed read snapshot, never across calls.
        task['_acl_allowed_refs'] = set()
        task['_read_view'] = {'sequence': self.journal.sequence, 'memory_version': self.journal.memory_version}
        purpose, inputs = task['inputs']['purpose'], task['inputs']
        sequence = task['_read_view']['sequence']
        memory = {'memory_version': self.journal.memory_version, 'cards': [], 'required_reads': [],
                  'read_manifest': {'refs': [], 'policy': '必需短正文直接提供；required_reads 未完成不得提交。目录不是已读正文。'}}
        context = {'run_id': self.journal.run_id, 'task_id': task['task_id'], 'task': deepcopy(inputs['context']),
                   'read_view': task['_read_view'], 'memory': memory,
                   'interface_reference': str(ROOT / 'QAViT/docs/EXECUTION.md'),
                   'constraints_source': 'user ResearchBrief; no inherited bitwidth/training defaults',
                   'validation_mode': not self.config['enable_l3'], 'repair_feedback': task.get('repair_feedback'),
                   'user_answers': task.get('user_answers', [])}
        if purpose == 'record' and context['task'].get('l3_outcome'):
            # 状态汇总不能再次夹带完整验证回执/工程日志；原文仍在本任务不可变输入中。
            for index, fact in enumerate(context['task']['l3_outcome'].get('tasks', [])):
                source = {'ref': task['task_card'], 'field': '/payload/inputs/context/l3_outcome/tasks/' + str(index)}
                value = fact.pop('result', None)
                if isinstance(value, dict):
                    fact['result_status'] = {k: value[k] for k in ('status', 'reason', 'message') if k in value}
                for key in ('error', 'result_status'):
                    if len(canonical(fact.get(key))) > 1600:
                        fact[key] = {'preview': canonical(fact[key])[:1200], 'truncated': True}
                fact['read_card'] = source
        if purpose == 'record':
            batch = inputs['context']['recording_batch']
            context['task'].pop('required_judgment_refs', None)
            context['task']['recording_batch'] = {
                'batch_id': batch['batch_id'], 'through_sequence': batch['through_sequence'],
                'source_counts': {k: len(batch[k]) for k in ('source_refs', 'scientific_refs', 'engineering_refs')},
                'source_index': {'ref': task['task_card'], 'field': '/payload/inputs/context/recording_batch'},
                'facts': 'fully_archived', 'scientific_analysis': 'pending',
                'instruction': '程序已绑定全部原件；不返回覆盖清单、不逐项审计。按科研需要深入读取。'}
            counts, failures = {}, []
            for ref in batch['engineering_refs']:
                card = self.require_visible(task, ref)
                body = card['payload']
                label = body['kind'] + '/' + body['state']
                counts[label] = counts.get(label, 0) + 1
                result = body.get('result') or {}
                if body['state'] != 'completed' or result.get('status') in ('rejected', 'failed', 'interrupted'):
                    failures.append(self.present_card(card))
            memory['engineering'] = {'counts': counts, 'failures': failures,
                                     'all_sources': context['task']['recording_batch']['source_index']}
        delivered, presented = set(), {}
        def deliver(ref, location=None):
            if ref in delivered:
                return
            card = self.require_visible(task, ref)
            view = self.present_card(card)
            text = canonical(card['payload'])
            body = view['payload'] if card['kind'] == 'EvidenceCard' else view
            # 每个角色都必须完整接收 Brief；有界全文直送省去仅为读取任务的模型回合。
            direct_limit = 2 * self.DIRECT_CHARACTERS if card['kind'] == 'ResearchBrief' else self.DIRECT_CHARACTERS
            if card['kind'] == 'ExperimentCard':
                direct_limit = 4 * self.PAGE_CHARACTERS
            if purpose == 'record' and card['kind'] in ('HypothesisCard', 'ReviewDecisionCard'):
                direct_limit = 4 * self.PAGE_CHARACTERS
            if card['kind'] == 'EvidenceCard':
                # A bounded table of scalar results must not fall back to paging megabytes of array state.
                bulk_refs = body['observations'] != card['payload']['observations']
                direct_limit = (16 if bulk_refs else 1) * self.PAGE_CHARACTERS
            if len(canonical(body)) > direct_limit:
                digest = fingerprint(card['payload'])
                stop = 0
                for start, end in sorted(self.read_ranges(task, ref, '/payload', digest)):
                    if start > stop:
                        break
                    stop = max(stop, end)
                memory['required_reads'].append({'ref': ref, 'field': '/payload', 'sha256': digest,
                    'total_characters': len(text), 'next_offset': int(stop) if stop < len(text) else None,
                    'reason': 'purpose=' + purpose + ' 的明确处理对象；须分段/同批读取完整正文'})
                view = {**self.retriever.entry(card), 'id': ref, 'required': True, 'body_location': 'read_card /payload'}
            else:
                key = 'refs' if view == card else 'projected_refs'
                memory['read_manifest'].setdefault(key, []).append(ref)
            if view != card and self.read_ranges(task, ref, '', fingerprint(card)) == [(0, float('inf'))]:
                view = {'id': ref, 'version': card['version'], 'body_location': 'tool_results complete immutable card'}
            presented[ref] = view
            if location is None:
                memory['cards'].append(view)
            else:
                container, key = location
                container[key] = view
            delivered.add(ref)
        deliver(self.brief['id'], (context, 'research_brief'))
        if purpose in ('code', 'reference', 'safeguard'):
            deliver(inputs['context']['experiment_ref'], (context, 'frozen_experiment'))
        if purpose == 'evolve':
            context['task'].pop('parent_hypothesis', None)
            deliver(inputs['context']['parent_ref'], (context['task'], 'parent_hypothesis'))
        for ref in self.required_refs(task):
            deliver(ref)
        if purpose != 'reference':
            page = self.retriever.retrieve(task['role'], sequence=sequence, visible=lambda c: self.can_read_card(task, c))
            memory['directory'] = page
            state = self.memory.snapshot(sequence)
            latest = page['latest_metareview_ref']
            self.deliver_bound_feedback(task, memory, state, latest, deliver)
            if purpose == 'supervise':
                memory['program_state'] = {k: v for k, v in state['L1'].items() if k != 'open_questions'}
            if purpose in ('record', 'generate', 'evolve', 'supervise'):
                memory['recording_status'] = self.memory.recording_status(sequence)
        budget = self.budget.input_receipt()
        context.update(available_seconds=budget['remaining_seconds'], budget_receipt=budget)
        if 'literature_fetch' in TASKS[purpose]['tools']:
            context['literature_capabilities'] = {'pdf_text_extraction': find_spec('pypdf') is not None,
                'pdf_dependency': 'pypdf', 'note': 'PDF 解析依赖未安装时只能保留原件，不能声称读过正文；HTTP/解析失败保留为未核实。'}
        if purpose in ('code', 'reference', 'safeguard'):
            context['numeric_imports'] = sorted(NUMERIC_IMPORTS)
            context['code_interface'] = (ROOT / 'QAViT/docs/CODE_INTERFACES.md').read_text()
            context['public_interface_assets'] = [str(ROOT / 'QAViT/tools/ptq_agent/execution' / name)
                                                  for name in PUBLIC_INTERFACE_FILES]
        if 'request_experiment' in TASKS[purpose]['tools']:
            from tools.profiling import PROFILE_NAMES
            context['services'] = {**SERVICES, 'profile': {**SERVICES['profile'], 'tools': list(PROFILE_NAMES)}}
            context['experiment_interface'] = (ROOT / 'QAViT/docs/EXPERIMENT_INPUTS.md').read_text()
            context['method_catalog'] = [{k: row[k] for k in ('id', 'targets', 'document')}
                for row in read_json(ROOT / 'QAViT/docs/methods/index.json')['methods']]
            draft = inputs['context'].get('experiment_draft')
            if inputs['context'].get('experiment_ref'):
                draft = self.require_visible(task, inputs['context']['experiment_ref'])['payload']
            evidence = [c for c in self.journal.find_cards('EvidenceCard') if self.can_read_card(task, c)]
            context['data_receipt'] = data_receipt(self.brief['payload'], (draft or {}).get('execution'), evidence=evidence)
            context['data_receipt']['selection_source'] = (inputs['context'].get('experiment_ref') or 'context.task.experiment_draft.execution') if draft else '尚未提交本实验数据选择'
        if purpose == 'safeguard':
            for key, artifact in (('candidate_source', inputs['context']['candidate']),
                                  ('independent_reference_source', inputs['context']['reference'])):
                self.deliver_asset(task, context, key, artifact)
        if purpose == 'code' and inputs['context'].get('revision_feedback', {}).get('previous_candidate'):
            self.deliver_asset(task, context, 'previous_candidate_source', inputs['context']['revision_feedback']['previous_candidate'])
        if purpose == 'rank':
            context['task']['matches'] = self.selection_matches(inputs['context'])
            context['task']['direct_results'] = ranking_table(context['task']['matches'], inputs['context']['eligible'])
        # 任务内嵌的长冻结草案/验证回执也为必需对象，不能只截断后放行。
        for key, value in list(context['task'].items()):
            if (key in ('recording_batch', 'required_judgment_refs', 'parent_hypothesis')
                    or (purpose == 'record' and key == 'l3_outcome')
                    or len(canonical(value)) <= self.DIRECT_CHARACTERS):
                continue
            ref, field = task['task_card'], '/payload/inputs/context/' + key
            digest, stop = fingerprint(value), 0
            for start, end in sorted(self.read_ranges(task, ref, field, digest)):
                if start > stop: break
                stop = max(stop, end)
            memory['required_reads'].append({'ref': ref, 'field': field, 'sha256': digest,
                'total_characters': len(canonical(value)), 'next_offset': int(stop) if stop < len(canonical(value)) else None,
                'reason': '当前任务的必需字段'})
            context['task'][key] = {'read_card': {'ref': ref, 'field': field}, 'required': True}
        bodies = self.complete_tool_bodies(task)
        self.coalesce_feedback(memory, presented, bodies)
        context['tool_results'] = self.compact_results(task, presented, memory.get('asset_locations', {}), memory.get('field_manifest', []), bodies)
        handle_refs = set(delivered)
        handle_refs.update(row['ref'] for row in memory.get('directory', {}).get('entries', []))
        handle_refs.update(row['read_card']['ref'] for row in memory.get('bound_feedback', []))
        for _, child, result in self.result_values(task):
            handle_refs.update(child['outputs'])
            if isinstance(result, dict):
                handle_refs.update(r for r in (result.get('ref'), result.get('id')) if r in self.journal.cards)
        memory['source_handles'] = {self.retriever.handle(r): r for r in sorted(handle_refs)
                                    if r in self.journal.card_versions and self.can_read_card(task, self.journal.card(r))}
        context['reading_instructions'] = ('按 purpose 使用已交付正文；required_reads/required_assets 是必读而非建议。'
            'Evidence 的大型纯数值数组/状态以精确字段入口和哈希交付，按科研需要读取；逐项标量统计不作摘要。'
            '完全相同的重复结构仅保留首次正文，后续引用该字段与哈希；不同统计值全部保留。'
            '可同批调用多次 read_card/read_asset 或分页 retrieve_memory；无需每次先检索。'
            '同一正文只在一个位置提供，续调保留已读分段；历史原件可按目录读取。反馈不是冻结规格或角色策略。')
        return context

    def deliver_asset(self, task, context, key, artifact):
        path = Path(artifact['path']).resolve()
        if str(path) not in self.authorized_assets(task):
            raise ContractError('必需源码不属于本调用的授权来源')
        digest = artifact['source_sha256']
        if file_sha256(path) != digest:
            raise ContractError('必需源码资产哈希改变')
        text = path.read_text()
        if len(text) <= 4 * self.PAGE_CHARACTERS:
            context[key] = text
            context['memory'].setdefault('asset_locations', {})[str(path)] = {'sha256': digest, 'location': 'context.' + key}
        else:
            stop = 0
            ranges = []
            for row, child, result in self.result_values(task):
                if (child['state'] == 'completed' and child['kind'] == 'host_tool' and child['inputs']['name'] == 'read_asset'
                        and isinstance(result, dict) and result.get('asset') == str(path) and result.get('sha256') == digest):
                    ranges.append((result['offset'], result['offset'] + len(result['text'])))
            for start, end in sorted(ranges):
                if start > stop: break
                stop = max(stop, end)
            context[key] = {'asset': str(path), 'sha256': digest, 'required': True}
            context['memory'].setdefault('required_assets', []).append({**context[key], 'total_characters': len(text),
                'next_offset': stop if stop < len(text) else None})

    def complete_tool_bodies(self, task):
        """完整且核对过的工具正文范围；选最宽范围的一处交付，不凭裸 ID。"""
        groups = {}
        paths = {str(self.journal.card_path(r)): r for r in self.journal.cards}
        for _, child, result in self.result_values(task):
            if child['kind'] != 'host_tool' or child['state'] != 'completed' or not isinstance(result, dict):
                continue
            name = child['inputs']['name']
            ref = result.get('ref', result.get('id')) if name == 'read_card' else paths.get(result.get('asset'))
            if ref not in self.journal.cards or name not in ('read_card', 'read_asset'):
                continue
            card = self.journal.card(ref)
            field = result.get('field', '') if name == 'read_card' else ''
            if name == 'read_card' and result == card:
                mode, text, start, part = 'card', canonical(card), 0, canonical(card)
            else:
                mode = 'pages' if name == 'read_card' else 'asset'
                try:
                    value = self.card_field(card, field)
                except ContractError:
                    continue
                path = self.journal.card_path(ref)
                text = canonical(value) if mode == 'pages' else path.read_text()
                digest = fingerprint(value) if mode == 'pages' else file_sha256(path)
                start, part = result.get('offset'), result.get('text')
                if (type(start) is not int or start < 0 or not isinstance(part, str)
                        or result.get('sha256') != digest or result.get('total_characters') != len(text)
                        or text[start:start + len(part)] != part):
                    continue
            key = (ref, field, mode)
            group = groups.setdefault(key, {'ref': ref, 'field': field, 'mode': mode,
                'length': len(text), 'spans': [], 'tasks': set()})
            group['spans'].append((start, start + len(part)))
            group['tasks'].add(child['task_id'])
        complete = []
        for group in groups.values():
            stop = 0
            for start, end in sorted(group['spans']):
                if start > stop: break
                stop = max(stop, end)
            if stop >= group['length']:
                complete.append(group)
        selected = []
        for group in sorted(complete, key=lambda g: (len(g['field']), {'card': 0, 'pages': 1, 'asset': 2}[g['mode']])):
            if not any(g['ref'] == group['ref'] and (not g['field'] or group['field'] == g['field']
                    or group['field'].startswith(g['field'] + '/')) for g in selected):
                selected.append(group)
        for group in selected:
            group['covered_tasks'] = set().union(*(g['tasks'] for g in groups.values()
                if g['ref'] == group['ref'] and (not group['field'] or g['field'] == group['field']
                    or g['field'].startswith(group['field'] + '/'))))
        return selected

    def coalesce_feedback(self, memory, presented, bodies):
        """展开完成后仅保留来源/版本/读取记录；正文留在实际完整交付位置。"""
        moved = set()
        for entry in memory.get('bound_feedback', []):
            ref, pointer = entry['read_card']['ref'], entry['read_card']['field']
            location = None
            if ref in presented:
                try:
                    if fingerprint(self.card_field(presented[ref], pointer)) == entry['sha256']:
                        location = 'context bound object'
                except ContractError:
                    pass
            if location is None and any(g['ref'] == ref and (not g['field'] or pointer == g['field']
                    or pointer.startswith(g['field'] + '/')) for g in bodies):
                location = 'tool_results complete body'
            if location:
                entry.pop('value', None)
                entry.update(body_location=location, version=self.journal.card(ref)['version'])
                moved.add((ref, pointer))
        memory['field_manifest'] = [f for f in memory.get('field_manifest', []) if (f['ref'], f['field']) not in moved]

    def compact_results(self, task, delivered, asset_locations=None, delivered_fields=(), complete_bodies=None):
        result, seen = [], set()
        asset_locations = asset_locations or {}
        values = list(self.result_values(task))
        complete_bodies = self.complete_tool_bodies(task) if complete_bodies is None else complete_bodies
        paths = {str(self.journal.card_path(r)): r for r in self.journal.cards}
        full_refs = {r for r, view in delivered.items() if view == self.journal.card(r)}
        full_refs |= {v['id'] for _, _, v in values if isinstance(v, dict) and v.get('id') in self.journal.cards
                      and v == self.journal.card(v['id'])}
        direct_full = {r for r, view in delivered.items() if view == self.journal.card(r)}
        card_paths = {str(self.journal.card_path(ref)): ref for ref in full_refs}
        def equivalent_field(value):
            if not isinstance(value, dict) or 'text' not in value or value.get('ref') not in delivered:
                return False
            try:
                shown = self.card_field(delivered[value['ref']], value.get('field', ''))
            except ContractError:
                return False
            return fingerprint(shown) == value.get('sha256') and verified_page(value, value['ref'])
        def verified_page(value, ref, asset=False):
            card = self.journal.card(ref)
            try:
                field = self.card_field(card, value.get('field', ''))
            except ContractError:
                return False
            path = self.journal.card_path(ref)
            text = path.read_text() if asset else canonical(field)
            digest = file_sha256(path) if asset else fingerprint(field)
            start, part = value.get('offset'), value.get('text')
            return (type(start) is int and start >= 0 and isinstance(part, str)
                    and value.get('sha256') == digest and value.get('total_characters') == len(text)
                    and text[start:start + len(part)] == part)
        for row, child, value in values:
            item = {k: deepcopy(row[k]) for k in ('task', 'state', 'call_id', 'name', 'outputs', 'error') if k in row}
            item['source_ref'] = child['task_card']
            if child['kind'] == 'host_tool':
                args = child['inputs']['arguments']
                item['arguments'] = deepcopy(args) if len(canonical(args)) <= 1600 else {
                    'read_card': {'ref': child['task_card'], 'field': '/payload/inputs/arguments'}}
            elif child['kind'] == 'literature':
                item['arguments'] = {k: child['inputs'][k] for k in ('query', 'url') if k in child['inputs']}
            ref = value.get('ref', value.get('id')) if isinstance(value, dict) else None
            field = value.get('field', '') if isinstance(value, dict) else ''
            if isinstance(value, dict) and value.get('asset') in paths:
                ref, field = paths[value['asset']], ''
            covering = next((g for g in complete_bodies if child['task_id'] in g['covered_tasks'] and g['ref'] == ref and
                (not g['field'] or field == g['field'] or field.startswith(g['field'] + '/'))), None)
            if covering and child['task_id'] not in covering['tasks']:
                item['result'] = {'ref': ref, 'field': field, 'body_location': 'tool_results complete ancestor',
                                  'ancestor_field': covering['field']}
            elif isinstance(value, dict) and value.get('id') in direct_full and value == self.journal.card(value['id']):
                item['result'] = {'ref': value['id'], 'body_location': 'context bound object'}
            elif equivalent_field(value):
                item['result'] = {'ref': value['ref'], 'field': value.get('field', ''), 'body_location': 'context equivalent field'}
            elif isinstance(value, dict) and 'text' in value and any(
                    value.get('ref') == f['ref'] and value.get('field') == f['field']
                    and value.get('sha256') == f['sha256'] for f in delivered_fields) and verified_page(value, value['ref']):
                item['result'] = {'ref': value['ref'], 'field': value['field'], 'body_location': 'context bound feedback field'}
            elif isinstance(value, dict) and value.get('ref') in full_refs and verified_page(value, value['ref']):
                item['result'] = {'ref': value['ref'], 'body_location': 'context bound object or complete tool result'}
            elif isinstance(value, dict) and value.get('asset') in card_paths and verified_page(value, card_paths[value['asset']], True):
                item['result'] = {'ref': card_paths[value['asset']], 'body_location': 'context bound object or complete tool result'}
            elif isinstance(value, dict) and value.get('asset') in asset_locations and value.get('sha256') == asset_locations[value['asset']]['sha256']:
                item['result'] = {'asset': value['asset'], 'body_location': asset_locations[value['asset']]['location']}
            else:
                digest = fingerprint(value)
                if digest in seen:
                    item['result'] = {'same_result_sha256': digest}
                else:
                    item['result'] = deepcopy(value)
                    seen.add(digest)
            result.append(item)
        return result

    @staticmethod
    def asset_references(value):
        """仅跟随记录的文件引用，不将整个运行目录变成读取权限。"""
        found = {}
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('path', 'asset', 'raw_asset', 'receipt', 'worker_log', 'full_card_asset', 'complete_result_asset') and isinstance(item, str):
                    path = Path(item)
                    path = (path if path.is_absolute() else ROOT / path).resolve()
                    # literature.asset 是提取 JSON，其 sha256 指原件，不能混作 JSON 身份。
                    found[str(path)] = value.get('source_sha256', value.get('sha256')) if key in ('path', 'raw_asset') else None
                else:
                    found.update(RoleInteraction.asset_references(item))
        elif isinstance(value, list):
            for item in value:
                found.update(RoleInteraction.asset_references(item))
        return found

    def authorized_assets(self, task):
        if 'assets' in self.read_view(task):
            return self.read_view(task)['assets']
        task = dict(task, _acl_allowed_refs=set())
        allowed = {}
        purpose, sequence = task['inputs']['purpose'], self.read_view(task)['sequence']
        public = {str((ROOT / 'QAViT/tools/ptq_agent/execution' / name).resolve())
                  for name in PUBLIC_INTERFACE_FILES} if purpose in ('code', 'reference', 'safeguard') else set()
        static = self.read_view(task).get('static_assets')
        if static is None:
            paths = [p for p in (ROOT / 'QAViT/docs').rglob('*') if p.suffix in ('.md', '.json', '.txt')]
            paths += [Path(p) for p in self.brief['payload'].get('source_documents', []) if isinstance(p, str)]
            static = {str(p.resolve()): file_sha256(p) for p in paths if p.is_file() and
                      any(root in p.resolve().parents for root in (ROOT / 'reference', ROOT / 'QAViT/docs'))}
        allowed.update(static)
        allowed.update({path: file_sha256(path) for path in public})
        cards = [self.journal.card(ref) for ref, entry in self.journal.cards.items() if entry['sequence'] <= sequence]
        tasks = self.visible_task_snapshots(sequence)
        # Ownership comes from host outputs, never from arbitrary path mentions in science prose.
        # The first committed owner wins: a visible re-reference cannot launder a private file.
        owners = {str(self.journal.card_path(c['id']).resolve()): (c, file_sha256(self.journal.card_path(c['id']))) for c in cards}
        def register(owner, values):
            for path, digest in self.asset_references(values).items():
                owners.setdefault(path, (owner, digest))
        for card in cards:
            body = card['payload']
            if card['kind'] == 'EvidenceCard' and card['author'] == 'program':
                register(card, body.get('artifacts', []))
            elif card['kind'] == 'TaskCard':
                current = tasks.get(body['task_id'])
                owner = self.journal.card(current['task_card']) if current else card
                if body['inputs'].get('purpose') in ('code', 'reference'):
                    register(owner, (body.get('result') or {}).get('artifact'))
                if body['kind'] in ('literature', 'experiment', 'verification'):
                    register(owner, body.get('result'))
                    receipts = (body.get('result') or {}).get('receipts', {})
                    if isinstance(receipts, dict):
                        register(owner, [{'path': p} for p in receipts.values() if isinstance(p, str)])
                if body.get('request_path'):
                    directory = Path(body['request_path']).parent
                    register(owner, [{'path': str(directory / name)} for name in ('request.json', 'result.json', 'worker.log')])
        for path, (owner, digest) in owners.items():
            if self.can_read_card(task, owner) and (purpose != 'reference' or path == str(self.journal.card_path(owner['id']).resolve())):
                allowed[path] = digest
        if purpose != 'reference':
            # 发布后的测量原件随 Evidence 开放；仅允许真正产出该 Evidence 的数值任务。
            # 科研正文提及的任意路径、候选源码或另一分支的私有证据不能借此开放。
            for card in cards:
                if card['kind'] != 'EvidenceCard' or card['author'] != 'program' or not self.can_read_card(task, card):
                    continue
                for item in card['payload']['artifacts']:
                    path = str(Path(item['path']).resolve())
                    owner, _ = owners.get(path, (card, None))
                    body = owner['payload']
                    if (owner['id'] == card['id'] or (owner['kind'] == 'TaskCard' and body['kind'] == 'experiment'
                            and card['id'] in body['outputs']
                            and body['inputs'].get('experiment_ref') == card['payload']['experiment_ref'])):
                        allowed[path] = item['sha256']
        if purpose != 'reference':
            # Code manifests are written and content-bound by the host. A mere path/hash pair
            # is insufficient; the manifest's frozen experiment must itself be visible.
            candidates = self.asset_references(self.bound_code_sources(task))
            for card in cards:
                if card['kind'] == 'TaskCard' and self.can_read_card(task, card):
                    candidates.update(self.asset_references((card['payload'].get('result') or {}).get('artifact')))
            for path in candidates:
                # Manifest validation proves identity, not permission to bypass its producer.
                if path in owners and not self.can_read_card(task, owners[path][0]):
                    continue
                source = Path(path)
                if source.name != 'implementation.py' or source.parent.parent != self.run_dir / 'code':
                    continue
                manifest_path = source.with_name('manifest.json')
                if not manifest_path.is_file():
                    continue
                manifest = read_json(manifest_path)
                ref = manifest.get('experiment_ref')
                if ref not in self.journal.cards:
                    continue
                spec = self.journal.card(ref)
                if (self.can_read_card(task, spec) and manifest.get('path') == str(source)
                        and manifest.get('experiment_sha256') == spec['content_sha256']
                        and manifest.get('source_sha256') == file_sha256(source)):
                    allowed[path] = manifest['source_sha256']
        return {path: digest or file_sha256(path) for path, digest in allowed.items() if Path(path).is_file()
                and (path in public or any(root in Path(path).parents
                                           for root in (self.run_dir, ROOT / 'reference', ROOT / 'QAViT/docs')))}

    def host_tool(self, task):
        inputs = task['inputs']
        name, args = inputs['name'], inputs['arguments']
        parent = self.queue.get(inputs['parent_task'])
        if 'read_view' in inputs:
            parent['_read_view'] = inputs['read_view']
        schemas = {t['name']: t['parameters'] for t in tools_for(parent['inputs']['purpose'])}
        if name not in schemas:
            raise ContractError('角色调用了未授权工具')
        validate_schema(args, schemas[name], name)
        if name == 'read_card':
            card = self.require_visible(parent, args['ref'])
            field = args.get('field', '')
            value = self.card_field(card, field)
            text = canonical(value)
            if not field and 'offset' not in args and 'limit' not in args and len(text) <= self.PAGE_CHARACTERS:
                return card
            start, limit = args.get('offset', 0), args.get('limit', self.PAGE_CHARACTERS)
            return {'ref': card['id'], 'field': field, 'sha256': fingerprint(value), 'offset': start,
                    'text': text[start:start + limit], 'total_characters': len(text),
                    'next_offset': start + limit if start + limit < len(text) else None}
        if name == 'retrieve_memory':
            query = deepcopy(args['query'])
            if 'evidence_ids' in query:
                aliases = {self.retriever.handle(r): r for r in self.journal.card_versions}
                query['evidence_ids'] = [aliases.get(r, r) for r in query['evidence_ids']]
                # 查询沿原有可见性过滤返回空集，不因输入私有/未知 ID 泄露其存在性。
            return self.retriever.retrieve(parent['role'], query=query, after_version=args.get('after_version'),
                offset=args.get('offset', 0), limit=args.get('limit', 12), sequence=self.read_view(parent)['sequence'],
                visible=lambda c: self.can_read_card(parent, c))
        if name == 'read_asset':
            path = Path(args['asset'])
            path = (path if path.is_absolute() else ROOT / path).resolve()
            if not path.is_file() and not Path(args['asset']).is_absolute():
                path = (ROOT / 'QAViT/docs' / args['asset']).resolve()
            allowed = self.authorized_assets(parent)
            if str(path) not in allowed or not path.is_file():
                raise ContractError('资产读取限本调用已提交授权来源；独立参考/立场隔离禁止目录旁路')
            expected = allowed[str(path)]
            digest = file_sha256(path)
            if expected and digest != expected:
                raise ContractError('资产内容已变化，不属于本调用冻结可见版本')
            try:
                text = path.read_text(encoding='utf-8')
            except UnicodeError as exc:
                raise ContractError('read_asset 仅支持 UTF-8 文本') from exc
            start, limit = args.get('offset', 0), args.get('limit', self.PAGE_CHARACTERS)
            return {'asset': str(path), 'sha256': digest, 'offset': start, 'text': text[start:start + limit],
                    'total_characters': len(text), 'next_offset': start + limit if start + limit < len(text) else None}
        raise ContractError('未知宿主工具')


    def receive_tools(self, task, response, receipt_path):
        task = deepcopy(task)
        task['_read_view'] = self.read_view(task)
        allowed = {t["name"]: t["parameters"] for t in tools_for(task["inputs"]["purpose"])}
        ids, call_ids, drafts = [], set(), {}
        validate_schema(response["tool_calls"], {"type": "array", "minItems": 1, "items": {
            "type": "object", "required": ["id", "name", "arguments"],
            "properties": {"id": TEXT, "name": TEXT, "arguments": {"type": "object"}}}}, "tool_calls")
        # 先检查整个批次，避免后一个草案被拒绝时留下前一个已入队的实验。
        for call in response["tool_calls"]:
            if call["name"] not in allowed or call["id"] in call_ids:
                raise ContractError("角色调用了未授权工具或重复工具调用ID")
            call_ids.add(call["id"])
            validate_schema(call["arguments"], allowed[call["name"]], call["name"])
            if call["name"] == "request_experiment":
                drafts[call["id"]] = self.broker.preflight(self.resolve_card_refs(task, call["arguments"]["experiment"]))
        for call in response["tool_calls"]:
            key = task["key"] + f"/tools/a{task['attempt']}/" + fingerprint(call["id"])[:12]
            name, args = call["name"], call["arguments"]
            if name == "request_experiment":
                identifier = self.queue.enqueue(key, "experiment_request", role="supervisor", inputs={
                    "draft": drafts[call["id"]], "requester": task["role"], "refs": task["inputs"]["refs"],
                    "call_id": call["id"], "parent_task": task["task_id"]})
            elif name in ("literature_search", "literature_fetch"):
                identifier = self.queue.enqueue(key, "literature", inputs={
                    "action": "search" if name.endswith("search") else "fetch", **args, "call_id": call["id"],
                    "parent_task": task['task_id']})
            else:
                identifier = self.queue.enqueue(key, "host_tool", inputs={"name": name, "arguments": args,
                                      "parent_task": task["task_id"], "call_id": call["id"], "read_view": self.read_view(task)})
            ids.append(identifier)
        conversation = task.get("conversation", []) + [{"role": "assistant", "content": {"tool_calls": response["tool_calls"],
                                                                                           "receipt": str(receipt_path)}}]
        self.wait_for(task, ids, conversation=conversation, last_memory_read=task["memory_version"])


    def validate_review(self, task, payload):
        purpose = task["inputs"]["purpose"]
        # 身份由 bind_submission 核对，screening 结构/资格一致性由完整卡片合同核对。
        if purpose == "position":
            for field in ("arguments", "objections", "responses_to_prior", "key_predictions"):
                if field not in payload:
                    raise ContractError("辩论立场缺少 " + field)
        if purpose == "arbitrate":
            validate_verdict(payload, task["inputs"]["context"]["exchange"])
            if payload["outcome"] == "ineligible":
                raise ContractError("硬性不合格由筛查阶段决定，辩论不能偷偷跳过审查")
        if purpose in ("experiment_approval", "safeguard") and type(payload.get("approved")) is not bool:
            raise ContractError("批准结果必须为 boolean")
        if purpose == "safeguard":
            for field in ("fidelity", "numerical_correctness", "experimental_validity"):
                if not payload.get(field):
                    raise ContractError("Safeguard 缺少完整三项审查")
            if payload["approved"] and task["inputs"]["context"]["numerical"]["status"] != "passed":
                raise ContractError("实际数值验证失败，LLM 不得放行")


    def scientific_refs(self, payload, task):
        refs = set(payload.get("evidence_refs", []))
        for row in (payload.get("knowledge_updates", []) + payload.get("judgment_audit", [])
                    + payload.get('predictive_metrics', []) + payload.get('changed_judgments', [])):
            refs.update(row.get("evidence_refs", []))
            if row.get("judgment_ref"):
                refs.add(row["judgment_ref"])
        for ref in payload.get("evidence_refs", []):
            if self.journal.card(ref)["kind"] != "EvidenceCard":
                raise ContractError("evidence_refs 只能指向真实 EvidenceCard，不能用假设冒充测量")
        for row in payload.get("feedback_used", []):
            if row.get("metareview_ref"):
                refs.add(row["metareview_ref"])
        for claim_id, revision in self.prior_basis_pairs(payload):
            self.prior_basis_field(task, claim_id, revision)
            refs.add(revision)
        for ref in refs:
            self.journal.card(ref)
        return refs

    @staticmethod
    def rejected_draft(task):
        feedback = task.get('repair_feedback', {})
        if feedback.get('task_id', task['task_id']) != task['task_id']:
            return None
        response = feedback.get('rejected_response')
        decision = response.get('decision') if isinstance(response, dict) else None
        content = decision.get('content') if isinstance(decision, dict) else None
        return content if isinstance(content, dict) else None


    def submit(self, task, content, receipt_path, *, author=None):
        if not isinstance(content, dict):
            raise ContractError("submit.content 必须是对象")
        # 一次提交共用同一个已保存调用快照，避免每个来源都重读整份模型请求。
        task = deepcopy(task)
        task['_read_view'] = self.read_view(task)
        purpose, context = task["inputs"]["purpose"], task["inputs"]["context"]
        if task.get('request_path'):
            delivered = read_json(task['request_path'])['spec']['context']['memory']
            missing_reads = [r for r in delivered.get('required_reads', []) + delivered.get('required_assets', [])
                             if r['next_offset'] is not None]
            if missing_reads:
                raise ContractError('尚有未读完的必需材料；按 required_reads/required_assets 分段读取后提交: ' + canonical(missing_reads))
        if purpose == 'record' and 'repair_patch' in content:
            prior = self.rejected_draft(task)
            if not isinstance(prior, dict) or not isinstance(content['repair_patch'], dict) or set(content) != {'repair_patch'}:
                raise ContractError('Recorder repair_patch 仅能修复本任务上次被拒的完整提交')
            if any(key.startswith(field + suffix) for key in content['repair_patch']
                   for field in PAYLOAD_SCHEMAS['MetareviewCard']['properties'] for suffix in ('.', '[')):
                raise ContractError('repair_patch 仅替换 Meta-review 顶层字段，不支持 knowledge_updates[0].scope 等路径；'
                                    '请用 {repair_patch:{knowledge_updates:[完整修正数组]}} 或提交完整正文')
            content = {**deepcopy(prior), **content['repair_patch']}
        schema = submission_schema(purpose)
        if schema and purpose != 'plan_l3':
            validate_schema(content, schema, 'scientific_submission')
            content = self.bind_submission(task, content)
        if purpose == 'record' and self.queue.get(task['task_id'])['state'] == 'completed':
            prior = self.result_card(self.queue.get(task['task_id']))
            if prior and prior['payload'] == content:
                return
            raise ContractError('已发布 Recorder 批次不可改写')
        if purpose in ("rank", "evolve") and self.queue.get(task["task_id"])["state"] == "completed":
            raise ContractError("选择或 H4 已持久化，不能重新提交并更换父子身份")
        if purpose in ("position", "arbitrate"):
            if "-".join(context["pair"]) in self.matches(context["round"]):
                raise ContractError("本场已结案，不能再次提交立场或改写最终裁决（包括 tie/undetermined）")
            if self.queue.get(task["task_id"])["state"] == "completed":
                raise ContractError("本次交锋已提交，不能重复提交或更换裁决")
        if purpose == "rank":
            if content.get("verdict_revisions"):
                raise ContractError("verdict_revisions：已结案裁决不可改写，包括 tie/undetermined；后到证据只能用于科研分析")
            # 接受旧客户端的空字段，但不再生成或保存该冗余字段；历史卡片不变。
            content = {k: v for k, v in content.items() if k != "verdict_revisions"}
        refs = set(task["inputs"]["refs"])
        cards, events = [], []
        if purpose in ("code", "reference"):
            experiment = self.journal.card(context["experiment_ref"])
            artifact = write_artifact(self.run_dir, experiment, content, task["role"])
            self.queue.finish(task["task_id"], result={"artifact": artifact, "receipt": str(receipt_path)})
            return
        if purpose == "supervise":
            self.queue.finish(task["task_id"], result=content)
            return
        if purpose == "plan_l3":
            content = self.broker.preflight(self.resolve_card_refs(task, content))
            if content["level"] != "L3":
                raise ContractError("正式计划必须明确 L3")
            self.queue.finish(task["task_id"], result={"draft": content})
            return
        kind = TASKS[purpose]["output"]
        validate_payload(kind, content)
        refs.update(self.scientific_refs(content, task))
        for ref in refs:
            self.require_visible(task, ref)
        parents = []
        if kind == "HypothesisCard":
            if purpose == "evolve":
                parents = content.get("parent_refs", [])
                selection = self.result_card(self.tasks().get(f"r{context['round']:03d}/select_parent"))
                if (not selection or selection["id"] != context.get("parent_selection_ref")
                        or selection["payload"]["selected_hypothesis_ref"] != context.get("parent_ref")
                        or parents != [context.get("parent_ref")] or not context.get("parent_ref")):
                    raise ContractError("H4.parent_refs 必须仅含已持久化 H* 的完整卡片 ID，不得更换、多父或省略版本")
                refs.update(parents)
                if not content.get("improvement_objective"):
                    raise ContractError("H4 需要明确基于辩论/证据的改进目标")
            assigned = [r for r in task['inputs']['refs'] if self.journal.cards[r]['kind'] == 'MetareviewCard']
            if purpose == "generate" and assigned:
                if not any(row.get("metareview_ref") == assigned[-1] and row.get("change") for row in content["feedback_used"]):
                    raise ContractError("下一轮 Generation 必须具体说明如何使用上一轮 Meta-review，不接受仅列引用")
        if kind == "ReviewDecisionCard":
            self.validate_review(task, content)
        if purpose == "rank":
            matches = self.selection_matches(context)
            table = ranking_table(matches, context["eligible"])
            selected = content.get("selected_slot")
            if selected not in context["eligible"] and not (selected is None and not context["eligible"]):
                raise ContractError("所选方案未通过资格审查")
            if (not isinstance(content.get("selection_basis"), str) or not content["selection_basis"].strip()
                    or type(content.get("provisional")) is not bool):
                raise ContractError("排名必须说明选择依据与是否暂定")
            winners = [slot for slot, wins in table["wins"].items() if wins == len(context["eligible"]) - 1]
            if winners and selected != winners[0]:
                raise ContractError("必须选择已结案裁决中直接战胜其余所有合格方案的全胜者")
            if table["requires_provisional_basis"] and not content["provisional"]:
                raise ContractError("循环胜负/未决问题尚存，不能伪造确定全序")
            if selected and selected not in table["groups_by_direct_results"][0]:
                raise ContractError("不能违背无争议的直接交锋优胜结果")
            bindings = {"selection_stage": context["selection_stage"],
                        "selected_hypothesis_ref": context["candidate_refs"].get(selected),
                        "parent_selection_ref": context.get("parent_selection_ref")}
            for key, value in bindings.items():
                if key in content and content[key] != value:
                    raise ContractError(key + " 与当前选择阶段绑定的完整卡片身份冲突")
            refs.update(context["candidate_refs"].values())
            if bindings["parent_selection_ref"]:
                refs.add(bindings["parent_selection_ref"])
            content = {**content, **bindings, "direct_results": table, "candidate_refs": context["candidate_refs"],
                       "selection_source": "existing_verdicts_and_screening" if author == "program" else "judger"}
        staged = None
        if purpose == 'record':
            staged = self.run_dir / 'recording_submissions' / (task['task_id'] + f"-{task['attempt']}-{task['step']}.json")
        if staged and staged.exists():
            card = read_json(staged)
            from core.cards import validate_card
            validate_card(card)
            if card['payload'] != content or card['refs'] != sorted(refs) or card['author'] != 'recorder':
                raise ContractError('已有通过校验但尚未发布的 Recorder 草稿；先恢复同一提交，不能覆盖原件')
        else:
            card = make_card(kind, self.journal.run_id, content, author=author or task["role"], parents=parents, refs=sorted(refs))
        cards.append(card)
        if kind == "HypothesisCard":
            events.append(self.memory.live_update(round_number=context["round"], hypothesis_ref=card["id"]))
        if purpose == "rank" and context["selection_stage"] == "final":
            events.append(self.memory.live_update(round_number=context["round"], selection={"slot": content["selected_slot"], "review_ref": card["id"]}))
        if purpose == "record":
            if content["round"] != context["round"]:
                raise ContractError("Meta-review 轮次错误")
            batch = context['recording_batch']
            if any(e['type'] == 'recorder_batch_completed' and e['batch_id'] == batch['batch_id'] for e in self.journal.events):
                raise ContractError('Recorder 批次已发布，不能重复整理或移动游标')
            events.append(self.memory.apply_metareview(card,
                hypothesis_refs=[c["id"] for c in self.round_hypotheses(context["round"]).values()],
                evidence_refs=content["evidence_refs"], selection={"slot": context["selection"], "review_ref": context["selection_ref"]}))
            events.append({'type': 'recorder_batch_completed', **batch, 'metareview_ref': card['id'],
                           'analysis_scope': content['analysis_scope'],
                           'analyzed_judgment_refs': sorted({r['judgment_ref'] for r in content['judgment_audit']})})
            if not staged.exists():
                atomic_json(staged, card)  # 可恢复的待发布响应，不是事实/成功游标。
        self.queue.finish(task["task_id"], outputs=[card["id"]], result={"receipt": str(receipt_path) if receipt_path is not None else None}, cards=cards, events=events)

