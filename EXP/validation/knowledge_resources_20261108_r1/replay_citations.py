"""Read-only historical real outputs replayed through current scripted submission.

No LLM or measured science; only the historical Brief ID is rebound to this run.
"""
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'QAViT'))
from core.workflow import Workflow
from core.journal import read_json, atomic_json, file_sha256
from tests.support import validation_config
from tests.test_memory_delivery import DeliveryRegression

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / 'scientific_skills_20261108_r1'


def main():
    result = []
    helper = DeliveryRegression(methodName='runTest')
    for name in ('A1_skills', 'C1_skills'):
        original = read_json(OLD / (name + '.json'))['output']
        payload = deepcopy(original['payload'])
        issues = payload.pop('submission_issues', [])
        for issue in sorted(issues, key=lambda row: row['index']):
            assert issue['field'] == 'sources'
            payload['sources'].insert(issue['index'], issue['submission'])
        config = validation_config()
        config['research_brief'] = read_json(OLD / 'live' / name / 'research_run.json')['config']['research_brief']
        wf = Workflow(BASE / 'replay' / name, config)
        for row in payload['sources']:
            if row.get('ref', '').startswith('ResearchBrief-'):
                row['ref'] = wf.brief['id']
        tid = wf.enqueue_role('replay/' + name, 'generate', refs=[wf.brief['id']],
                              context={'round': 1, 'slot': 'H3', 'hypothesis_type': 'new_algorithm'})
        task, _ = helper.begin(wf, wf.queue.get(tid))
        documents = [row['ref'] for row in payload['sources']
                     if row.get('ref', '').endswith('.md')]
        task, _ = helper.tools(wf, task, [('read_asset', {'asset': path}) for path in documents])
        wf = Workflow(wf.run_dir)
        task = wf.queue.get(tid)
        wf.submit(task, payload, 'SCRIPTED historical real-output replay; no new LLM')
        card = wf.result_card(wf.queue.get(tid))
        updated = card['payload']
        assert updated['mechanism'] == original['payload']['mechanism']
        assert len([s for s in updated['sources'] if 'document' in s]) == len(documents)
        remaining = updated.get('submission_issues', [])
        assert len(remaining) == (1 if name == 'C1_skills' else 0)
        assert not wf.journal.find_cards('EvidenceCard')
        result.append({'case': name, 'original_sha256': file_sha256(OLD / (name + '.json')),
                       'original_source_issues': len(issues), 'document_citations_preserved': len(documents),
                       'remaining_issues': remaining, 'mechanism_unchanged': True,
                       'actual_read_asset_calls': len(documents), 'card': card,
                       'status': wf.queue.get(tid)['state']})
    atomic_json(BASE / 'citation_replay.json', {'real_llm': False, 'measurement_source': None,
                'historical_inputs_modified': False, 'cases': result})
    print([(r['case'], r['status'], r['original_source_issues'], len(r['remaining_issues'])) for r in result])


if __name__ == '__main__':
    main()
