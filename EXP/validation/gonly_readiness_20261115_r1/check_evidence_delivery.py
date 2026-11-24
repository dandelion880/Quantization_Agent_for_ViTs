"""Read-only replay of real evidence delivery; no numerical experiment or LLM call."""
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / 'QAViT'))
from core.cards import canonical, fingerprint
from core.evidence import evidence_facts, project_observations

SOURCE = PROJECT / 'EXP/runs/deitb_static_w4a4_gonly_20261115_r1'
OUTPUT = Path(__file__).resolve().parent

def main():
    before, deliveries = {}, []
    for path in sorted((SOURCE / 'cards').glob('*.json')):
        raw = path.read_bytes()
        card = json.loads(raw)
        if card['kind'] != 'EvidenceCard':
            continue
        before[str(path)] = hashlib.sha256(raw).hexdigest()
        for artifact in card['payload'].get('artifacts', []):
            asset = Path(artifact.get('path', ''))
            if asset.name == 'segmentation.json' and asset.is_file():
                before[str(asset)] = hashlib.sha256(asset.read_bytes()).hexdigest()
        facts = evidence_facts(card, SOURCE)
        view = project_observations(card['payload']['observations'], card['id'])
        deliveries.append({'ref': card['id'], 'key_facts': facts, 'observations': view,
                           'source_bytes': len(raw), 'projection_characters': len(canonical({'key_facts': facts, 'observations': view}))})
    assert len(deliveries) == 5
    rows = {row['ref']: row for row in deliveries}
    first = rows['EvidenceCard-15a015a9c39044759caf9a9edd7a0c22@1']['key_facts']
    last = rows['EvidenceCard-e643fc1b1e154c17a7553394f97281f1@1']['key_facts']
    for facts in (first, last):
        assert facts['evaluations']['classification']['sample_count'] == 50000
        assert facts['evaluations']['segmentation']['sample_count'] == 2000
        assert facts['evaluations']['segmentation']['paired_metrics']['mIoU']['FP'] == 39.8295824953736
        assert facts['evaluations']['segmentation']['paired_metrics']['mIoU']['delta_Q_minus_FP'] < 0
        assert facts['protocol_hash'] and facts['identity']['model_ref']
        assert facts['reload']['host_verified'] is True
    local = rows['EvidenceCard-71a683a780904b70b1cbe91282a3378c@1']['key_facts']['arm_results']
    assert {key: arm['whole_network']['top1'] for key, arm in local.items()} == {
        'fp': 96.09375, 'seed': 89.453125, 'pv': 87.5, 'mass': 91.015625}
    changes = [name for name, digest in before.items() if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
    assert not changes
    (OUTPUT / 'evidence_delivery_views.json').write_text(json.dumps(deliveries, ensure_ascii=False, indent=2) + '\n')
    report = {'status': 'passed', 'kind': 'read_only_projection_of_real_historical_evidence',
              'source_run': str(SOURCE), 'evidence_count': len(deliveries), 'new_measurements': 0,
              'immutable_sources_checked': before, 'changed_sources': changes,
              'checks': ['FP/Q/delta/unit/full scope and identity', 'actual L2 arms remain distinct', 'immutable raw Evidence and ADE assets'],
              'sizes': [{k: row[k] for k in ('ref', 'source_bytes', 'projection_characters')} for row in deliveries],
              'views_sha256': hashlib.sha256((OUTPUT / 'evidence_delivery_views.json').read_bytes()).hexdigest()}
    (OUTPUT / 'evidence_delivery_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'immutable_sources_checked'}, ensure_ascii=False))

if __name__ == '__main__':
    main()
