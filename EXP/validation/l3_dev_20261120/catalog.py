"""Attach the two measured dev views to their exact historical model identities."""
from pathlib import Path
from core.journal import Journal, atomic_json, read_json
from core.solutions import SolutionLibrary
from runtime.paths import ROOT
from tools.experiments import ExperimentBroker
from core.cards import fingerprint
out = Path(__file__).resolve().parent
j = Journal(out/'journal','development_20261120')
b = j.find_cards('ResearchBrief')[0]
broker = ExperimentBroker(j, b['id'], None)
class ValidationLibrary(SolutionLibrary):
    def seeds(self):
        return []  # This validation journal contains exactly the two referenced models.

library = ValidationLibrary(j, b['payload'], shared=False)
# Worker implementation identity is in each receipt. Source archives and original
# code for model reuse are read from the existing official entries, never rebuilt.
old = [read_json(next((ROOT/'EXP/runs').glob('dino*/solution_library/'+pattern+'*.json'))) for pattern in ('7c18','7f70')]
if not any(e['type']=='solution_archive' for e in j.events):
    j.commit('original-models',events=[{'type':'solution_archive','entries':old}])
for e in j.find_cards('EvidenceCard'):
    exp = j.card(e['payload']['experiment_ref'])
    source = next(r for r in old if r['id'] == e['payload']['provenance']['model_ref'])
    # The standalone validation journal has no Workflow research_run.json.
    # Use the exact archived model source. Evaluator version remains in the measurement.
    class ReplayLibrary(ValidationLibrary):
        def source_record(self, sequence=None):
            return {k:source[k] for k in ('source_version','source_snapshot','source_archive_sha256')}
    ReplayLibrary(j,b['payload'],shared=False).consider_round(None,e,exp,None,validator=broker.validate_evidence_for)
index = []
for r in library.available():
    if r.get('development'):
        assert r['metrics']==next(row['metrics'] for row in old if row['id']==r['id'])
        index.append({k:r[k] for k in ('id','metrics','development')})
assert len(index)==2
atomic_json(out/'catalog.json',index)
