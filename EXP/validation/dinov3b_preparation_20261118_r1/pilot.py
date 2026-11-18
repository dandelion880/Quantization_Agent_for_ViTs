"""Engineering pilot of one source-traced mature plan, outside all research rounds."""
from pathlib import Path
import argparse
import sys
import time

HERE = Path(__file__).resolve().parent
import os
SOURCE = Path(os.environ.get('QAVIT_PILOT_SOURCE', HERE.parents[2] / 'QAViT'))
sys.path.insert(0, str(SOURCE))
from runtime.paths import ROOT
from core.journal import atomic_json, read_json
from runtime.identity import code_identity
from core.cards import fingerprint
from tools.experiment_worker import backend_and_data, evaluate_services
from ptq_agent.execution.baselines import build_baseline
from tools.vision_models import setup_segmentation


def pilot(name, out, count):
    out.mkdir(parents=True, exist_ok=False)
    setup_segmentation(42, 3)
    start = time.monotonic()
    cfg = read_json(HERE.parents[2] / "EXP/inputs/dinov3b_static_w4a4/input.json")
    brief = cfg["research_brief"]
    backend, options, data = backend_and_data({"backend": brief["model"]["adapter"]})
    from seed import build, partitioned
    if name in ('group32_log2','partitioned'):
        plan = (partitioned if name == 'partitioned' else build)(backend)
    else:
        plan = build_baseline(backend, name)
    spec = {"service": "quantize_evaluate", "plan": plan, "resolved_choices": {"training": False},
            "quantization_contract": brief["quantization_contract"],
            "calibration": {"split": "calibration", "count": count},
            "evaluations": [{"kind": "classification", "split": "search", "count": 24},
                            {"kind": "ade_frozen", "count": 2}]}
    card = {"id": "engineering-pilot", "payload": {"level": "L2", "resources": {"cpu_threads": 3}, "execution": spec}}
    atomic_json(out / "input.json", {"experiment": card, "source": code_identity(), "plan_sha256": fingerprint(plan)})
    result = evaluate_services(card, backend, options, data, out, None)
    atomic_json(out / "pilot.json", {"seed": name, "seconds": time.monotonic() - start,
                                     "observations": result["observations"], "model_ref": result["model_ref"]})
    print([(k, v["metrics"]) for k, v in result["observations"]["evaluations"].items()], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("out", type=Path)
    p.add_argument("--calibration", type=int, default=8)
    a = p.parse_args()
    pilot(a.name, a.out, a.calibration)
