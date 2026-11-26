"""Bind real local observations and read the Recorder's exact version back via the existing Journal."""

from pathlib import Path
from runtime.paths import ROOT
from core.cards import make_card, fingerprint
from core.journal import Journal, read_json, atomic_json, file_sha256
from core.memory import ResearchMemory
from tools.experiments import asset

OUT = Path(__file__).resolve().parent


def bind():
    j = Journal(OUT / "journal", "ptq-foundation-focus")
    if j.sequence:
        return j, next(ref for ref in j.cards if j.cards[ref]["kind"] == "EvidenceCard")
    m = read_json(OUT / "focus_measurement.json")
    dec = read_json(OUT / "generation.json")[-1]["decision"]["content"]
    p = {
        "level": "L1",
        "question": dec["question"],
        "measurements": ["local MSE by image and arm"],
        "controls": {
            "sample_ids": m["fit"] + m["heldout"],
            "reference": m["reference"],
        },
        "execution": {
            "service": "custom_experiment",
            "resolved_choices": {
                "training": False,
                "calibration_samples": len(m["fit"]),
            },
            "plan": m["plan"],
            "script": asset(OUT / "focus_experiment.py"),
            "decision_sha256": m["decision"],
        },
        "resources": {"estimated_seconds": 60, "cpu_threads": 2},
        "replication": {
            "requested": False,
            "purpose": "integration",
            "configuration_difference": {},
        },
        "scope": {
            "model": m["model"],
            "sites": [m["site"]],
            "protocol": "ptq_extension/1",
        },
    }
    req = make_card("ExperimentCard", j.run_id, p)
    ev = make_card(
        "EvidenceCard",
        j.run_id,
        {
            "experiment_ref": req["id"],
            "execution_fingerprint": fingerprint(p),
            "status": "completed",
            "observations": {
                k: m[k]
                for k in (
                    "training_mse_by_image",
                    "heldout_mse_by_image",
                    "shape",
                    "state_bytes",
                )
            },
            "provenance": {
                "model": m["model"],
                "fit": m["fit"],
                "heldout": m["heldout"],
                "site": m["site"],
                "protocol": m["protocol"],
                "reference": m["reference"],
            },
            "resources": {
                "device": "cuda",
                "fit_images": len(m["fit"]),
                "heldout_images": len(m["heldout"]),
            },
            "artifacts": [
                asset(OUT / name)
                for name in (
                    "focus_measurement.json",
                    "focus_cold.json",
                    "generation.json",
                )
            ],
            "measurement_source": "real",
            "execution_id": "pretrained-local-compensation",
        },
        refs=[req["id"]],
    )
    j.commit("record-local-experiment", cards=[req, ev])
    return j, ev["id"]


if __name__ == "__main__":
    j, ev = bind()
    memory = ResearchMemory(j)
    if not memory.snapshot()["L3"]:
        raw = read_json(OUT / "recorder.json")[-1]["decision"]["content"]
        batch = memory.recording_batch()
        payload = memory.prepare_metareview(raw, batch, 1)
        card = make_card(
            "MetareviewCard",
            j.run_id,
            payload,
            author="recorder",
            refs=[*batch["source_refs"], *payload["evidence_refs"]],
        )
        projection = memory.apply_metareview(card)
        j.commit(
            "recorder/foundation",
            cards=[card],
            events=[
                projection,
                {
                    "type": "recorder_batch_completed",
                    "source_refs": batch["source_refs"],
                },
            ],
        )
    cold = Journal(OUT / "journal", j.run_id)
    state = ResearchMemory(cold).snapshot()
    ref = state["L3"][-1]
    assert (
        ev in cold.card(ref)["refs"]
        and cold.card(ev)["payload"]["measurement_source"] == "real"
    )
    atomic_json(
        OUT / "memory_restore.json",
        {
            "experiment_ref": cold.card(ev)["payload"]["experiment_ref"],
            "evidence_ref": ev,
            "recorder_ref": ref,
            "content_sha256": cold.card(ref)["content_sha256"],
            "L2": list(state["L2"]),
            "L3": state["L3"],
            "scientific_analysis": cold.card(ref)["payload"]["scientific_analysis"],
        },
    )
