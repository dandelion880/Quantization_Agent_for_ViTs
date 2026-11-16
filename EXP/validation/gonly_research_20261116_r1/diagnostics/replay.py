"""Small real PyTorch component replay; seeded synthetic images, not task evidence."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "QAViT"))
from runtime.paths import QAVIT
import torch
from tests.test_frozen_propagation import tiny_frozen
from ptq_agent.execution.propagation import profile_frozen_propagation


def main():
    torch.set_num_threads(1)
    backend, cfg, data, model, manifest = tiny_frozen()
    result = profile_frozen_propagation(backend, model, data,
        blocks=["blocks.0", "blocks.1"], split="search",
        token_layout={"axis": 1, "cls_index": 0, "patch_start": 1},
        interventions=[{"id": "block0_fp", "kind": "fp_block", "block": "blocks.0"},
                       {"id": "block1_fp", "kind": "fp_block", "block": "blocks.1"},
                       {"id": "q0_bypass", "kind": "bypass", "site": "blocks.0.attn::attention_query",
                        "target": "activation"}],
        source={"fixture": "seed23 tiny ViT; exported and independently reloaded AdaLog static W4A4",
                "measurement_source": "synthetic_computation", "formal_evaluation": False})
    path = Path(__file__).with_name("numerical_result.json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    summary = {"measurement_source": "synthetic_computation", "sample_count": result["sample_count"],
               "formal_evaluation": False,
               "boundaries": {name: {metric: row[metric]["mse"]
                           for metric in ("input_total", "local_added", "upstream_mapped", "output_total")}
                              for name, row in result["boundaries"].items()},
               "baseline_logit_mse": result["baseline_output_error"]["mse"],
               "intervention_logit_mse_reduction": {name: row["mse_reduction_to_fp"]
                                      for name, row in result["interventions"].items()},
               "state_unchanged": result["state_unchanged"]}
    Path(__file__).with_name("summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
