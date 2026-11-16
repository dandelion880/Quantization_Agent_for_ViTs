"""Actual frozen tiny-ViT replay, algebraic decomposition, and intervention cleanup."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.paths import QAVIT
import torch
from ptq_agent.execution.baselines import build_baseline
from ptq_agent.execution.model import RTNExecutor
from ptq_agent.execution.pipeline import quantizers
from ptq_agent.execution.propagation import profile_frozen_propagation
from ptq_agent.schemas import PTQError
from tools.runtime_interface import interface_fixture, execute_custom
from tests import test_infrastructure as infra


def tiny_frozen():
    backend, config, data = interface_fixture({"backend": {"config_path": "QAViT/configs/deitb_backend.json"}}, {
        "shape": [1, 3, 32, 32], "sites": ["patch_embed.proj", "blocks.0.attn.qkv", "blocks.1.mlp.fc2"],
        "seed": 23, "counts": {"calibration": 2, "search": 2, "final": 1}})
    backend.identity = lambda: {"fixture": "propagation_actual_tiny_vit", "state": backend.original_state}
    plan = build_baseline(backend, "adalog")
    for row in plan["layers"]:
        if row.get("activation_method") == "adalog":
            row["options"]["log_base_candidates"] = [19]
            row["activation_calibration"]["factors"] = [1.0]
    executor = RTNExecutor(backend, config, data)
    model, weights, activations, audit = executor.prepare(plan, {})
    with tempfile.TemporaryDirectory(prefix="propagation-package-") as tmp:
        directory = Path(tmp) / "frozen"
        executor.export(model, plan, weights, activations, directory, audit)
        loaded, manifest = executor.load(directory)
    return backend, config, data, loaded, manifest


class FrozenPropagation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.backend, cls.config, cls.data, cls.model, cls.manifest = tiny_frozen()

    def test_actual_frozen_blocks_decompose_and_interventions_reach_downstream(self):
        sid = next(s for s, q in quantizers(self.model).items()
                   if s.startswith("blocks.0") and q.enabled and q.layer["activation"])
        result = profile_frozen_propagation(self.backend, self.model, self.data,
            blocks=["blocks.0", "blocks.1"], split="search",
            token_layout={"axis": 1, "cls_index": 0, "patch_start": 1},
            interventions=[{"id": "first_block_fp", "kind": "fp_block", "block": "blocks.0"},
                           {"id": "first_a_bypass", "kind": "bypass", "site": sid, "target": "activation"}])
        self.assertEqual(result["sample_ids"], self.data.ids("search"))
        self.assertTrue(result["state_unchanged"])
        for block, row in result["boundaries"].items():
            for group in ("all", "cls", "patch"):
                total = row["output_total"]["outputs"]["output." + group]["mse"]
                local = row["local_added"]["outputs"]["output." + group]["mse"]
                upstream = row["upstream_mapped"]["outputs"]["output." + group]["mse"]
                cross = result["decomposition_cross_term"][block][group]
                self.assertAlmostEqual(total, local + upstream + cross["twice_dot"] / cross["elements"], places=10)
        self.assertGreater(result["boundaries"]["blocks.1"]["upstream_mapped"]["mse"], 0)
        self.assertGreater(result["interventions"]["first_block_fp"]["boundaries"]["blocks.1"]
                           ["change_from_frozen"]["mse"], 0)
        self.assertTrue(quantizers(self.model)[sid].enabled)
        # A negative intervention result is valid data, never filtered as a failure.
        self.assertIn("output", result["interventions"]["first_a_bypass"]["mse_reduction_to_fp"])

    def test_zero_error_fp_copy_and_named_task_representations(self):
        model = deepcopy(self.backend.model)
        def representations(m, x):
            logits = m(x)
            return {"classification_logits": logits, "task_feature": logits[:, :12]}
        result = profile_frozen_propagation(self.backend, model, self.data,
            blocks=["blocks.0"], forward_fn=representations)
        self.assertEqual(result["baseline_output_error"]["mse"], 0)
        self.assertEqual(set(result["baseline_output_error"]["outputs"]),
                         {"output.classification_logits", "output.task_feature"})
        self.assertEqual(result["classification_diagnostic"], {})

    def test_exception_restores_bypass_and_hooks(self):
        sid = next(s for s, q in quantizers(self.model).items()
                   if s.startswith("blocks.0") and q.enabled and q.layer["activation"])
        q = quantizers(self.model)[sid]
        def failing(m, x):
            if m is self.model and not q.enabled:
                raise RuntimeError("injected intervention failure")
            return m(x)
        counts = {name: len(m._forward_hooks) for name, m in self.model.named_modules()}
        with self.assertRaisesRegex(RuntimeError, "injected"):
            profile_frozen_propagation(self.backend, self.model, self.data,
                blocks=["blocks.0"], forward_fn=failing,
                interventions=[{"id": "break", "kind": "bypass", "site": sid, "target": "activation"}])
        self.assertTrue(q.enabled)
        self.assertEqual(counts, {name: len(m._forward_hooks) for name, m in self.model.named_modules()})

    def test_final_split_and_training_are_rejected(self):
        with self.assertRaises(PTQError):
            profile_frozen_propagation(self.backend, self.model, self.data, blocks=["blocks.0"], split="final")
        self.model.train()
        try:
            with self.assertRaises(PTQError):
                profile_frozen_propagation(self.backend, self.model, self.data, blocks=["blocks.0"])
        finally:
            self.model.eval()

    def test_custom_data_observer_records_actual_consumption(self):
        model = self.model
        class Plugin:
            @staticmethod
            def execute(request, context):
                return {"observations": {"propagation": profile_frozen_propagation(
                    context["backend"], model, context["data"], blocks=["blocks.0", "blocks.1"])}}
        payload = {"execution": {"resolved_choices": {}, "parameters": {},
                                  "calibration": {"split": "calibration", "count": 2}}}
        with tempfile.TemporaryDirectory(prefix="propagation-context-") as tmp:
            result = execute_custom(Plugin, payload, self.backend, self.config, self.data, Path(tmp))
        self.assertEqual(result["execution_facts"]["data"]["calibration"]["sample_count"], 2)
        self.assertEqual(result["execution_facts"]["forward_calls"], 4)


class PropagationSkillAccess(infra.TempCase):
    def test_generation_discovers_then_reads_the_actual_api_reference(self):
        wf = self.workflow(1)
        tid = wf.enqueue_role("propagation/read", "generate", refs=[wf.brief["id"]],
                              context={"round": 1, "slot": "H2", "hypothesis_type": "engineering"})
        context = wf.context(wf.queue.get(tid))
        self.assertIn("diagnose-ptq-failure/SKILL.md", context["skill_catalog"])
        for path in ("skills/diagnose-ptq-failure/SKILL.md",
                     "skills/diagnose-ptq-failure/references/frozen-propagation.md"):
            result = wf.host_tool({"inputs": {"parent_task": tid, "name": "read_asset",
                                  "arguments": {"asset": path, "limit": 12000}}})
            self.assertIn("profile_frozen_propagation", result["text"])


if __name__ == "__main__":
    unittest.main()
