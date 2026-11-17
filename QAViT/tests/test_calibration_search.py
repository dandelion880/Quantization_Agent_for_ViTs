"""Real tiny Transformer forward/selection checks; no accuracy claims."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.paths import QAVIT
import torch
from tools.runtime_interface import interface_fixture
from ptq_agent.execution.pipeline import QuantizationExecutor, quantizers
from ptq_agent.execution.baselines import build_baseline
from ptq_agent.execution.calibration_search import search_frozen_scales
from ptq_agent.execution.model import forward, tensor_digest
from ptq_agent.schemas import PTQError


def fixture():
    torch.set_num_threads(1)
    execution = {"backend": {"config_path": str(QAVIT / "configs/deitb_backend.json")}}
    backend, cfg, data = interface_fixture(execution, {
        "shape": [1, 3, 32, 32], "seed": 7,
        "sites": ["patch_embed.proj", "blocks.0.attn.qkv", "blocks.1.mlp.fc2"],
        "counts": {"calibration": 3, "search": 2, "final": 1}})
    backend.identity = lambda: {"fixture": "calibration_search", "state": backend.original_state}
    plan = build_baseline(backend, "adalog")
    plan["preprocess"] = []
    for layer in plan["layers"]:
        layer["options"] = {}
        for target in ("weight", "activation"):
            if layer[target]:
                layer[target + "_method"] = "uniform_affine"
                layer[target + "_calibration"] = {"strategy": "minmax"}
    return backend, cfg, data, plan


class CalibrationSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend, cls.cfg, cls.data, cls.plan = fixture()
        cls.executor = QuantizationExecutor(cls.backend, cls.cfg, cls.data)
        cls.model, cls.weights, cls.activations, cls.audit = cls.executor.prepare(cls.plan, {})

    def setUp(self):
        self.model_copy = deepcopy(self.model)
        self.candidates = [{"id": str(f), "scales": {"blocks.0.attn.qkv": {"activation": f}}} for f in (.5, 1., 1.5)]

    def test_block_output_selects_actual_candidate_and_records_scores(self):
        for reference in ("fp_model", "local_fp"):
            model = deepcopy(self.model)
            report = search_frozen_scales(self.backend, model, self.data, self.candidates,
                                           block="blocks.0", reference=reference, token_scope="patch")
            self.assertEqual(report["selected_score"], min(row["score"] for row in report["candidates"]))
            self.assertGreater(len({row["score"] for row in report["candidates"]}), 1)
            factor = report["selected_scales"]["blocks.0.attn.qkv"]["activation"]
            expected = quantizers(self.model)["blocks.0.attn.qkv"].params()["scale"] * factor
            self.assertTrue(torch.equal(quantizers(model)["blocks.0.attn.qkv"].params()["scale"], expected))
            self.assertEqual(report["sample_ids"], self.data.ids("calibration"))
            self.assertEqual(set(report["baseline_active_targets"]), set(quantizers(self.model)))
            self.assertTrue(report["baseline_configuration_sha256"])

    def test_state_hashes_compare_the_same_full_frozen_state(self):
        candidate = [{"id": "identity", "scales": {"blocks.0.attn.qkv": {"activation": 1.0}}}]
        x, _ = next(self.data.batches("search"))
        before = forward(self.backend, self.model_copy, x).clone()
        report = search_frozen_scales(self.backend, self.model_copy, self.data, candidate)
        self.assertTrue(torch.equal(before, forward(self.backend, self.model_copy, x)))
        self.assertEqual(report["baseline_state_sha256"], report["selected_state_sha256"])
        candidate[0]["scales"]["blocks.0.attn.qkv"]["activation"] = .8
        changed = search_frozen_scales(self.backend, self.model_copy, self.data, candidate)
        self.assertNotEqual(changed["baseline_state_sha256"], changed["selected_state_sha256"])

    def test_network_metrics_consume_logits_and_true_labels(self):
        labels = {sid: int(forward(self.backend, self.backend.model, x).argmax(-1).item()) for x, sid in self.data.batches("calibration")}
        # Explicit fixture labels for objective mechanics, not ImageNet accuracy.
        self.data.labels.update(labels)
        scores = {}
        for metric in ("mse", "cosine", "teacher_kl", "task_loss", "top1_error", "hessian_diag_ce"):
            report = search_frozen_scales(self.backend, deepcopy(self.model), self.data, self.candidates, metric=metric)
            scores[metric] = [r["score"] for r in report["candidates"]]
            self.assertEqual(report["selected_score"], min(scores[metric]))
            self.assertTrue(all(torch.isfinite(torch.tensor(scores[metric]))))
        self.assertNotEqual(scores["mse"], scores["teacher_kl"])
        self.assertGreater(len(set(scores["hessian_diag_ce"])), 1)

    def test_float_context_changes_measured_scores_and_restores_full_background(self):
        reports = [search_frozen_scales(self.backend, deepcopy(self.model), self.data, self.candidates,
                    block="blocks.0", context=context) for context in ("float", "quantized")]
        self.assertNotEqual([r["score"] for r in reports[0]["candidates"]], [r["score"] for r in reports[1]["candidates"]])
        search_frozen_scales(self.backend, self.model_copy, self.data, self.candidates, context="float")
        self.assertTrue(all(q.enabled for q in quantizers(self.model_copy).values()))

    def test_failure_restores_state_and_removes_hooks(self):
        before = {s: tensor_digest(q.params()["scale"]) for s, q in quantizers(self.model_copy).items()}
        self.candidates[-1]["scales"]["blocks.0.attn.qkv"]["activation"] = -1
        with self.assertRaises(PTQError):
            search_frozen_scales(self.backend, self.model_copy, self.data, self.candidates, block="blocks.0")
        self.assertEqual(before, {s: tensor_digest(q.params()["scale"]) for s, q in quantizers(self.model_copy).items()})
        self.assertFalse(self.model_copy.blocks[0]._forward_hooks)
        self.assertFalse(self.backend.model.blocks[0]._forward_hooks)

    def test_isolated_snapshot_classes_support_frozen_search(self):
        import importlib.util
        name = "ptq_agent.execution._isolated_calibration_pipeline"
        spec = importlib.util.spec_from_file_location(name, QAVIT / "tools/ptq_agent/execution/pipeline.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            # Same numerical interface, intentionally distinct class identity,
            # as used by qavit_history_<hash> source-isolated frozen loading.
            isolated = module.build_model(self.backend, self.plan, self.executor.preprocess_state,
                                           self.weights, self.activations)
            for q in module.quantizers(isolated).values():
                q.enabled = q.layer["activation"] is not None
                if hasattr(q, "weight_enabled"):
                    q.weight_enabled = q.layer["weight"] is not None
            self.assertEqual(quantizers(isolated), {})
            report = search_frozen_scales(self.backend, isolated, self.data, self.candidates, metric="teacher_kl")
            self.assertTrue(report["parameters_installed"])
            self.assertEqual(len(report["candidates"]), 3)
        finally:
            sys.modules.pop(name, None)

    def test_weight_selection_and_selected_package_reload(self):
        candidates = [{"id": str(f), "scales": {"blocks.0.attn.qkv": {"weight": f}}} for f in (.8, 1., 1.2)]
        model = self.model_copy
        report = search_frozen_scales(self.backend, model, self.data, candidates, metric="teacher_kl")
        qmap = quantizers(model)
        weights = {q.layer["module"]: q.weight_record for q in qmap.values() if getattr(q, "weight_record", None) is not None}
        params = {s: q.params() for s, q in qmap.items() if q.layer["activation"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected"
            self.executor.report["custom_selection"] = report
            self.executor.export(model, self.plan, weights, params, path, self.executor.audit(model, self.plan))
            loaded, manifest = QuantizationExecutor(self.backend, self.cfg, self.data).load(path)
            self.assertEqual(manifest["calibration_report"]["custom_selection"], report)
            x, _ = next(self.data.batches("search"))
            self.assertTrue(torch.equal(forward(self.backend, model, x), forward(self.backend, loaded, x)))
        self.assertTrue(report["parameters_installed"])

    def test_defaults_share_numerical_identity_and_unused_prep_fields_rejected(self):
        from ptq_agent.execution.capabilities import plan_key
        explicit = deepcopy(self.plan)
        explicit["execution"].update(calibration_unit="operator", layer_order="forward",
                                      reestimate_passes=1, calibration_context="current")
        self.assertEqual(plan_key(self.plan), plan_key(explicit))
        bad = build_baseline(self.backend, "repq")
        bad["preprocess"][0]["calibration"] = {"strategy": "minmax", "percentiles": [.99]}
        with self.assertRaisesRegex(PTQError, "未消费字段"):
            QuantizationExecutor(self.backend, self.cfg, self.data).prepare(bad, {})

    def test_actual_block_schedule_and_reestimation_change_parameters(self):
        # Retain two linear operators per actual Transformer block to inspect
        # W/A interleaving; this is a diagnostic plan, not a complete candidate.
        plan = deepcopy(self.plan)
        plan["layers"] = [l for l in plan["layers"] if l["module"] in (
            "blocks.0.attn.qkv", "blocks.0.mlp.fc2", "blocks.1.attn.qkv", "blocks.1.mlp.fc2")]
        plan["execution"].update(order="w_then_a", calibration_unit="block", block_modules=["blocks.0", "blocks.1"],
                                  layer_order="reverse", reestimate_passes=2, calibration_context="current")
        executor = QuantizationExecutor(self.backend, self.cfg, self.data, diagnostic=True)
        _, _, _, _ = executor.prepare(plan, {})
        rows = executor.report["layers"] if "layers" in executor.report else executor.report["calibration"]
        self.assertEqual([r["pass"] for r in rows], [1] * 8 + [2] * 8)
        self.assertTrue(all(r["block"] == "blocks.1" for r in rows[:4]))
        self.assertEqual([r["target"] for r in rows[:4]], ["weight", "weight", "activation", "activation"])
        first, second = rows[2], rows[10]
        self.assertNotEqual(first["active_quantization_targets"], second["active_quantization_targets"])
        self.assertNotEqual(first["selected_parameters"], second["selected_parameters"])
        self.assertNotIn(second["site"], second["active_quantization_targets"]["activation"])
        fp_plan = deepcopy(plan)
        fp_plan["execution"].update(calibration_context="float", reestimate_passes=1)
        fp_executor = QuantizationExecutor(self.backend, self.cfg, self.data, diagnostic=True)
        fp_executor.prepare(fp_plan, {})
        fp_rows = fp_executor.report.get("layers", fp_executor.report.get("calibration"))
        self.assertTrue(all(not any(r.get("active_quantization_targets", {}).values()) for r in fp_rows))


if __name__ == "__main__":
    unittest.main()
