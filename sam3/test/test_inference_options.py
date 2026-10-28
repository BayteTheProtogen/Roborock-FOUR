"""Numerical and selection tests for image inference optimizations."""

import unittest
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.vitdet import Mlp, _chunked_attention
from scripts.inference_options import DEFAULTS, apply_environment, configure_model, exporter_flags
from scripts.inference_suite import baseline, combined_configs, repeat_summary, screening_configs


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.image_size = 28
        self.inst_interactive_predictor = None
        self.backbone = SimpleNamespace(forward_image=lambda image: {"image": image},
                                        forward_text=lambda prompts, device: {"language_features": torch.ones(1, device=device)})
        self.nan = False

    def _get_dummy_prompt(self):
        return None

    def forward_grounding(self, **kwargs):
        masks = torch.tensor([[[[2., -2.], [2., -2.]], [[-2., 2.], [-2., 2.]]]])
        if self.nan:
            masks[0, 0, 0, 0] = float("nan")
        return {"pred_masks": masks, "pred_boxes": torch.ones(1, 2, 4) * .5,
                "pred_logits": torch.ones(1, 2, 1) * 4, "presence_logit_dec": torch.ones(1, 1) * 4}


class InferenceOptionsTests(unittest.TestCase):
    def test_mlp_chunks_preserve_token_values_and_shape(self):
        torch.manual_seed(9)
        mlp = Mlp(16, 48, 8, norm_layer=torch.nn.LayerNorm).eval()
        x = torch.randn(2, 23, 19, 16)
        expected = mlp(x)
        for chunk in (64, 256, 512, 2048):
            torch.testing.assert_close(mlp._forward_chunked(x, chunk), expected, rtol=1e-5, atol=1e-6)

    def test_disabled_mps_flags_are_absent_not_zero(self):
        with patch.dict(os.environ, {"PYTORCH_MPS_PREFER_METAL": "1", "PYTORCH_MPS_FAST_MATH": "1"}):
            apply_environment(SimpleNamespace(prefer_metal=False, fast_math=False))
            self.assertNotIn("PYTORCH_MPS_PREFER_METAL", os.environ)
            self.assertNotIn("PYTORCH_MPS_FAST_MATH", os.environ)
            apply_environment(SimpleNamespace(prefer_metal=True, fast_math=True))
            self.assertEqual(os.environ["PYTORCH_MPS_PREFER_METAL"], "1")

    def test_chunked_attention_matches_sdpa_across_chunk_sizes(self):
        torch.manual_seed(7)
        q, k, v = [torch.randn(2, 3, 301, 16) for _ in range(3)]
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        for chunk in (64, 128, 256, 512, 1024):
            with self.subTest(chunk=chunk):
                torch.testing.assert_close(_chunked_attention(q, k, v, chunk), expected, rtol=1e-5, atol=1e-6)

    def test_cpu_preprocessing_and_mask_postprocessing_preserve_output(self):
        model = DummyModel()
        image = Image.new("RGB", (40, 60), color=(70, 100, 120))
        base = Sam3Processor(model)
        cpu = Sam3Processor(model, preprocess_on_cpu=True, postprocess_on_cpu=True)
        a = base.set_image(image)
        b = cpu.set_image(image)
        torch.testing.assert_close(a["backbone_out"]["image"], b["backbone_out"]["image"])
        a = base.set_text_prompt("sidewalk", a)
        b = cpu.set_text_prompt("sidewalk", b)
        self.assertEqual(a["masks"].shape, (2, 1, 60, 40))
        self.assertTrue(torch.equal(a["masks"], b["masks"]))

    def test_nonfinite_outputs_are_rejected_before_filtering(self):
        model = DummyModel()
        model.nan = True
        processor = Sam3Processor(model, check_finite=True)
        state = processor.set_image(Image.new("RGB", (40, 60)))
        with self.assertRaisesRegex(RuntimeError, "non-finite model output"):
            processor.set_text_prompt("sidewalk", state)

    def test_half_weights_retain_full_precision_norm_and_decoder_ffn(self):
        model = torch.nn.Module()
        model.trunk = torch.nn.Linear(4, 4)
        model.norm = torch.nn.LayerNorm(4)
        model.ffn = torch.nn.Module()
        model.ffn.linear1 = torch.nn.Linear(4, 4)
        model.ffn.linear2 = torch.nn.Linear(4, 4)
        configure_model(model, SimpleNamespace(precision="fp16"))
        self.assertEqual(model.trunk.weight.dtype, torch.float16)
        self.assertEqual(model.norm.weight.dtype, torch.float32)
        self.assertEqual(model.ffn.linear1.weight.dtype, torch.float32)

    def test_export_flags_represent_the_selected_configuration(self):
        cfg = {**baseline(), "precision": "fp16", "preprocess": "cpu", "prefer_metal": True}
        flags = exporter_flags(cfg, .5, "mps")
        self.assertIn("--precision", flags)
        self.assertIn("fp16", flags)
        self.assertIn("--preprocess", flags)
        self.assertIn("--prefer-metal", flags)
        self.assertNotIn("--fast-math", flags)

    def test_sweep_ids_unique_and_baseline_unchanged(self):
        configs = screening_configs([1008, 672, 504, 336])
        self.assertEqual(configs[0], baseline())
        self.assertEqual(len(configs), len({cfg["id"] for cfg in configs}))
        self.assertTrue(any(cfg["precision"] == "fp16" for cfg in configs))
        self.assertTrue(any(cfg["size"] == 952 for cfg in configs))

    def test_combinations_exclude_failed_quality_changes(self):
        bad = {**baseline(), "id": "bad", "precision": "amp-bf16"}
        good = {**baseline(), "id": "cpu", "preprocess": "cpu"}
        rows = [{"configuration": baseline(), "eligible": True, "mean_seconds": 4},
                {"configuration": bad, "eligible": False, "mean_seconds": 1},
                {"configuration": good, "eligible": True, "mean_seconds": 3}]
        configs = combined_configs(rows)
        self.assertTrue(all(cfg["precision"] != "amp-bf16" for cfg in configs))
        self.assertTrue(any(cfg["preprocess"] == "cpu" for cfg in configs))

    def test_failed_repeat_invalidates_candidate(self):
        good = {"config_id": "1008", "configuration": baseline(), "eligible": True, "mean_seconds": 4,
                "mean_iou": 1, "worst_frame_iou": 1, "mps_driver_peak_gib": 7}
        failed = {**good, "eligible": False, "error": "OOM"}
        rows = repeat_summary([{"results": [good]}, {"results": [failed]}])
        self.assertFalse(rows[0]["eligible_every_repeat"])


if __name__ == "__main__":
    unittest.main()
