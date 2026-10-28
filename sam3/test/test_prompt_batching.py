"""Exercise native prompt batching and compare labels with sequential inference."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from scripts.export_video_overlay import (
    _infer_image_batch, export_image_dataset, parse_args, prompt_batch_size,
)


class PromptModel(torch.nn.Module):
    """Deterministic outputs that depend on prompt IDs, with empty classes too."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.image_size = 336
        self.inst_interactive_predictor = None
        self.image_calls = 0
        self.text_calls = []
        self.grounding_calls = []
        self.nan = False
        self.backbone = SimpleNamespace(
            forward_image=self.encode_image,
            forward_text=self.encode_text,
            language_backbone=torch.nn.Linear(2, 2),
        )

    def encode_image(self, image):
        self.image_calls += 1
        return {"image": image}

    def encode_text(self, prompts, device):
        self.text_calls.append(list(prompts))
        ids = torch.tensor([int(prompt.removeprefix("class")) for prompt in prompts], device=device)
        return {"language_features": ids.float().reshape(1, -1, 1),
                "language_mask": torch.zeros((len(prompts), 1), device=device, dtype=torch.bool)}

    def _get_dummy_prompt(self, num_prompts=1):
        return SimpleNamespace(count=num_prompts)

    def forward_grounding(self, backbone_out, find_input, geometric_prompt, **kwargs):
        ids = backbone_out["language_features"][0, find_input.text_ids, 0].long()
        if find_input.img_ids.tolist() != [0] * len(ids) or geometric_prompt.count != len(ids):
            raise AssertionError("prompt group must refer to the single shared image")
        self.grounding_calls.append(ids.tolist())
        masks = torch.full((len(ids), 2, 4, 4), -2.)
        masks[:, 0, :2, :2] = 2.
        masks[:, 1, 2:, 2:] = 2.
        logits = torch.full((len(ids), 2, 1), 4.)
        logits[ids == 1] = -8.  # a class with no detections must not shift IDs
        logits[ids == 2, 1] = -8.
        if self.nan:
            masks[-1, 0, 0, 0] = float("nan")
        return {"pred_masks": masks, "pred_boxes": torch.full((len(ids), 2, 4), .5),
                "pred_logits": logits, "presence_logit_dec": torch.full((len(ids), 1), 4.)}


class PromptBatchingTests(unittest.TestCase):
    def test_percentages_round_up_and_none_stays_one(self):
        for count, expected in (
            (8, [8, 4, 2, 1]), (5, [5, 3, 2, 1]), (3, [3, 2, 1, 1]), (1, [1, 1, 1, 1]),
        ):
            actual = [prompt_batch_size(count, level) for level in ("high", "medium", "low", "none")]
            self.assertEqual(actual, expected)
        with self.assertRaises(ValueError):
            prompt_batch_size(0, "high")

    def test_native_batch_matches_sequential_results_and_preserves_image_state(self):
        for cpu_postprocess in (False, True):
            model = PromptModel()
            processor = Sam3Processor(model, postprocess_on_cpu=cpu_postprocess)
            state = processor.set_image(Image.new("RGB", (20, 12)))
            backbone = state["backbone_out"]
            prompts = [f"class{i}" for i in range(5)]
            expected = []
            for prompt in prompts:
                result = processor.set_text_prompt(prompt, state)
                expected.append({key: result[key].clone() for key in ("masks", "scores", "boxes")})
                processor.reset_all_prompts(state)
            model.grounding_calls.clear()
            features = processor.encode_text_prompts(prompts)
            # Cached text encoding must be sufficient after discarding the encoder.
            with patch.object(model.backbone, "forward_text", side_effect=AssertionError("must use cache")):
                iterator = processor.set_text_prompts(prompts, state, text_features=features)
                self.assertEqual(model.grounding_calls, [[0, 1, 2, 3, 4]])
                actual = list(iterator)
            self.assertEqual(model.image_calls, 1)
            self.assertIs(state["backbone_out"], backbone)
            self.assertNotIn("language_features", state["backbone_out"])
            self.assertNotIn("masks", state)
            self.assertEqual(processor.find_stage.text_ids.tolist(), [0])
            self.assertEqual([len(row["scores"]) for row in actual], [2, 0, 1, 2, 2])
            for single, batched in zip(expected, actual):
                for key in single:
                    torch.testing.assert_close(single[key], batched[key])

    def test_batch_rejects_bad_inputs_and_nonfinite_model_output(self):
        model = PromptModel()
        processor = Sam3Processor(model, check_finite=True)
        with self.assertRaises(ValueError):
            processor.encode_text_prompts([])
        with self.assertRaises(ValueError):
            processor.set_text_prompts(["class0"], {})
        state = processor.set_image(Image.new("RGB", (20, 12)))
        with self.assertRaises(ValueError):
            processor.set_text_prompts([], state)
        features = processor.encode_text_prompts(["class0"])
        with self.assertRaisesRegex(ValueError, "group size"):
            processor.set_text_prompts(["class0", "class1"], state, features)
        model.nan = True
        with self.assertRaisesRegex(RuntimeError, "non-finite model output"):
            list(processor.set_text_prompts(["class0", "class1"], state))

    def test_all_levels_use_expected_grounding_groups_and_keep_exported_labels(self):
        from sam3 import model_builder

        class Worker:
            def __init__(self, target, args):
                self.args = args
                self.exitcode = None

            def start(self):
                _infer_image_batch(*self.args)
                self.exitcode = 0

            def join(self):
                pass

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input"
            source.mkdir()
            for index in range(2):
                cv2.imwrite(str(source / f"{index}.jpg"), np.full((12, 20, 3), index, dtype=np.uint8))
            checkpoint = root / "checkpoint.pt"
            checkpoint.touch()
            baseline_rows, baseline_labels = None, None
            for level in ("none", "low", "medium", "high"):
                model = PromptModel()
                output = root / level
                flags = [flag for index in range(5) for flag in ("--text", f"class{index}")]
                args = parse_args([str(source), str(output), "--image-mode", "--device", "cpu",
                                   "--checkpoint", str(checkpoint), "--export-format", "yolo",
                                   "--inference-size", "336", "--no-overlays", "--batching", level, *flags])
                with patch.object(model_builder, "build_sam3_image_model", return_value=model), \
                     patch("scripts.export_video_overlay.mp.get_context", return_value=SimpleNamespace(Process=Worker)):
                    export_image_dataset(args)
                size = prompt_batch_size(5, level)
                groups = [list(range(start, min(start + size, 5))) for start in range(0, 5, size)]
                self.assertEqual(model.grounding_calls, groups * 2)
                self.assertEqual(model.image_calls, 2)
                self.assertEqual(model.text_calls, [[f"class{i}" for i in group] for group in groups])
                config = json.loads((output / "dataset.json").read_text())
                self.assertEqual(config["batching"], level)
                self.assertEqual(config["prompt_batch_size"], size)
                rows = [json.loads(line) for line in (output / "metadata.jsonl").read_text().splitlines()]
                with zipfile.ZipFile(output / "training" / "dataset.zip") as archive:
                    labels = {name: archive.read(name) for name in archive.namelist() if name.endswith(".txt")}
                if baseline_rows is None:
                    baseline_rows, baseline_labels = rows, labels
                else:
                    self.assertEqual(rows, baseline_rows)
                    self.assertEqual(labels, baseline_labels)

    def test_cli_default_and_image_mode_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input"
            source.mkdir()
            cv2.imwrite(str(source / "0.jpg"), np.zeros((2, 2, 3), dtype=np.uint8))
            base = [str(source), str(Path(folder) / "out"), "--text", "class0"]
            self.assertEqual(parse_args(base + ["--image-mode"]).batching, "none")
            with self.assertRaises(SystemExit):
                parse_args(base + ["--batching", "high"])
            with self.assertRaises(SystemExit):
                parse_args(base + ["--image-mode", "--batching", "invalid"])


if __name__ == "__main__":
    unittest.main()
