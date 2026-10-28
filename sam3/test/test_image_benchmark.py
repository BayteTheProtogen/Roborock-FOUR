"""Check quality aggregation and recommendation safeguards without loading SAM3."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from scripts.benchmark_image_inference import (
    extract_samples,
    mask_counts,
    read_mask,
    recommend,
    summarize_quality,
    system_swap_used,
)


class ImageBenchmarkTests(unittest.TestCase):
    def test_unavailable_swap_counter_is_optional(self):
        with patch("psutil.swap_memory", side_effect=OSError):
            self.assertIsNone(system_swap_used())

    def test_empty_matches_do_not_hide_missing_foreground(self):
        empty = np.zeros((2, 2), dtype=bool)
        foreground = empty.copy()
        foreground[0, 0] = True
        rows = [mask_counts(empty, empty)] * 100
        rows.append(mask_counts(empty, foreground))
        quality = summarize_quality(rows)
        self.assertEqual(quality["mean_iou"], 0)
        self.assertEqual(quality["aggregate_iou"], 0)
        self.assertEqual(quality["both_empty_frames"], 100)

    def test_overlap_and_aggregate_weighting(self):
        reference = np.array([[1, 1], [0, 0]], dtype=bool)
        prediction = np.array([[1, 0], [1, 0]], dtype=bool)
        row = mask_counts(prediction, reference)
        self.assertEqual(row["intersection"], 1)
        self.assertEqual(row["union"], 3)
        quality = summarize_quality([row, mask_counts(reference, reference)])
        self.assertAlmostEqual(quality["aggregate_iou"], 3 / 5)
        self.assertAlmostEqual(quality["foreground_recall"], 3 / 4)

    def result(self, size, seconds, iou=1, worst=1, peak=5):
        return dict(size=size, mean_seconds=seconds, mean_iou=iou,
                    aggregate_iou=iou, worst_frame_iou=worst,
                    reference_pixels=10, mps_driver_peak_gib=peak, error=None)

    def test_choose_fastest_within_quality_and_memory_limits(self):
        rows = [self.result(1008, 4, peak=8), self.result(672, 2, iou=.97, worst=.9),
                self.result(504, 1, iou=.8), self.result(336, .5, iou=.97, worst=.4)]
        self.assertEqual(recommend(rows, .05, .8, 6)["size"], 672)
        self.assertFalse(rows[0]["eligible"])
        self.assertFalse(rows[2]["eligible"])
        self.assertFalse(rows[3]["eligible"])

    def test_all_empty_and_failed_results_cannot_win(self):
        empty = self.result(672, 1)
        empty.update(reference_pixels=0, mean_iou=None, aggregate_iou=None, worst_frame_iou=None)
        self.assertIsNone(recommend([empty, {"size": 504, "error": "OOM"}], .05, .8))

    def test_bad_shapes_and_nonbinary_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mask.png"
            cv2.imwrite(str(path), np.array([[0, 255]], dtype=np.uint8))
            with self.assertRaises(ValueError):
                read_mask(path, (2, 2))
            cv2.imwrite(str(path), np.array([[0, 2]], dtype=np.uint8))
            with self.assertRaises(ValueError):
                read_mask(path)

    def test_sampling_spans_clip_and_preserves_label_names(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input"
            source.mkdir()
            for index in range(10):
                cv2.imwrite(str(source / f"photo{index}.jpg"), np.full((8, 12, 3), index, dtype=np.uint8))
            args = SimpleNamespace(input=source, output=Path(folder) / "out", samples=3, warmup=1, sample_every=0)
            samples = extract_samples(args)
            self.assertEqual([s["frame_index"] for s in samples], [0, 4, 9])
            self.assertEqual([s["reference_name"] for s in samples], ["photo0.png", "photo4.png", "photo9.png"])


if __name__ == "__main__":
    unittest.main()
