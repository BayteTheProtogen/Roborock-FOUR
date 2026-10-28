"""Real video decoding and dataset sampling, without SAM weights."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from scripts.export_video_overlay import extract_image_samples, image_dataset_sources, parse_args


class VideoFolderInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "videos"
        self.source.mkdir()

    def args(self, **changes):
        values = dict(input=self.source, output=self.root / "dataset", sample_every=2,
                      max_samples=0, no_overlays=True, large_image_action="ask")
        values.update(changes)
        return SimpleNamespace(**values)

    def video(self, name, values, fps=10):
        path = self.source / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (24, 16))
        if not writer.isOpened():
            self.skipTest("OpenCV MJPEG encoder is unavailable")
        try:
            for value in values:
                writer.write(np.full((16, 24, 3), value, dtype=np.uint8))
        finally:
            writer.release()
        return path

    def test_folder_sampling_restarts_and_filenames_never_collide(self):
        b = self.video("b.AVI", [100, 110, 120, 130], fps=20)
        a = self.video("a.avi", [10, 20, 30], fps=10)
        (self.source / "notes.txt").write_text("ignore")
        (self.source / "nested").mkdir()
        (self.source / "nested" / "ignored.mp4").touch()
        args = self.args()
        samples, sources = extract_image_samples(args)
        self.assertEqual([sample["frame_index"] for sample in samples], [0, 2, 3, 5])
        self.assertEqual([sample["source_frame_index"] for sample in samples], [0, 2, 0, 2])
        self.assertEqual([sample["time_seconds"] for sample in samples], [0, .2, 0, .1])
        self.assertEqual([sample["source"] for sample in samples], [str(a.resolve())] * 2 + [str(b.resolve())] * 2)
        self.assertEqual([source["fps"] for source in sources], [10, 20])
        names = sorted(path.name for path in (args.output / "images").iterdir())
        self.assertEqual(names, ["000000.jpg", "000002.jpg", "000003.jpg", "000005.jpg"])
        for name, value in zip(names, [10, 30, 100, 120]):
            image = cv2.imread(str(args.output / "images" / name))
            self.assertAlmostEqual(float(image.mean()), value, delta=3)

    def test_max_samples_is_global_and_does_not_open_later_clips(self):
        self.video("a.avi", [10, 20, 30])
        self.video("b.avi", [100, 110, 120])
        (self.source / "c.avi").write_bytes(b"invalid video; must not be opened")
        samples, sources = extract_image_samples(self.args(max_samples=3))
        self.assertEqual(len(samples), 3)
        self.assertEqual(len(sources), 2)
        self.assertEqual(samples[-1]["source_frame_index"], 0)

    def test_single_video_retains_frame_numbers_and_timestamps(self):
        path = self.video("single.avi", [10, 20, 30], fps=10)
        samples, sources = extract_image_samples(self.args(input=path))
        self.assertEqual([sample["frame_index"] for sample in samples], [0, 2])
        self.assertEqual([sample["time_seconds"] for sample in samples], [0, .2])
        self.assertEqual(len(sources), 1)

    def test_image_folder_keeps_numeric_order_and_has_no_video_timestamp(self):
        for name, value in (("10.jpg", 100), ("2.jpg", 20), ("1.jpg", 10)):
            cv2.imwrite(str(self.source / name), np.full((16, 24, 3), value, dtype=np.uint8))
        samples, sources = extract_image_samples(self.args())
        self.assertEqual([Path(sample["source"]).name for sample in samples], ["1.jpg", "10.jpg"])
        self.assertTrue(all(sample["time_seconds"] is None for sample in samples))
        self.assertIsNone(sources[0]["fps"])

    def test_video_folder_requires_image_mode_and_mixed_folder_is_rejected(self):
        self.video("clip.avi", [10])
        base = [str(self.source), str(self.root / "dataset"), "--text", "person"]
        parsed = parse_args(base + ["--image-mode"])
        self.assertEqual(parsed.input, self.source)
        with self.assertRaises(SystemExit):
            parse_args([str(self.source), str(self.root / "out.mp4"), "--text", "person"])
        cv2.imwrite(str(self.source / "image.jpg"), np.zeros((16, 24, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "mixes images and videos"):
            image_dataset_sources(self.source)

    def test_empty_clip_stops_and_closes_reader_even_after_another_clip(self):
        (self.source / "a.avi").touch()
        (self.source / "b.avi").touch()
        readers = []

        class Reader:
            def __init__(self, path):
                self.paths = None
                self.fps = 10
                self.num_frames = 1
                self.frames = [np.zeros((16, 24, 3), dtype=np.uint8)] if path.name == "a.avi" else []
                self.closed = False
                readers.append(self)

            def read(self):
                return self.frames.pop() if self.frames else None

            def close(self):
                self.closed = True

        with patch("scripts.export_video_overlay.FrameReader", Reader):
            with self.assertRaisesRegex(ValueError, "no decodable frames.*b.avi"):
                extract_image_samples(self.args())
        self.assertEqual(len(readers), 2)
        self.assertTrue(all(reader.closed for reader in readers))


if __name__ == "__main__":
    unittest.main()
