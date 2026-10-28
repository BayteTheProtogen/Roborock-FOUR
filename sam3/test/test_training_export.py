"""Verify training schemas and upload safeguards without loading SAM weights."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from scripts.export_video_overlay import _infer_image_batch, export_image_dataset, parse_args
from scripts.training_export import (
    LARGE_IMAGE_BYTES, archive_size, compress_image, export_training_dataset,
    joined_polygon, label_rows, large_image_decision, mask_instances, package_entries,
    main as reexport_main,
)


class TrainingExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()

    def args(self, **changes):
        values = dict(output=self.source, export_format="yolo", export_task="segment",
                      text="sidewalk", class_name=None, val_fraction=.2, chunk_size=0, chunk_max_mb=0)
        values.update(changes)
        return SimpleNamespace(**values)

    def rows(self, count=4):
        (self.source / "images").mkdir(exist_ok=True)
        mask = np.zeros((12, 20), dtype=np.uint8)
        mask[2:8, 4:10] = 1
        rows = []
        for index in range(count):
            image = f"images/{index:06d}.jpg"
            cv2.imwrite(str(self.source / image), np.full((12, 20, 3), 100, dtype=np.uint8))
            rows.append(dict(frame_index=index, image=image, width=20, height=12,
                             instances=mask_instances([mask], [.9]) if index else [], split="train"))
        (self.source / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return rows

    def test_separate_instances_boxes_area_and_tiny_components(self):
        a = np.zeros((12, 20), dtype=np.uint8)
        a[2:8, 4:10] = 1
        b = np.zeros_like(a)
        b[0, 0] = 1
        b[10:12, 18:20] = 1
        instances = mask_instances([a, b, np.zeros_like(a)], [.9, .8, .7])
        self.assertEqual(len(instances), 2)
        self.assertEqual(instances[0]["bbox"], [4, 2, 6, 6])
        self.assertEqual(instances[0]["area"], 36)
        self.assertEqual(len(instances[1]["segmentation"]), 2)
        self.assertTrue(all(len(p) >= 6 for p in instances[1]["segmentation"]))
        row = dict(instances=instances, width=20, height=12)
        self.assertEqual(label_rows(row, "detect")[0], [0, .35, 5 / 12, .3, .5])
        self.assertEqual(len(label_rows(row, "segment")), 2)
        for label in label_rows(row, "segment"):
            self.assertTrue(all(0 <= v <= 1 for v in label[1:]))

    def test_bridges_keep_disconnected_components_in_one_instance(self):
        polygons = [[0, 0, 2, 0, 2, 2, 0, 2], [7, 7, 9, 7, 9, 9, 7, 9]]
        joined = joined_polygon(polygons)
        # A bridge traversed in both directions contributes no extra area.
        self.assertEqual(cv2.contourArea(joined.astype(np.float32)), 8)
        self.assertEqual(set(map(tuple, joined)), set(map(tuple, np.array(polygons).reshape(-1, 2))))

    def test_yolo_zip_layout_normalization_and_background_labels(self):
        self.rows()
        manifest = export_training_dataset(self.args(export_task="detect", class_name='road: "sidewalk"'))
        with zipfile.ZipFile(self.source / "training" / "dataset.zip") as handle:
            self.assertEqual(handle.read("labels/train/000000.txt"), b"")
            values = list(map(float, handle.read("labels/train/000001.txt").split()))
            self.assertEqual(values, [0, .35, 5 / 12, .3, .5])
            config = handle.read("data.yaml").decode()
            self.assertIn("train: images/train", config)
            self.assertIn("val: images/val", config)
            self.assertNotIn(str(self.source), config)
            self.assertIn("images/val/000003.jpg", handle.namelist())
        self.assertEqual(manifest["chunks"][0]["val"], 1)

    def test_ndjson_metadata_first_and_images_resolve_after_moving(self):
        self.rows()
        export_training_dataset(self.args(export_format="ndjson"))
        with zipfile.ZipFile(self.source / "training" / "dataset.zip") as handle:
            destination = self.root / "moved"
            handle.extractall(destination)
        records = [json.loads(line) for line in (destination / "dataset.ndjson").read_text().splitlines()]
        self.assertEqual(records[0]["type"], "dataset")
        self.assertEqual(records[0]["task"], "segment")
        self.assertEqual(records[0]["class_names"], {"0": "sidewalk"})
        for record in records[1:]:
            image = destination / records[0]["path"] / "images" / record["split"] / record["file"]
            self.assertTrue(image.is_file())
            self.assertEqual(cv2.imread(str(image)).shape[:2], (record["height"], record["width"]))
            self.assertIn("segments", record["annotations"])
        self.assertEqual(records[1]["annotations"]["segments"], [])
        # Platform's documented precedence detects the YAML companion for cloud
        # import; every image and label can be resolved without a remote URL.
        import yaml
        config = yaml.safe_load((destination / "data.yaml").read_text())
        self.assertEqual(config["names"], {0: "sidewalk"})
        for record in records[1:]:
            label = destination / "labels" / record["split"] / (Path(record["file"]).stem + ".txt")
            actual = [list(map(float, line.split())) for line in label.read_text().splitlines()]
            self.assertEqual(actual, record["annotations"]["segments"])

    def test_hosted_ndjson_has_fetchable_urls_and_no_local_path_override(self):
        rows = self.rows()
        entries = dict(package_entries(self.source, rows, "ndjson", "detect", ["sidewalk", "person"],
                                       "https://example.org/dataset"))
        self.assertNotIn("data.yaml", entries)
        records = [json.loads(line) for line in entries["dataset.ndjson"].splitlines()]
        self.assertNotIn("path", records[0])
        self.assertEqual(records[0]["class_names"], {"0": "sidewalk", "1": "person"})
        self.assertEqual(records[1]["url"], "https://example.org/dataset/images/train/000000.jpg")
        export_training_dataset(self.args(export_format="ndjson", ndjson_image_url_base="https://example.org/dataset"))
        self.assertTrue((self.source / "training" / "dataset.ndjson").is_file())

    def test_saved_annotations_can_be_reexported_without_replacing_old_files(self):
        self.rows()
        config = {"text_prompt": "sidewalk", "class_name": "road", "export_task": "detect", "val_fraction": .25}
        (self.source / "dataset.json").write_text(json.dumps(config))
        destination = self.root / "fixed"
        before = (self.source / "metadata.jsonl").read_bytes()
        self.assertEqual(reexport_main([str(self.source), str(destination)]), 0)
        self.assertEqual(before, (self.source / "metadata.jsonl").read_bytes())
        with zipfile.ZipFile(destination / "dataset.zip") as handle:
            self.assertIn("data.yaml", handle.namelist())
            records = [json.loads(line) for line in handle.read("dataset.ndjson").splitlines()]
            self.assertEqual(records[0]["class_names"], {"0": "road"})
            self.assertEqual(records[0]["task"], "detect")
        # Do not emit a path-only standalone upload that will fail in the cloud.
        self.assertFalse((destination / "dataset.ndjson").exists())
        self.assertEqual(reexport_main([str(self.source), str(destination)]), 1)

    def test_class_metadata_is_preserved_for_all_chunks_even_if_class_absent(self):
        rows = self.rows(5)
        rows[1]["instances"][0]["class_id"] = 1
        (self.source / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        classes = [{"id": 0, "name": "ground", "text": "sidewalk"},
                   {"id": 1, "name": "pedestrian", "text": "person"}]
        manifest = export_training_dataset(self.args(export_format="ndjson", chunk_size=2, classes=classes))
        for chunk in manifest["chunks"]:
            with zipfile.ZipFile(self.source / "training" / chunk["file"]) as handle:
                records = [json.loads(line) for line in handle.read("dataset.ndjson").splitlines()]
                self.assertEqual(records[0]["class_names"], {"0": "ground", "1": "pedestrian"})
                if chunk["file"] == "chunk-0001.zip":
                    self.assertEqual(records[2]["annotations"]["segments"][0][0], 1)

    def test_coco_pixel_boxes_categories_and_instance_links(self):
        self.rows()
        export_training_dataset(self.args(export_format="coco"))
        with zipfile.ZipFile(self.source / "training" / "dataset.zip") as handle:
            coco = json.loads(handle.read("annotations/instances_train.json"))
            self.assertEqual(coco["categories"][0]["id"], 1)
            ids = {image["id"] for image in coco["images"]}
            self.assertEqual(len(coco["annotations"]), 2)
            for annotation in coco["annotations"]:
                self.assertIn(annotation["image_id"], ids)
                self.assertEqual(annotation["bbox"], [4, 2, 6, 6])
                self.assertEqual(annotation["area"], 36)
                self.assertTrue(annotation["segmentation"])
            for image in coco["images"]:
                self.assertIn(image["file_name"], handle.namelist())

    def test_chunks_have_no_duplicate_images_and_complete_metadata(self):
        self.rows(5)
        manifest = export_training_dataset(self.args(export_format="ndjson", chunk_size=2))
        seen = set()
        for chunk in manifest["chunks"]:
            self.assertLessEqual(chunk["images"], 2)
            with zipfile.ZipFile(self.source / "training" / chunk["file"]) as handle:
                records = [json.loads(line) for line in handle.read("dataset.ndjson").splitlines()]
                self.assertEqual(records[0]["type"], "dataset")
                for record in records[1:]:
                    self.assertNotIn(record["file"], seen)
                    seen.add(record["file"])
        self.assertEqual(len(seen), 5)
        self.assertEqual(sum(chunk["val"] for chunk in manifest["chunks"]), 1)

    def test_byte_limit_includes_zip_and_annotation_overhead(self):
        rows = self.rows(4)
        for export_format in ("yolo", "ndjson", "coco"):
            # A limit accommodating any one image, but too small for two.
            single_max = max(archive_size(package_entries(self.source, [r], export_format, "segment", "sidewalk"))
                             for r in rows)
            destination = self.root / export_format
            destination.mkdir()
            # Run each export from its own source to avoid replacing previous output.
            import shutil
            shutil.copytree(self.source / "images", destination / "images")
            shutil.copy2(self.source / "metadata.jsonl", destination / "metadata.jsonl")
            args = self.args(output=destination, export_format=export_format, chunk_max_mb=(single_max + 100) / 1_000_000)
            manifest = export_training_dataset(args)
            self.assertGreater(len(manifest["chunks"]), 1)
            for chunk in manifest["chunks"]:
                self.assertLessEqual(chunk["bytes"], single_max + 100)

    def test_unfit_single_image_fails_before_writing_archives(self):
        self.rows()
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            export_training_dataset(self.args(chunk_max_mb=.0001))
        self.assertEqual(list((self.source / "training").glob("*.zip")), [])

    def test_exact_50_mb_threshold_all_three_choices_and_invalid_input(self):
        path = self.root / "large.jpg"
        with path.open("wb") as handle:
            handle.truncate(LARGE_IMAGE_BYTES)
        with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=["bad", "compress"]):
            self.assertEqual(large_image_decision([path]), "compress")
        self.assertEqual(large_image_decision([path], "continue"), "continue")
        with self.assertRaisesRegex(ValueError, "aborted"):
            large_image_decision([path], "abort")
        with patch("sys.stdin.isatty", return_value=False):
            with self.assertRaisesRegex(ValueError, "require a decision"):
                large_image_decision([path])
        with path.open("wb") as handle:
            handle.truncate(LARGE_IMAGE_BYTES - 1)
        self.assertEqual(large_image_decision([path], "abort"), "continue")

    def test_compression_preserves_dimensions_and_meets_limit(self):
        path = self.root / "copy.jpg"
        frame = np.random.default_rng(8).integers(0, 256, (100, 120, 3), dtype=np.uint8)
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with patch("scripts.training_export.LARGE_IMAGE_BYTES", 5_000):
            quality = compress_image(path)
        self.assertLess(path.stat().st_size, 5_000)
        self.assertEqual(cv2.imread(str(path)).shape, frame.shape)
        self.assertLessEqual(quality, 90)

    def test_source_abort_happens_before_decoding_or_output_creation(self):
        path = self.source / "large.jpg"
        with path.open("wb") as handle:
            handle.truncate(LARGE_IMAGE_BYTES)
        output = self.root / "output"
        args = parse_args([str(self.source), str(output), "--image-mode", "--text", "road",
                           "--export-format", "yolo", "--large-image-action", "abort"])
        with patch("scripts.export_video_overlay.FrameReader.read", side_effect=AssertionError("must not decode")):
            with self.assertRaisesRegex(ValueError, "aborted"):
                export_image_dataset(args)
        self.assertFalse(output.exists())

    def test_exported_frame_abort_happens_before_labeling(self):
        cv2.imwrite(str(self.source / "small.jpg"), np.zeros((2, 2, 3), dtype=np.uint8))
        args = parse_args([str(self.source), str(self.root / "output"), "--image-mode", "--text", "road",
                           "--large-image-action", "abort"])
        def oversized_image(path, frame, parameters):
            with path.open("wb") as handle:
                handle.truncate(LARGE_IMAGE_BYTES)
        with patch("scripts.export_video_overlay.write_image", side_effect=oversized_image), \
             patch("scripts.export_video_overlay.mp.get_context") as worker_context:
            with self.assertRaisesRegex(ValueError, "aborted"):
                export_image_dataset(args)
            worker_context.assert_not_called()
        self.assertFalse((args.output / "dataset.json").exists())

    def test_full_export_flow_preserves_instances_for_all_formats(self):
        import torch
        from sam3.model import sam3_image_processor
        from sam3 import model_builder

        cv2.imwrite(str(self.source / "small.jpg"), np.full((12, 20, 3), 100, dtype=np.uint8))
        checkpoint = self.root / "checkpoint.pt"
        checkpoint.touch()

        events = []

        class Processor:
            def __init__(self, *args, **kwargs):
                pass

            def encode_text_prompt(self, text):
                events.append(("encode", text))
                return {"prompt": text}

            def set_image(self, image):
                events.append(("image",))
                return dict(original_height=image.height, original_width=image.width)

            def set_text_prompt(self, text, state, **kwargs):
                if kwargs["text_features"]["prompt"] != text or "masks" in state:
                    raise AssertionError("cached prompts or previous results leaked between classes")
                events.append(("ground", text))
                if text == "person":
                    masks = torch.zeros((1, 1, state["original_height"], state["original_width"]), dtype=torch.bool)
                    masks[0, 0, :2, :2] = True
                    return dict(state, masks=masks, scores=torch.tensor([.95]))
                masks = torch.zeros((2, 1, state["original_height"], state["original_width"]), dtype=torch.bool)
                masks[0, 0, 2:8, 4:10] = True
                masks[1, 0, 5:11, 8:15] = True  # overlapping objects must remain separate
                return dict(state, masks=masks, scores=torch.tensor([.9, .8]))

        class Worker:
            def __init__(self, target, args):
                self.args = args
                self.exitcode = None

            def start(self):
                _infer_image_batch(*self.args)
                self.exitcode = 0

            def join(self):
                pass

        def build_model(**kwargs):
            return SimpleNamespace(backbone=SimpleNamespace(language_backbone=torch.nn.Linear(2, 2)))

        for export_format, multiclass in ((fmt, multi) for fmt in ("yolo", "coco", "ndjson") for multi in (False, True)):
            events.clear()
            output = self.root / f"{export_format}-{multiclass}"
            flags = ["--text", "person", "--class-name", "ground", "--class-name", "pedestrian"] if multiclass else []
            args = parse_args([str(self.source), str(output), "--image-mode", "--text", "road",
                               "--device", "cpu", "--checkpoint", str(checkpoint), "--no-overlays",
                               "--export-format", export_format, *flags])
            with patch.object(sam3_image_processor, "Sam3Processor", Processor), \
                 patch.object(model_builder, "build_sam3_image_model", side_effect=build_model), \
                 patch("scripts.export_video_overlay.configure_model"), \
                 patch("scripts.export_video_overlay.mp.get_context", return_value=SimpleNamespace(Process=Worker)):
                export_image_dataset(args)
            row = json.loads((args.output / "metadata.jsonl").read_text())
            expected_ids = [0, 0, 1] if multiclass else [0, 0]
            self.assertEqual([instance["class_id"] for instance in row["instances"]], expected_ids)
            self.assertEqual(row["class_ids"], expected_ids)
            self.assertEqual(row["mask_count"], len(expected_ids))
            self.assertEqual(events.count(("image",)), 1)
            self.assertEqual(events.count(("encode", "road")), 1)
            if multiclass:
                self.assertEqual(events.count(("encode", "person")), 1)
                self.assertEqual(set(row["class_masks"]), {"0", "1"})
                for path in row["class_masks"].values():
                    self.assertTrue((args.output / path).is_file())
            union = cv2.imread(str(args.output / row["mask"]), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(np.count_nonzero(union), 76 if multiclass else 72)
            with zipfile.ZipFile(args.output / "training" / "dataset.zip") as handle:
                if export_format == "yolo":
                    labels = handle.read("labels/train/000000.txt").splitlines()
                    self.assertEqual([int(line.split()[0]) for line in labels], expected_ids)
                elif export_format == "coco":
                    coco = json.loads(handle.read("annotations/instances_train.json"))
                    self.assertEqual([ann["category_id"] for ann in coco["annotations"]], [i + 1 for i in expected_ids])
                    self.assertEqual(len(coco["categories"]), 2 if multiclass else 1)
                else:
                    records = [json.loads(line) for line in handle.read("dataset.ndjson").splitlines()]
                    self.assertEqual([ann[0] for ann in records[1]["annotations"]["segments"]], expected_ids)

    def test_cli_rejects_invalid_combinations(self):
        cv2.imwrite(str(self.source / "small.jpg"), np.zeros((2, 2, 3), dtype=np.uint8))
        base = [str(self.source), str(self.root / "out"), "--text", "road"]
        for flags in (["--image-mode", "--chunk-size", "10"],
                      ["--export-format", "yolo"], ["--image-mode", "--chunk-size", "-1"],
                      ["--image-mode", "--val-fraction", "nan"],
                      ["--image-mode", "--chunk-max-mb", "inf"],
                      ["--text", "person"],
                      ["--image-mode", "--text", "person", "--class-name", "road"],
                      ["--image-mode", "--text", "road"],
                      ["--image-mode", "--export-format", "ndjson", "--ndjson-image-url-base", "file:///tmp"],
                      ["--image-mode", "--ndjson-image-url-base", "https://example.org"],
                      ["--drop-empty"]):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                parse_args(base + flags)

    def _export_empty_fixture(self, output_name, drop_empty=False, multiclass=False,
                              all_empty=False, export_format="yolo"):
        import torch
        from sam3.model import sam3_image_processor
        from sam3 import model_builder

        inputs = self.root / f"inputs-{output_name}"
        inputs.mkdir()
        for index, value in enumerate((0, 0 if all_empty else 100)):
            cv2.imwrite(str(inputs / f"{index}.jpg"), np.full((12, 20, 3), value, dtype=np.uint8))
        originals = {path: path.read_bytes() for path in inputs.iterdir()}
        checkpoint = self.root / "checkpoint.pt"
        checkpoint.touch()
        prompts = ["road", "person"] if multiclass else ["person"]
        flags = [flag for text in prompts for flag in ("--text", text)]
        args = parse_args([str(inputs), str(self.root / output_name), "--image-mode", *flags,
                           "--device", "cpu", "--checkpoint", str(checkpoint),
                           "--restart-every", "1", "--export-format", export_format,
                           *(["--drop-empty"] if drop_empty else [])])

        class Processor:
            def __init__(self, *args, **kwargs):
                pass

            def encode_text_prompt(self, text):
                return {}

            def set_image(self, image):
                return dict(original_height=image.height, original_width=image.width,
                            empty=image.getpixel((0, 0))[0] < 50)

            def set_text_prompt(self, text, state, **kwargs):
                count = int(not state["empty"] and text == "person")
                masks = torch.zeros((count, 1, 12, 20), dtype=torch.bool)
                if count:
                    masks[0, 0, 2:8, 4:10] = True
                return dict(state, masks=masks, scores=torch.ones(count) * .9)

        class Worker:
            def __init__(self, target, args):
                self.args = args
                self.exitcode = None

            def start(self):
                _infer_image_batch(*self.args)
                self.exitcode = 0

            def join(self):
                pass

        def build_model(**kwargs):
            return SimpleNamespace(backbone=SimpleNamespace(language_backbone=torch.nn.Linear(2, 2)))

        with patch.object(sam3_image_processor, "Sam3Processor", Processor), \
             patch.object(model_builder, "build_sam3_image_model", side_effect=build_model), \
             patch("scripts.export_video_overlay.configure_model"), \
             patch("scripts.export_video_overlay.mp.get_context", return_value=SimpleNamespace(Process=Worker)):
            export_image_dataset(args)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)
        return args.output

    def test_drop_empty_removes_generated_files_and_excludes_all_training_formats(self):
        for export_format in ("masks", "yolo", "coco", "ndjson"):
            with self.subTest(export_format=export_format):
                output = self._export_empty_fixture(export_format, drop_empty=True,
                                                     multiclass=True, export_format=export_format)
                rows = [json.loads(line) for line in (output / "metadata.jsonl").read_text().splitlines()]
                self.assertEqual([row["frame_index"] for row in rows], [1])
                self.assertEqual(rows[0]["class_counts"], [{"class_id": 0, "mask_count": 0},
                                                          {"class_id": 1, "mask_count": 1}])
                for directory in ("images", "masks", "overlays"):
                    self.assertEqual(list((output / directory).rglob("000000.*")), [])
                    self.assertTrue((output / directory / ("000001.png" if directory == "masks" else "000001.jpg")).is_file())
                self.assertEqual(len(list((output / "masks").rglob("000001.png"))), 3)
                config = json.loads((output / "dataset.json").read_text())
                self.assertEqual((config["num_extracted"], config["num_samples"], config["num_dropped_empty"]), (2, 1, 1))
                self.assertTrue(config["drop_empty"])
                if export_format != "masks":
                    with zipfile.ZipFile(output / "training" / "dataset.zip") as handle:
                        self.assertFalse(any("000000" in name for name in handle.namelist()))
                        images = [name for name in handle.namelist() if name.startswith("images/")]
                        self.assertEqual(images, ["images/train/000001.jpg"])

    def test_empty_images_remain_by_default(self):
        output = self._export_empty_fixture("default")
        rows = [json.loads(line) for line in (output / "metadata.jsonl").read_text().splitlines()]
        self.assertEqual([row["mask_count"] for row in rows], [0, 1])
        self.assertTrue((output / "images" / "000000.jpg").is_file())
        self.assertTrue((output / "masks" / "000000.png").is_file())
        self.assertTrue((output / "overlays" / "000000.jpg").is_file())

    def test_all_empty_export_finishes_without_archive(self):
        output = self._export_empty_fixture("all-empty", drop_empty=True, all_empty=True)
        self.assertEqual((output / "metadata.jsonl").read_text(), "")
        self.assertEqual(list((output / "images").iterdir()), [])
        self.assertEqual(list((output / "masks").iterdir()), [])
        self.assertEqual(list((output / "overlays").iterdir()), [])
        self.assertFalse((output / "training").exists())
        config = json.loads((output / "dataset.json").read_text())
        self.assertEqual((config["num_extracted"], config["num_samples"], config["num_dropped_empty"]), (2, 0, 2))


if __name__ == "__main__":
    unittest.main()
