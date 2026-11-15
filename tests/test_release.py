"""验证发布完整性与目录边界，临时小文件仅是防篡改测试夹具。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from knowledge_model_finetuning.evaluation import evaluate
from knowledge_model_finetuning.release import build_release, verify_release
from test_evaluation import evaluation_records


class ReleaseTests(unittest.TestCase):
    """覆盖真实文件清单规则，不宣称夹具是实际模型。"""

    def setUp(self):
        """准备仅用于路径与摘要测试的小文件。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        groups = {"adapter": ["adapter/adapter_model.safetensors", "adapter/adapter_config.json"],
                  "merged_model": ["merged/model.safetensors"], "gguf": ["quantized/model.gguf"],
                  "tokenizer": ["tokenizer/tokenizer.json", "tokenizer/chat_template.jinja"],
                  "training_config": ["training/config.json"], "data_manifest": ["data/manifest.jsonl"],
                  "licenses": ["licenses/NOTICE.txt"]}
        for paths in groups.values():
            for name in paths:
                target = self.root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("{}" if name.endswith(".json") else "仅防篡改规则测试夹具", encoding="utf-8")
        report = evaluate(evaluation_records())
        for name in ("evaluation.json", "quantized-evaluation.json"):
            (self.root / name).write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        self.specification = {"model_version": "test-fixture-only", "base_revision": "a" * 40,
                              "evaluation_report": "evaluation.json", "quantized_evaluation_report": "quantized-evaluation.json",
                              "artifact_groups": groups}

    def write_manifest(self, manifest):
        """把被测清单写入临时目录，模拟CLI的文件职责。"""
        path = self.root / "release-manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        return path

    def test_round_trip_includes_reports_and_platform_fields(self):
        """完整清单可以重验，两个报告都纳入逐文件摘要。"""
        manifest = build_release(self.root, self.specification)
        self.assertEqual(manifest, verify_release(self.write_manifest(manifest)))
        self.assertEqual(manifest["model_name"], "kb-grounded-qwen3-4b-v1")
        self.assertEqual(manifest["base_model"], "Qwen/Qwen3-4B-Instruct-2507")
        paths = {item["path"] for item in manifest["artifacts"]}
        self.assertTrue({"evaluation.json", "quantized-evaluation.json"} <= paths)
        self.assertFalse((self.root / "release-manifest.json").is_dir())

    def test_tampering_or_missing_hash_is_rejected(self):
        """修改产物以及去除报告摘要都不能通过验签。"""
        manifest = build_release(self.root, self.specification)
        path = self.write_manifest(manifest)
        artifact = self.root / "merged/model.safetensors"
        artifact.write_text("改变后的夹具", encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_release(path)
        manifest = build_release(self.root, self.specification)
        manifest["artifacts"] = [item for item in manifest["artifacts"] if item["path"] != "evaluation.json"]
        with self.assertRaises(ValueError):
            verify_release(self.write_manifest(manifest))

    def test_revision_groups_duplicates_and_weight_shapes_are_required(self):
        """不可变revision、固定分组及必要权重格式不可缺失。"""
        for revision in ("main", "a" * 39, "g" * 40):
            specification = copy.deepcopy(self.specification)
            specification["base_revision"] = revision
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                build_release(self.root, specification)
        for group in self.specification["artifact_groups"]:
            specification = copy.deepcopy(self.specification)
            specification["artifact_groups"][group] = []
            with self.subTest(group=group), self.assertRaises(ValueError):
                build_release(self.root, specification)
        specification = copy.deepcopy(self.specification)
        specification["artifact_groups"]["licenses"].append("adapter/adapter_model.safetensors")
        with self.assertRaises(ValueError):
            build_release(self.root, specification)
        specification = copy.deepcopy(self.specification)
        specification["artifact_groups"]["adapter"] = ["adapter/adapter_model.safetensors"]
        with self.assertRaises(ValueError):
            build_release(self.root, specification)
        (self.root / "merged/model.bin").write_bytes(b"fixture")
        specification = copy.deepcopy(self.specification)
        specification["artifact_groups"]["merged_model"] = ["merged/model.bin"]
        with self.assertRaises(ValueError):
            build_release(self.root, specification)

    def test_portable_path_boundaries_and_symlinks(self):
        """即使在非Windows执行，也拒绝Windows路径穿越与盘符。"""
        for name in ("../outside", "/absolute", "C:/outside", "C:outside", "..\\outside", "\\\\host\\share", "licenses/../evaluation.json", "licenses/NOTICE.txt:stream", "licenses/NOTICE.txt."):
            specification = copy.deepcopy(self.specification)
            specification["artifact_groups"]["licenses"] = [name]
            with self.subTest(name=name), self.assertRaises(ValueError):
                build_release(self.root, specification)
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "external.txt"
            target.write_text("outside", encoding="utf-8")
            link = self.root / "escape.txt"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("当前环境未授予创建符号链接权限")
            specification = copy.deepcopy(self.specification)
            specification["artifact_groups"]["licenses"] = ["escape.txt"]
            with self.assertRaises(ValueError):
                build_release(self.root, specification)

    def test_quantized_gate_and_stable_test_set_are_mandatory(self):
        """量化报告执行同一门槛，且基准不能仅凭相同样本ID判定。"""
        records = evaluation_records()
        records[0]["test_case_hash"] = "f" * 64
        quantized = self.root / "quantized-evaluation.json"
        quantized.write_text(json.dumps(evaluate(records)), encoding="utf-8")
        with self.assertRaises(ValueError):
            build_release(self.root, self.specification)
        records = evaluation_records()
        records[0]["finetuned"]["text"] = "无引用"
        report = evaluate(records)
        report["passed"] = True
        quantized.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaises(ValueError):
            build_release(self.root, self.specification)

    def test_empty_files_missing_template_and_aliases_cannot_publish(self):
        """空文件、没有模板和重复文件别名均不能满足产物分组。"""
        specification = copy.deepcopy(self.specification)
        specification["artifact_groups"]["tokenizer"] = ["tokenizer/tokenizer.json"]
        with self.assertRaises(ValueError):
            build_release(self.root, specification)
        target = self.root / "licenses/NOTICE.txt"
        target.write_bytes(b"")
        with self.assertRaises(ValueError):
            build_release(self.root, self.specification)
        target.write_bytes(b"fixture")
        specification = copy.deepcopy(self.specification)
        specification["evaluation_report"] = specification["quantized_evaluation_report"]
        with self.assertRaises(ValueError):
            build_release(self.root, specification)

    def test_hash_covers_data_beyond_first_chunk(self):
        """大于单个摘要块的测试夹具尾部变动也必须被检出。"""
        target = self.root / "merged/model.safetensors"
        target.write_bytes(b"x" * (1024 * 1024 + 17))
        manifest_path = self.write_manifest(build_release(self.root, self.specification))
        with target.open("r+b") as output:
            output.seek(-1, 2)
            output.write(b"y")
        with self.assertRaises(ValueError):
            verify_release(manifest_path)
