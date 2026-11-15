"""通过独立进程验证命令行输出、错误码和文件保护行为。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CliTests(unittest.TestCase):
    """覆盖真实命令入口的消费边界。"""

    def invoke(self, *arguments):
        """在当前解释器的真实子进程调用项目入口。"""
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"), PYTHONIOENCODING="utf-8")
        return subprocess.run([sys.executable, "-B", "-m", "knowledge_model_finetuning", *map(str, arguments)], capture_output=True, text=True, encoding="utf-8", env=environment, check=False)

    def test_help_lists_usable_commands(self):
        """帮助列出真实子命令并正常退出。"""
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("partition", "export", "evaluate", "release", "verify-release"):
            self.assertIn(command, result.stdout)

    def test_invalid_corpus_fails_without_output_and_includes_run_id(self):
        """缺少溯源的资料必须失败，不能生成可用分区文件。"""
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source.jsonl", Path(directory) / "partitioned.jsonl"
            source.write_text('{"id":"missing-provenance"}\n', encoding="utf-8")
            result = self.invoke("partition", source, target)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(target.exists())
            self.assertIn("run_id=", result.stderr)
            self.assertIn("ERROR", result.stderr)

    def test_existing_output_is_not_overwritten(self):
        """错误输入或重复调用不能覆盖用户已有结果。"""
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "review.jsonl", Path(directory) / "report.json"
            source.write_text("", encoding="utf-8")
            target.write_text("user-owned-result", encoding="utf-8")
            result = self.invoke("evaluate", source, target)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(target.read_text(encoding="utf-8"), "user-owned-result")

    def test_export_requires_local_tokenizer_without_downloading(self):
        """未给出可用本地分词器时明确拒绝导出。"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "empty.jsonl"
            source.write_text("", encoding="utf-8")
            result = self.invoke("export", source, source, Path(directory) / "dataset", "--tokenizer", Path(directory) / "missing-tokenizer")
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("本地", result.stderr)
            self.assertFalse((Path(directory) / "dataset").exists())
