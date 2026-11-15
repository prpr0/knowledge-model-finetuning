"""正式导出门禁测试；所有样例均为临时合成数据。"""
import json
from pathlib import Path
import tempfile
import unittest

from knowledge_model_finetuning.corpus import assign_splits
from knowledge_model_finetuning.dataset import export_dataset
from test_corpus import document


def fixtures():
    """创建覆盖任务和拒答分层的合成候选。"""
    docs = assign_splits([document(i) for i in range(30)])
    samples = []
    for split in ("train", "validation", "test"):
        selected = [d for d in docs if d["split"] == split]
        for kind in ("single", "multi", "citation", "no_evidence", "irrelevant", "conflict", "missing_conditions"):
            refusal = kind not in ("single", "multi", "citation")
            evidence = [] if kind == "no_evidence" else [dict(document_id=d["id"], title="测试资料", version=d["version"], location=d["section"], content=d["text"]) for d in selected[:2 if kind in ("multi", "conflict") else 1]]
            sample = dict(id=f"{split}-{kind}", split=split, task_type="refusal" if refusal else kind, question=f"{split} {kind} 的问题", evidence=evidence, answer=f"资料不足，无法判断{split}{kind}。" if refusal else f"根据材料回答{split}{kind}。[S1]", batch_id="test-batch", review=dict(status="approved", reviewer="测试审核者", reviewed_at="2026-11-15T10:00:00+00:00"))
            if refusal:
                sample["refusal_type"] = kind
            samples.append(sample)
    return docs, samples


class DatasetTests(unittest.TestCase):
    """检查导出一致性和失败原子性。"""

    def test_export_schema(self):
        """输出独立消息、清单与摘要。"""
        docs, samples = fixtures()
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "export"
            result = export_dataset(samples, docs, target, lambda messages: 100)
            self.assertEqual(result["split_counts"], dict(train=7, validation=7, test=7))
            row = json.loads((target / "train.jsonl").read_text().splitlines()[0])
            self.assertEqual(set(row), {"messages"})
            manifest = json.loads((target / "manifest.jsonl").read_text().splitlines()[0])
            self.assertEqual(manifest["line_number"], 1)
            self.assertIn("sample_hash", manifest)
            with self.assertRaises(ValueError):
                export_dataset(samples, docs, target, lambda messages: 100)

    def test_failures_leave_no_dataset(self):
        """拒绝超长、缺审核、错引用和错命令，失败不留目录。"""
        docs, samples = fixtures()
        for change, counter in (({}, lambda messages: 4097), ({"answer": "结论[S99]"}, lambda messages: 100), ({"answer": "运行 `invented --bad`。[S1]"}, lambda messages: 100), ({"review": {"status": "pending"}}, lambda messages: 100)):
            candidates = [s.copy() for s in samples]
            candidates[7].update(change)
            with tempfile.TemporaryDirectory() as root:
                target = Path(root) / "export"
                with self.assertRaises(ValueError):
                    export_dataset(candidates, docs, target, counter)
                self.assertFalse(target.exists())

    def test_provenance_and_coverage_gates(self):
        """拒绝跨区引用、批次拒审、覆盖不足和非整数计数。"""
        for mode in ("cross_split", "rejected", "coverage", "tokens", "digits"):
            docs, samples = fixtures()
            counter = lambda messages: 100
            if mode == "cross_split":
                other = next(d for d in docs if d["split"] != samples[0]["split"])
                samples[0]["evidence"][0]["document_id"] = other["id"]
            elif mode == "rejected":
                samples[0]["review"]["status"] = "rejected"
            elif mode == "coverage":
                samples.pop()
            elif mode == "tokens":
                counter = lambda messages: True
            else:
                samples[0]["answer"] = "需要999999步。[S1]"
            with tempfile.TemporaryDirectory() as root:
                target = Path(root) / "export"
                with self.assertRaises(ValueError):
                    export_dataset(samples, docs, target, counter)
                self.assertFalse(target.exists())

    def test_refusal_subtype_requires_matching_evidence_shape(self):
        """不能用空证据样本冒充无关、冲突或缺条件能力覆盖。"""
        for kind in ("no_evidence", "irrelevant", "conflict", "missing_conditions"):
            docs, samples = fixtures()
            candidate = next(sample for sample in samples if sample.get("refusal_type") == kind)
            if kind == "no_evidence":
                candidate["evidence"] = samples[0]["evidence"]
            else:
                candidate["evidence"] = candidate["evidence"][:1] if kind == "conflict" else []
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaises(ValueError):
                    export_dataset(samples, docs, Path(root) / "dataset", lambda messages: 100)
