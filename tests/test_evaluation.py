"""验证人工评估计数、锁定测试基准与报告防篡改门禁。"""
import copy
import hashlib
import unittest

from knowledge_model_finetuning.evaluation import evaluate, validate_report


def evaluation_records():
    """构造仅用于门禁测试的人工审核输入，不代表真实模型评估。"""
    records = []
    kinds = [("single", None), ("multi", None), ("citation", None)]
    kinds += [("refusal", kind) for kind in ("no_evidence", "irrelevant", "conflict", "missing_conditions")]
    for index, (task, refusal) in enumerate(kinds):
        answerable = task != "refusal"
        labels = ["S1", "S2"] if answerable or refusal == "conflict" else ["S1"]
        if refusal == "no_evidence":
            labels = []
        record = {"id": str(index), "task_type": task, "answerable": answerable,
                  "evidence_labels": labels,
                  "test_case_hash": hashlib.sha256(f"固定问题和证据{index}".encode()).hexdigest()}
        if refusal:
            record["refusal_type"] = refusal
        for model in ("base", "finetuned"):
            record[model] = {"text": "操作说明。[S1]" if answerable else "资料不足，无法回答。", "review": {
                "reviewer": "测试审核人", "reviewed_at": "2026-11-15T10:00:00+08:00",
                "supported_claims": (10 if model == "finetuned" else 9) if answerable else 0,
                "total_claims": 10 if answerable else 0, "covered_keypoints": (10 if model == "finetuned" else 9) if answerable else 0,
                "total_keypoints": 10 if answerable else 0,
                "terminology_format": True, "refused": not answerable,
                "refusal_correct": not answerable}}
        records.append(record)
    return records


class EvaluationTests(unittest.TestCase):
    """检验实际计数、覆盖和基座对照约束。"""

    def test_counts_hashes_and_groups_are_reproducible(self):
        """顺序不影响锁定基准，报告保留计数和审核。"""
        report = evaluate(evaluation_records())
        validate_report(report)
        self.assertTrue(report["passed"])
        self.assertEqual(report["counts"]["finetuned"]["evidence_support"], {"numerator": 30, "denominator": 30})
        self.assertEqual(report["test_set_hash"], evaluate(list(reversed(evaluation_records())))["test_set_hash"])
        self.assertEqual(set(report["groups"]["refusal_type"]), {"no_evidence", "irrelevant", "conflict", "missing_conditions"})
        self.assertEqual(len(report["samples"][0]["base"]["output_hash"]), 64)

    def test_changed_output_keeps_benchmark_but_changed_case_does_not(self):
        """锁定集哈希绑定问题证据基准，独立于模型输出及评审结果。"""
        records = evaluation_records()
        first = evaluate(records)
        records[0]["finetuned"]["text"] += "新增说明。"
        self.assertEqual(first["test_set_hash"], evaluate(records)["test_set_hash"])
        records[0]["test_case_hash"] = "f" * 64
        self.assertNotEqual(first["test_set_hash"], evaluate(records)["test_set_hash"])

    def test_forged_report_or_unsupported_counts_are_rejected(self):
        """发布验证重算报告，拒绝伪造通过标志、比例、hash及人工计数。"""
        good = evaluate(evaluation_records())
        for key in ("metrics", "counts", "test_set_hash", "passed", "groups", "differences"):
            report = copy.deepcopy(good)
            report[key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_report(report)
        records = evaluation_records()
        for value in (-1, True, 11):
            bad = copy.deepcopy(records)
            bad[0]["finetuned"]["review"]["supported_claims"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate(bad)

    def test_missing_review_case_hash_and_inconsistent_keypoints_fail(self):
        """没有真实审核与稳定内容基准的记录不可参与正式评估。"""
        for field in ("test_case_hash", "base"):
            records = evaluation_records()
            del records[0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate(records)
        records = evaluation_records()
        records[0]["finetuned"]["review"]["total_keypoints"] = 11
        with self.assertRaises(ValueError):
            evaluate(records)

    def test_gates_reject_invalid_citation_coverage_and_no_improvement(self):
        """无引用、无覆盖、没有改善及忠实性退步均不能发布。"""
        cases = []
        records = evaluation_records()
        records[0]["finetuned"]["text"] = "未引用的答案。"
        cases.append(records)
        records = evaluation_records()
        records[0]["finetuned"]["text"] = "伪造引用。[S3]"
        cases.append(records)
        cases.append(evaluation_records()[:-1])
        records = evaluation_records()
        for record in records:
            record["base"] = copy.deepcopy(record["finetuned"])
        cases.append(records)
        records = evaluation_records()
        for record in records:
            record["base"]["review"]["supported_claims"] = 10 if record["answerable"] else 0
        records[0]["finetuned"]["review"]["supported_claims"] = 9
        cases.append(records)
        for index, records in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_report(evaluate(records))

    def test_each_numeric_gate_is_enforced(self):
        """证据支持、关键点、术语、拒答召回及误拒均执行实际阈值。"""
        mutations = [("supported_claims", 0), ("covered_keypoints", 0), ("terminology_format", False)]
        for field, value in mutations:
            records = evaluation_records()
            records[0]["finetuned"]["review"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_report(evaluate(records))
        records = evaluation_records()
        records[-1]["finetuned"]["review"]["refusal_correct"] = False
        with self.assertRaises(ValueError):
            validate_report(evaluate(records))
        records = evaluation_records()
        records[0]["finetuned"]["review"]["refused"] = True
        with self.assertRaises(ValueError):
            validate_report(evaluate(records))

    def test_integer_counts_cannot_round_a_failed_threshold_into_passing(self):
        """极大计数即使浮点显示95%，实际低于门槛仍须失败。"""
        records = evaluation_records()
        for record in records:
            if not record["answerable"]:
                continue
            record["base"]["review"]["total_claims"] = 10 ** 20
            record["base"]["review"]["supported_claims"] = 9 * 10 ** 19
            record["finetuned"]["review"]["total_claims"] = 10 ** 20
            record["finetuned"]["review"]["supported_claims"] = 95 * 10 ** 18
        records[0]["finetuned"]["review"]["supported_claims"] -= 1
        with self.assertRaises(ValueError):
            validate_report(evaluate(records))

    def test_empty_denominators_and_missing_task_coverage_cannot_pass(self):
        """空集、无主任务和拒答零声明均不能以空分母冒充质量通过。"""
        with self.assertRaises(ValueError):
            validate_report(evaluate([]))
        with self.assertRaises(ValueError):
            validate_report(evaluate([item for item in evaluation_records() if item["task_type"] == "refusal"]))

    def test_false_refusal_consistency_and_citation_labels_are_checked(self):
        """正确拒答标记与任务含义必须一致，证据编号不能重复或跳号。"""
        for labels in (["S1", "S1"], ["S2"], ["S01"]):
            records = evaluation_records()
            records[0]["evidence_labels"] = labels
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evaluate(records)
        records = evaluation_records()
        records[0]["finetuned"]["review"]["refusal_correct"] = True
        with self.assertRaises(ValueError):
            evaluate(records)

    def test_refusal_subtypes_require_matching_evidence_counts(self):
        """四种拒答必须保留可判定的证据数量，空证据不能虚增能力覆盖。"""
        invalid = [("no_evidence", ["S1"]), ("irrelevant", []),
                   ("missing_conditions", []), ("conflict", []), ("conflict", ["S1"])]
        for refusal_type, labels in invalid:
            records = evaluation_records()
            record = next(item for item in records if item.get("refusal_type") == refusal_type)
            record["evidence_labels"] = labels
            with self.subTest(refusal_type=refusal_type, labels=labels), self.assertRaises(ValueError):
                evaluate(records)

    def test_claims_without_valid_citations_are_unsupported(self):
        """人工误填支持数不能让无有效引用的事实增加忠实性。"""
        for text in ("没有引用的回答。", "无效来源。[S99]"):
            records = evaluation_records()
            for record in records:
                if record["answerable"]:
                    record["base"]["text"] = text
            report = evaluate(records)
            self.assertEqual(report["counts"]["base"]["evidence_support"], {"numerator": 0, "denominator": 30})
