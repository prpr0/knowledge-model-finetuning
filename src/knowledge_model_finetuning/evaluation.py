"""生成和重算人工评估报告，不把审核输入当作已证实的训练结果。

报告schema_version为kb-evaluation-v1：samples保存规范输入、sample_hash及两路
output_hash；counts/metrics保留六指标分子分母与比例；groups按task_type和
refusal_type分组；coverage保存四任务、四拒答及五能力覆盖；differences是微调
减基座；failure_reasons和passed由固定门槛生成。空分母的比例为None且不通过。

每条输入必须额外提供test_case_hash，即锁定测试用例规范JSON的UTF-8 SHA-256，
其内容必须包括问题、完整编号证据（正文、版本、出处）和人工预定关键点或拒答
基准；建议json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(',', ':'))。
不得仅hash样本ID或模型输出。接口没有问题/证据正文，因此本模块只能要求并绑定
该内容承诺，不能代替调用方保存原始锁定用例或证明审核身份真实。

test_set_hash由排序后的样本ID、任务、拒答类型、answerable、证据编号、
test_case_hash和两模型共享total_keypoints计算，独立于模型输出及审核结果。
引用有效率按所有实际[S编号]出现次数统计；可回答输出缺引用补一个无效计数。
人工支持率按声明数微平均，关键点按可回答题关键点微平均，术语按样本平均；
拒答召回只计不可回答题中refused且refusal_correct，误拒只计可回答题。
拒答证据数量约束：no_evidence必须为零片；irrelevant和missing_conditions
至少一片；conflict至少两片，避免用空证据虚增四类拒答覆盖。
"""
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import re

from .prompt import TEMPLATE_VERSION

SCHEMA_VERSION = "kb-evaluation-v1"
TASK_TYPES = ("single", "multi", "citation", "refusal")
REFUSAL_TYPES = ("no_evidence", "irrelevant", "conflict", "missing_conditions")
MODEL_TYPES = ("base", "finetuned")
METRIC_NAMES = ("citation_validity", "evidence_support", "keypoint_coverage", "terminology_format", "refusal_recall", "answerable_false_refusal")
MINIMUMS = {"citation_validity": 1.0, "evidence_support": 0.95, "keypoint_coverage": 0.90,
            "terminology_format": 0.95, "refusal_recall": 0.95}
_HEX_HASH = re.compile(r"[0-9a-f]{64}\Z")
_CITATION = re.compile(r"\[S([^\]\r\n]*)\]")


def _hash(value: object) -> str:
    """计算规范JSON的稳定摘要，不依赖字典插入顺序。"""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _text(value: object, name: str) -> str:
    """要求非空字符串，保留原始文本用于输出摘要。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}必须为非空字符串")
    return value


def _review(value: object, answerable: bool) -> dict:
    """规范人工审核元数据，拒绝布尔冒充计数及不可能的上下界。"""
    if not isinstance(value, dict):
        raise ValueError("每个模型输出必须提供人工review")
    if "citation_validity" in value:
        raise ValueError("citation_validity只能由输出文本自动计算")
    result = {"reviewer": _text(value.get("reviewer"), "reviewer"),
              "reviewed_at": _text(value.get("reviewed_at"), "reviewed_at")}
    try:
        reviewed = datetime.fromisoformat(result["reviewed_at"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("reviewed_at必须是ISO 8601审核时间") from error
    if reviewed.tzinfo is None or reviewed.utcoffset() is None:
        raise ValueError("reviewed_at必须包含时区")
    for name in ("supported_claims", "total_claims", "covered_keypoints", "total_keypoints"):
        count = value.get(name)
        if type(count) is not int or count < 0:
            raise ValueError(f"{name}必须为非负整数")
        result[name] = count
    if result["supported_claims"] > result["total_claims"]:
        raise ValueError("支持声明数不能大于声明总数")
    if result["covered_keypoints"] > result["total_keypoints"]:
        raise ValueError("覆盖关键点数不能大于关键点总数")
    if answerable and result["total_keypoints"] == 0:
        raise ValueError("可回答题必须有预定关键点")
    if not answerable and result["total_keypoints"] != 0:
        raise ValueError("不可回答题不能计入可回答关键点分母")
    for name in ("terminology_format", "refused", "refusal_correct"):
        if type(value.get(name)) is not bool:
            raise ValueError(f"{name}必须为布尔值")
        result[name] = value[name]
    if result["refusal_correct"] and (answerable or not result["refused"]):
        raise ValueError("正确拒答仅能标注在不可回答且实际拒答的输出")
    if not result["refused"] and result["total_claims"] == 0:
        raise ValueError("非拒答输出必须审核至少一项声明")
    return result


def _normalize(record: object) -> dict:
    """校验并复制锁定用例和两模型的独立人工审核。"""
    if not isinstance(record, dict):
        raise ValueError("评估记录必须是字典")
    task = record.get("task_type")
    if task not in TASK_TYPES:
        raise ValueError("task_type必须为四类主任务之一")
    answerable = record.get("answerable")
    if type(answerable) is not bool or answerable != (task != "refusal"):
        raise ValueError("answerable必须为布尔且与任务类型一致")
    refusal = record.get("refusal_type")
    if task == "refusal" and refusal not in REFUSAL_TYPES:
        raise ValueError("拒答任务必须标注四类refusal_type之一")
    if task != "refusal" and refusal is not None:
        raise ValueError("可回答任务不能标注refusal_type")
    labels = record.get("evidence_labels")
    if not isinstance(labels, list) or labels != [f"S{i}" for i in range(1, len(labels) + 1)]:
        raise ValueError("evidence_labels必须是从S1起连续且不重复的编号列表")
    if answerable and not labels:
        raise ValueError("可回答用例必须存在编号证据")
    if task == "multi" and len(labels) < 2:
        raise ValueError("多片段整合任务必须包含至少两片证据")
    if refusal == "no_evidence" and labels:
        raise ValueError("no_evidence拒答必须使用空证据")
    if refusal in ("irrelevant", "missing_conditions") and not labels:
        raise ValueError("irrelevant和missing_conditions拒答至少需要一片证据")
    if refusal == "conflict" and len(labels) < 2:
        raise ValueError("conflict拒答至少需要两片可判定冲突的证据")
    case_hash = record.get("test_case_hash")
    if not isinstance(case_hash, str) or not _HEX_HASH.fullmatch(case_hash):
        raise ValueError("test_case_hash必须是完整锁定问题、证据及审核基准的64位小写SHA-256")
    result = {"id": _text(record.get("id"), "id"), "task_type": task,
              "refusal_type": refusal, "answerable": answerable,
              "evidence_labels": list(labels), "test_case_hash": case_hash}
    for model in MODEL_TYPES:
        output = record.get(model)
        if not isinstance(output, dict):
            raise ValueError(f"{model}必须包含text和人工review")
        text = _text(output.get("text"), f"{model}.text")
        result[model] = {"text": text, "review": _review(output.get("review"), answerable),
                         "output_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    if result["base"]["review"]["total_keypoints"] != result["finetuned"]["review"]["total_keypoints"]:
        raise ValueError("基座与微调必须使用相同预定关键点分母")
    benchmark = {name: result[name] for name in ("id", "task_type", "refusal_type", "answerable", "evidence_labels", "test_case_hash")}
    benchmark["total_keypoints"] = result["base"]["review"]["total_keypoints"]
    result["sample_hash"] = _hash(benchmark)
    return result


def _aggregate(samples: list[dict]) -> tuple[dict, dict]:
    """从逐条输出与审核计算六指标分子分母，保留不可计算的空分母。"""
    counts = {}
    metrics = {}
    for model in MODEL_TYPES:
        totals = {name: {"numerator": 0, "denominator": 0} for name in METRIC_NAMES}
        for sample in samples:
            output = sample[model]
            review = output["review"]
            references = ["S" + suffix for suffix in _CITATION.findall(output["text"])]
            valid_references = sum(label in sample["evidence_labels"] for label in references)
            totals["citation_validity"]["numerator"] += valid_references
            totals["citation_validity"]["denominator"] += len(references) or int(sample["answerable"])
            totals["evidence_support"]["numerator"] += review["supported_claims"] if valid_references else 0
            totals["evidence_support"]["denominator"] += review["total_claims"]
            totals["terminology_format"]["numerator"] += int(review["terminology_format"])
            totals["terminology_format"]["denominator"] += 1
            if sample["answerable"]:
                totals["keypoint_coverage"]["numerator"] += review["covered_keypoints"]
                totals["keypoint_coverage"]["denominator"] += review["total_keypoints"]
                totals["answerable_false_refusal"]["numerator"] += int(review["refused"])
                totals["answerable_false_refusal"]["denominator"] += 1
            else:
                totals["refusal_recall"]["numerator"] += int(review["refused"] and review["refusal_correct"])
                totals["refusal_recall"]["denominator"] += 1
        counts[model] = totals
        metrics[model] = {name: pair["numerator"] / pair["denominator"] if pair["denominator"] else None
                          for name, pair in totals.items()}
    return counts, metrics


def _group(samples: list[dict]) -> dict:
    """输出分组样本ID以及与总体相同定义的计数和比例。"""
    counts, metrics = _aggregate(samples)
    return {"sample_count": len(samples), "sample_ids": [item["id"] for item in samples],
            "counts": counts, "metrics": metrics}


def evaluate(records: list[dict]) -> dict:
    """生成可审计评估报告；格式错误抛错，质量未达标保留失败理由。"""
    if not isinstance(records, list):
        raise ValueError("records必须是评估记录列表")
    samples = [_normalize(record) for record in records]
    samples.sort(key=lambda item: item["id"])
    ids = [sample["id"] for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("评估样本ID不能重复")
    case_hashes = [sample["test_case_hash"] for sample in samples]
    if len(case_hashes) != len(set(case_hashes)):
        raise ValueError("锁定测试用例不能用不同ID重复计数")
    counts, metrics = _aggregate(samples)
    groups = {"task_type": {}, "refusal_type": {}}
    for task in TASK_TYPES:
        groups["task_type"][task] = _group([sample for sample in samples if sample["task_type"] == task])
    for kind in REFUSAL_TYPES:
        groups["refusal_type"][kind] = _group([sample for sample in samples if sample["refusal_type"] == kind])
    coverage = {"task_counts": {key: value["sample_count"] for key, value in groups["task_type"].items()},
                "refusal_counts": {key: value["sample_count"] for key, value in groups["refusal_type"].items()},
                "capabilities": {"terminology_format": len(samples),
                                 "faithfulness": sum(item["answerable"] for item in samples),
                                 "integration": groups["task_type"]["multi"]["sample_count"],
                                 "citation": groups["task_type"]["citation"]["sample_count"],
                                 "refusal": groups["task_type"]["refusal"]["sample_count"]}}
    failures = []
    for dimension in ("task_counts", "refusal_counts", "capabilities"):
        for name, count in coverage[dimension].items():
            if count == 0:
                failures.append(f"缺少覆盖：{dimension}.{name}")
    for model in MODEL_TYPES:
        for name in METRIC_NAMES:
            if counts[model][name]["denominator"] == 0:
                failures.append(f"指标分母为空：{model}.{name}")
    tuned, base = metrics["finetuned"], metrics["base"]
    exact = {model: {name: Fraction(pair["numerator"], pair["denominator"]) if pair["denominator"] else None
                     for name, pair in counts[model].items()} for model in MODEL_TYPES}
    tuned_exact, base_exact = exact["finetuned"], exact["base"]
    for name, minimum in MINIMUMS.items():
        if tuned_exact[name] is None or tuned_exact[name] < Fraction(str(minimum)):
            failures.append(f"微调指标未达门槛：{name}>={minimum}")
    if tuned_exact["answerable_false_refusal"] is None or tuned_exact["answerable_false_refusal"] > Fraction(1, 20):
        failures.append("微调指标未达门槛：answerable_false_refusal<=0.05")
    if tuned_exact["evidence_support"] is None or base_exact["evidence_support"] is None or tuned_exact["evidence_support"] < base_exact["evidence_support"]:
        failures.append("忠实性不得低于基座")
    improved = any(tuned_exact[name] is not None and base_exact[name] is not None and tuned_exact[name] > base_exact[name]
                   for name in ("citation_validity", "keypoint_coverage", "refusal_recall"))
    if not improved:
        failures.append("引用、关键点覆盖或拒答召回必须至少一项优于基座")
    differences = {name: tuned[name] - base[name] if tuned[name] is not None and base[name] is not None else None
                   for name in METRIC_NAMES}
    return {"schema_version": SCHEMA_VERSION, "template_version": TEMPLATE_VERSION,
            "test_set_hash": _hash([{"id": item["id"], "sample_hash": item["sample_hash"]} for item in samples]),
            "sample_count": len(samples), "samples": samples, "counts": counts, "metrics": metrics,
            "coverage": coverage, "groups": groups, "differences": differences,
            "failure_reasons": failures, "passed": not failures}


def validate_report(report: dict) -> None:
    """重建逐条评估并核对完整报告，拒绝伪造通过标志和派生指标。"""
    if not isinstance(report, dict) or report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("评估报告schema不受支持")
    expected = evaluate(report.get("samples"))
    # 比较规范JSON避免True与1在Python字典相等中混淆，亦拒绝NaN。
    try:
        same = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("评估报告必须为有限JSON数据") from error
    if not same:
        raise ValueError("评估报告与逐条审核重算结果不一致")
    if not expected["passed"]:
        raise ValueError("评估未通过：" + "；".join(expected["failure_reasons"]))
