"""正式数据集验证与原子导出。

消息文件每行仅含messages；manifest保留sample_id、line_number、split、
源文档完整溯源、证据、审核、batch_id、template_version、sample_hash及可选标签。
summary含split_counts、task_counts、refusal_counts、review_counts及模板版本。
命令规则逐项检查反引号代码（含代码块）必须见于证据，数字词元也须见于证据；
这是保守字面质检，无法证明语义正确，人工审核仍为必需。
调用方必须提供真实tokenizer的消息计数器；工具只能验证返回值，无法鉴别
回调内部是否使用真实tokenizer，不能把本模块测试中的计数回调当作真实计数。
"""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable

from .corpus import assign_splits, _normalize, _required, _timestamp
from .prompt import build_messages, TEMPLATE_VERSION

_SPLITS = ("train", "validation", "test")
_TASKS = ("single", "multi", "citation", "refusal")
_REFUSALS = ("no_evidence", "irrelevant", "conflict", "missing_conditions")


def _json(value):
    """生成可复现的UTF-8 JSON文本。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    """计算结构化样例的稳定哈希。"""
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _check_answer(answer, evidence):
    """保守核对答案命令与数字是否存在于给定证据。"""
    text = "\n".join(e["content"] for e in evidence)
    without_citations = re.sub(r"\[S\d+\]", "", answer)
    commands = re.findall(r"```[^\n`]*\n(.*?)```|`([^`\n]+)`", without_citations, re.S)
    for block, inline in commands:
        snippet = block or inline
        if snippet.strip() and snippet.strip() not in text:
            raise ValueError("答案代码或命令未见于证据")
    for number in re.findall(r"(?<![0-9])\d+(?:\.\d+)*(?:%?)(?![0-9])", without_citations):
        if not re.search(r"(?<![0-9])" + re.escape(number) + r"(?![0-9])", text):
            raise ValueError("答案数字未见于证据")


def export_dataset(samples: list[dict], documents: list[dict], output_dir: Path, token_count: Callable[[list[dict]], int], max_tokens: int = 4096) -> dict:
    """校验完整正式数据集后在新目录一次发布，失败清除暂存文件。"""
    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise ValueError("禁止覆盖已有输出目录")
    if type(max_tokens) is not int or not 0 < max_tokens <= 4096 or not callable(token_count):
        raise ValueError("必须提供真实tokenizer计数器及不超过4096的预算")
    assigned = assign_splits(documents)
    registered = {}
    for raw, computed in zip(documents, assigned):
        if raw.get("split") != computed["split"] or raw.get("group_id") != computed["group_id"]:
            raise ValueError("文档分区或连通组与seed42计算不符")
        registered[raw["id"]] = computed
    counts = {s: 0 for s in _SPLITS}
    tasks = {s: {k: 0 for k in _TASKS} for s in _SPLITS}
    refusals = {s: {k: 0 for k in _REFUSALS} for s in _SPLITS}
    reviews = {s: {k: 0 for k in (*_TASKS, *_REFUSALS)} for s in _SPLITS}
    seen_ids, seen_questions, seen_pairs = set(), set(), set()
    rows = {s: [] for s in _SPLITS}
    manifest = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("样例必须为对象")
        for key in ("id", "question", "answer", "batch_id"):
            _required(sample.get(key), key)
        sid, split, task = sample["id"], sample.get("split"), sample.get("task_type")
        if sid in seen_ids or split not in _SPLITS or task not in _TASKS:
            raise ValueError("样例ID重复或分区/任务无效")
        question = _normalize(sample["question"])
        pair = (question, _normalize(sample["answer"]))
        if question in seen_questions or pair in seen_pairs:
            raise ValueError("重复问题或问答污染")
        seen_ids.add(sid)
        seen_questions.add(question)
        seen_pairs.add(pair)
        refusal = sample.get("refusal_type")
        if task == "refusal" and refusal not in _REFUSALS:
            raise ValueError("拒答子类无效")
        if task != "refusal" and refusal is not None:
            raise ValueError("可回答任务不能包含拒答子类")
        evidence = sample.get("evidence")
        if not isinstance(evidence, list) or (task != "refusal" and not evidence) or (task == "multi" and len(evidence) < 2):
            raise ValueError("证据数量不满足任务要求")
        if task == "refusal":
            if refusal == "no_evidence" and evidence:
                raise ValueError("无证据拒答不得携带证据")
            if refusal in ("irrelevant", "missing_conditions") and not evidence:
                raise ValueError("无关或缺条件拒答必须提供待判断的资料")
            if refusal == "conflict" and len(evidence) < 2:
                raise ValueError("冲突拒答至少需要两个证据片段")
        sources = []
        for item in evidence:
            if not isinstance(item, dict):
                raise ValueError("证据必须为对象")
            for key in ("document_id", "title", "version", "location", "content"):
                _required(item.get(key), key)
            doc = registered.get(item["document_id"])
            if not doc or doc["split"] != split or item["version"] != doc["version"]:
                raise ValueError("证据源不存在、版本不符或跨分区")
            if item["location"] not in doc["section"] and doc["section"] not in item["location"]:
                raise ValueError("证据出处无法追溯登记章节")
            if "original_content" in item or "transformation" in item:
                if task != "refusal" or refusal != "conflict":
                    raise ValueError("仅冲突拒答允许显式修改证据")
                _required(item.get("original_content"), "original_content")
                _required(item.get("transformation"), "transformation")
                if item["original_content"] not in doc["text"] or "演示" not in item["transformation"] or "演示" not in item["content"]:
                    raise ValueError("冲突修改必须追溯原文并在模型可见正文标注演示")
            elif item["content"] not in doc["text"]:
                raise ValueError("证据内容不属于登记原文")
            sources.append(doc)
        citations = re.findall(r"\[S(\d+)\]", sample["answer"])
        if (task != "refusal" and not citations) or any(int(n) < 1 or int(n) > len(evidence) or str(int(n)) != n for n in citations):
            raise ValueError("答案引用缺失或无效")
        if re.search(r"\[S[^\]]*\]", re.sub(r"\[S\d+\]", "", sample["answer"])):
            raise ValueError("答案引用格式无效")
        _check_answer(sample["answer"], evidence)
        review = sample.get("review")
        if not isinstance(review, dict) or review.get("status") not in ("pending", "approved", "rejected"):
            raise ValueError("审核状态无效")
        if review["status"] == "rejected":
            raise ValueError(f"生成批次{sample['batch_id']}存在拒绝样例，须整体修订重验")
        approved = review["status"] == "approved"
        if approved:
            _required(review.get("reviewer"), "reviewer")
            _timestamp(review.get("reviewed_at"))
        if split != "train" and not approved:
            raise ValueError("验证和测试集必须全量人工批准")
        messages = build_messages(sample["question"], evidence, sample["answer"])
        tokens = token_count(messages)
        if type(tokens) is not int or tokens <= 0 or tokens > max_tokens:
            raise ValueError("真实token计数无效或超出预算")
        counts[split] += 1
        tasks[split][task] += 1
        if task == "refusal":
            refusals[split][refusal] += 1
        if approved:
            reviews[split][task] += 1
            if task == "refusal":
                reviews[split][refusal] += 1
        rows[split].append({"messages": messages})
        manifest.append(dict(sample_id=sid, line_number=len(rows[split]), split=split, task_type=task, refusal_type=refusal, batch_id=sample["batch_id"], review=review, sources=sources, evidence=evidence, template_version=TEMPLATE_VERSION, sample_hash=_hash(sample), messages_hash=_hash({"messages": messages}), token_count=tokens, labels={k: v for k, v in sample.items() if k not in {"id", "split", "task_type", "refusal_type", "question", "evidence", "answer", "batch_id", "review"}}))
    for split in _SPLITS:
        for kind, total in (tasks[split] | refusals[split]).items():
            if not total:
                raise ValueError(f"{split}缺少{kind}覆盖")
            required = math.ceil(total / 10) if split == "train" else total
            if reviews[split][kind] < required:
                raise ValueError(f"{split}的{kind}人工审核覆盖不足")
    summary = dict(schema_version=1, template_version=TEMPLATE_VERSION, split_counts=counts, task_counts=tasks, refusal_counts=refusals, review_counts=reviews, max_tokens=max_tokens, quality_limits="命令和数字仅做字面证据核对，不能证明语义正确；tokenizer真实性由调用方保证", manifest_hash=_hash(manifest))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        for split, values in rows.items():
            (staging / f"{split}.jsonl").write_text("".join(_json(v) + "\n" for v in values), encoding="utf-8")
        (staging / "manifest.jsonl").write_text("".join(_json(v) + "\n" for v in manifest), encoding="utf-8")
        (staging / "summary.json").write_text(_json(summary) + "\n", encoding="utf-8")
        # 完整暂存目录以单次重命名发布，不暴露部分数据文件。
        if target.exists() or target.is_symlink():
            raise ValueError("禁止覆盖已有输出目录")
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return summary
