"""提供离线分区、正式数据导出、评估和发布校验的命令入口。"""
import argparse
import json
import logging
import logging.config
import os
from pathlib import Path
import sys
from uuid import uuid4

from .corpus import assign_splits
from .dataset import export_dataset
from .evaluation import evaluate
from .release import build_release, verify_release


def _read_jsonl(path: Path) -> list[dict]:
    """逐行读取对象并定位格式错误，不接受空行掩盖行号错位。"""
    values = []
    with path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"JSONL第{number}行为空")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSONL第{number}行格式错误") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL第{number}行必须为对象")
            values.append(value)
    return values


def _write_output(path: Path, text: str) -> None:
    """以独占方式写入完整序列化结果，拒绝覆盖用户文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as target:
        target.write(text)


def _json_text(value: dict) -> str:
    """生成可复核的标准JSON，不序列化NaN或无穷大。"""
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _token_counter(path: Path):
    """只加载现有本地分词器，按原生聊天模板统计完整训练序列。"""
    if not path.is_dir():
        raise ValueError("本地分词器目录不存在；命令不会下载模型")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ValueError("当前解释器未安装Transformers；请使用已获准配置的本地环境") from error
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("本地分词器缺少模型原生聊天模板")

    def count(messages: list[dict]) -> int:
        """读取真实input_ids长度，避免把BatchEncoding字段数当token数。"""
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=True)
        ids = encoded["input_ids"]
        if not isinstance(ids, list) or any(type(value) is not int for value in ids):
            raise ValueError("分词器未返回单条完整序列的token ID列表")
        return len(ids)

    return count


def parser() -> argparse.ArgumentParser:
    """声明实际支持的离线操作及显式文件参数。"""
    result = argparse.ArgumentParser(description="知识库微调离线数据与发布工具；不下载资源或执行训练")
    commands = result.add_subparsers(dest="command", required=True)
    partition = commands.add_parser("partition", help="校验官方资料并按近重复组分区")
    partition.add_argument("documents", type=Path)
    partition.add_argument("output", type=Path)
    export = commands.add_parser("export", help="校验审核记录并导出标准messages与manifest")
    export.add_argument("samples", type=Path)
    export.add_argument("documents", type=Path)
    export.add_argument("output_dir", type=Path)
    export.add_argument("--tokenizer", type=Path, required=True, help="已存在的Qwen3本地分词器目录，仅离线加载")
    evaluation = commands.add_parser("evaluate", help="根据人工审核计数和真实输出生成对照报告")
    evaluation.add_argument("records", type=Path)
    evaluation.add_argument("output", type=Path)
    release = commands.add_parser("release", help="核验现有产物并生成带摘要的发布清单")
    release.add_argument("root", type=Path)
    release.add_argument("specification", type=Path)
    verify = commands.add_parser("verify-release", help="重新校验发布文件和两份评估报告")
    verify.add_argument("manifest", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    """执行一个操作，流程日志含run_id，失败返回非零且不输出成功状态。"""
    arguments = parser().parse_args(argv)
    logging.config.fileConfig(Path(__file__).resolve().parents[2] / "configs" / "logging.ini", disable_existing_loggers=False)
    logger = logging.LoggerAdapter(logging.getLogger("knowledge_finetuning"), {"run_id": uuid4().hex})
    logger.info("流程开始 command=%s", arguments.command)
    try:
        if arguments.command == "partition":
            if arguments.output.exists():
                raise FileExistsError("输出文件已存在")
            documents = assign_splits(_read_jsonl(arguments.documents))
            serialized = "".join(json.dumps(document, ensure_ascii=False, allow_nan=False) + "\n" for document in documents)
            _write_output(arguments.output, serialized)
            summary = {"documents": len(documents), "split_counts": {split: sum(document["split"] == split for document in documents) for split in ("train", "validation", "test")}}
        elif arguments.command == "export":
            counter = _token_counter(arguments.tokenizer)
            summary = export_dataset(_read_jsonl(arguments.samples), _read_jsonl(arguments.documents), arguments.output_dir, counter)
        elif arguments.command == "evaluate":
            if arguments.output.exists():
                raise FileExistsError("输出文件已存在")
            report = evaluate(_read_jsonl(arguments.records))
            _write_output(arguments.output, _json_text(report))
            if not report.get("passed"):
                logger.error("评估未通过；已保留失败报告，不可发布")
                print(_json_text({"passed": False, "report": str(arguments.output)}), end="")
                return 2
            summary = {"passed": True, "report": str(arguments.output)}
        elif arguments.command == "release":
            target = arguments.root / "release-manifest.json"
            if target.exists():
                raise FileExistsError("发布清单已存在；请使用新的版本目录")
            specification = json.loads(arguments.specification.read_text(encoding="utf-8-sig"))
            manifest = build_release(arguments.root, specification)
            _write_output(target, _json_text(manifest))
            summary = {"model_name": manifest["model_name"], "model_version": manifest["model_version"], "manifest": str(target)}
        else:
            manifest = verify_release(arguments.manifest)
            summary = {"model_name": manifest["model_name"], "model_version": manifest["model_version"], "verified": True}
        logger.info("流程成功 command=%s", arguments.command)
        print(_json_text(summary), end="")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        logger.error("流程失败 command=%s reason=%s", arguments.command, error)
        return 2
    except Exception:
        logger.exception("流程异常 command=%s", arguments.command)
        return 1
