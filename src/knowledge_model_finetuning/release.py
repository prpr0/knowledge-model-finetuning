"""构建及复核平台兼容发布清单，不复制模型、不生成权重或伪造训练证明。

build_release仅返回字典，由CLI使用不存在模式写release-manifest.json。字典包含
model_name/model_version/base_model/base_revision/artifacts/evaluation_report及
artifact_groups/quantized_evaluation_report/template_version；artifacts每项为
{path, sha256}。两个报告必须为目录内不同文件，并逐条重算同一评估门槛，且绑定
相同test_set_hash。verify_release重新构建并逐字段对比，覆盖缺失/多余摘要。

只接受用正斜杠表示的可移植相对文件路径，拒绝盘符、UNC、反斜杠、点路径、
Windows设备名/ADS及尾随点或空格；解析符号链接后仍须在root中。路径按大小写
不敏感去重，解析后的别名也不得重复。所有文件必须真实非空，权重分块SHA-256。
组内要求adapter的safetensors与adapter_config.json、merged_model的safetensors、
gguf的.gguf，以及tokenizer.json/tokenizer.model/vocab.json之一和真实聊天模板。
模板可为非空chat_template.jinja，或tokenizer_config.json中的非空chat_template。
这些结构及摘要检查不证明权重可推理、训练已发生或人工审核身份真实。
"""
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

from .evaluation import validate_report
from .prompt import BASE_MODEL, MODEL_NAME, TEMPLATE_VERSION

_GROUPS = ("adapter", "merged_model", "gguf", "tokenizer", "training_config", "data_manifest", "licenses")
_REVISION = re.compile(r"[0-9a-fA-F]{40}\Z")
_RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?\Z", re.IGNORECASE)


def _root_directory(root: Path) -> Path:
    """解析实际发布根目录，避免不存在的路径被当作有效发布目录。"""
    try:
        resolved = Path(root).resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as error:
        raise ValueError("发布根目录不存在或不可解析") from error
    if not resolved.is_dir():
        raise ValueError("发布根路径必须为目录")
    return resolved


def _confined(root: Path, name: object) -> Path:
    """跨平台拒绝危险路径，并复查符号链接的真实文件边界。"""
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("产物路径必须是使用正斜杠的相对文件路径")
    if PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive or PureWindowsPath(name).root:
        raise ValueError("产物路径不能为绝对路径、盘符或UNC")
    components = name.split("/")
    for part in components:
        if part in ("", ".", "..") or part.endswith((".", " ")) or any(ord(char) < 32 for char in part):
            raise ValueError("产物路径存在空段、点路径、尾随别名或控制字符")
        if any(char in part for char in '<>"|?*') or _RESERVED.fullmatch(part):
            raise ValueError("产物路径包含Windows保留字符或设备名")
    try:
        target = (root / name).resolve(strict=True)
        if not target.is_relative_to(root) or not target.is_file() or target.stat().st_size <= 0:
            raise ValueError("产物必须是真实非空文件且符号链接不能越界")
    except (OSError, RuntimeError) as error:
        raise ValueError(f"发布文件不可访问：{name}") from error
    return target


def _file_hash(path: Path) -> str:
    """分块计算SHA-256，不把大模型权重整体读入内存。"""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise ValueError("发布产物无法读取") from error
    return digest.hexdigest()


def _json_file(path: Path) -> dict:
    """读取JSON对象，统一报告损坏文件的可诊断异常。"""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"JSON文件不可读取：{path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON文件必须包含对象：{path.name}")
    return value


def _require_shapes(root: Path, groups: dict) -> None:
    """检查必要权重与分词器模板结构，避免单个不明bin冒充完整交付。"""
    adapter = [PurePosixPath(name) for name in groups["adapter"]]
    if not any(path.suffix == ".safetensors" for path in adapter) or not any(path.name == "adapter_config.json" for path in adapter):
        raise ValueError("adapter必须含safetensors权重及adapter_config.json")
    for name in groups["adapter"]:
        if PurePosixPath(name).name == "adapter_config.json":
            _json_file(_confined(root, name))
    if not any(PurePosixPath(name).suffix == ".safetensors" for name in groups["merged_model"]):
        raise ValueError("merged_model必须包含原精度safetensors权重")
    if not any(PurePosixPath(name).suffix == ".gguf" for name in groups["gguf"]):
        raise ValueError("gguf必须包含.gguf量化权重")
    tokenizer_paths = groups["tokenizer"]
    if not any(PurePosixPath(name).name in ("tokenizer.json", "tokenizer.model", "vocab.json") for name in tokenizer_paths):
        raise ValueError("tokenizer必须包含真实分词器文件")
    template = False
    for name in tokenizer_paths:
        filename = PurePosixPath(name).name
        if filename == "chat_template.jinja":
            try:
                template = bool(_confined(root, name).read_text(encoding="utf-8").strip()) or template
            except (OSError, UnicodeError) as error:
                raise ValueError("聊天模板必须为非空UTF-8文本") from error
        elif filename == "tokenizer_config.json":
            value = _json_file(_confined(root, name)).get("chat_template")
            if isinstance(value, str) and value.strip():
                template = True
            elif isinstance(value, list) and value and all(isinstance(item, dict) and isinstance(item.get("name"), str)
                    and item["name"].strip() and isinstance(item.get("template"), str) and item["template"].strip() for item in value):
                template = True
    if not template:
        raise ValueError("tokenizer必须包含真实非空聊天模板")


def build_release(root: Path, specification: dict) -> dict:
    """依据真实文件构建清单，强制原精度及量化报告通过同一锁定测试集。"""
    root = _root_directory(root)
    if not isinstance(specification, dict):
        raise ValueError("specification必须为字典")
    version = specification.get("model_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("model_version必须为非空字符串")
    revision = specification.get("base_revision")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise ValueError("base_revision必须为40位不可变十六进制revision")
    supplied_groups = specification.get("artifact_groups")
    if not isinstance(supplied_groups, dict) or set(supplied_groups) != set(_GROUPS):
        raise ValueError("artifact_groups必须精确包含七类固定产物")
    groups = {}
    names = []
    for group in _GROUPS:
        paths = supplied_groups[group]
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"产物分组不能为空：{group}")
        if not all(isinstance(name, str) for name in paths):
            raise ValueError("产物路径必须为字符串")
        groups[group] = sorted(paths)
        names.extend(groups[group])
    report_name = specification.get("evaluation_report")
    quantized_name = specification.get("quantized_evaluation_report")
    names.extend((report_name, quantized_name))
    paths = {}
    seen_names = set()
    seen_targets = set()
    for name in names:
        target = _confined(root, name)
        name_key = name.casefold()
        target_key = str(target).casefold()
        if name_key in seen_names or target_key in seen_targets:
            raise ValueError("发布路径重复，或大小写/符号链接别名指向同一文件")
        seen_names.add(name_key)
        seen_targets.add(target_key)
        paths[name] = target
    _require_shapes(root, groups)
    report = _json_file(paths[report_name])
    quantized_report = _json_file(paths[quantized_name])
    validate_report(report)
    validate_report(quantized_report)
    if report["test_set_hash"] != quantized_report["test_set_hash"]:
        raise ValueError("原精度和量化评估必须使用相同锁定问题、证据及审核基准")
    artifacts = [{"path": name, "sha256": _file_hash(paths[name])} for name in sorted(paths)]
    return {"model_name": MODEL_NAME, "model_version": version, "base_model": BASE_MODEL,
            "base_revision": revision.lower(), "template_version": TEMPLATE_VERSION,
            "artifacts": artifacts, "artifact_groups": groups, "evaluation_report": report_name,
            "quantized_evaluation_report": quantized_name}


def verify_release(manifest_path: Path) -> dict:
    """重新计算文件摘要、报告门槛及清单结构，返回验证后的平台兼容字典。"""
    manifest_path = Path(manifest_path)
    root = _root_directory(manifest_path.parent)
    manifest_file = _confined(root, manifest_path.name)
    manifest = _json_file(manifest_file)
    expected = build_release(root, manifest)
    try:
        matches = json.dumps(manifest, sort_keys=True, ensure_ascii=False, allow_nan=False) == json.dumps(expected, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("发布清单必须为有限JSON数据") from error
    if not matches:
        raise ValueError("发布清单字段、分组或文件摘要与真实产物不一致")
    return manifest
