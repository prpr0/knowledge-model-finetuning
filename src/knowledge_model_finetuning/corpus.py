"""官方语料验证及稳定文档分区。

先校验原始 UTF-8 哈希，再以 NFKC、空白折叠及大小写折叠生成比较文本。
近重复使用字符三元组集合 Jaccard>=0.85；同源 URL、规范化全文相同或
近重复的文档通过并查集合为连通组。算法为全对比较，适用于离线资料整理，
不是语义去重证明。组按固定种子散列排序后按组数近似80/10/10划分，
每个分区至少一组，组大小不均时文档比例不保证恰为80/10/10。
"""
import hashlib
import math
import re
import unicodedata
from datetime import datetime
from urllib.parse import urlsplit, unquote


def _required(value, name):
    """读取必需的非空字符串。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}必须为非空字符串")
    return value


def _timestamp(value):
    """验证带时区的采集或审核时间。"""
    _required(value, "时间")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("时间必须为ISO8601格式") from exc
    if parsed.tzinfo is None:
        raise ValueError("时间必须包含时区")


def validate_document(document: dict) -> dict:
    """校验官方来源、许可、版本和原文哈希并保留全部溯源字段。"""
    if not isinstance(document, dict):
        raise ValueError("文档必须为对象")
    for key in ("id", "source_url", "fetched_at", "version", "section", "license", "attribution", "transformation", "text", "sha256"):
        _required(document.get(key), key)
    _timestamp(document["fetched_at"])
    try:
        url = urlsplit(document["source_url"])
        port = url.port
    except ValueError as exc:
        raise ValueError("来源URL无效") from exc
    path = unquote(url.path)
    if url.scheme != "https" or url.username or url.password or port is not None or url.query or url.fragment or "\\" in path or any(p in (".", "..") for p in path.split("/")):
        raise ValueError("来源URL不符合官方技术页约束")
    python = url.hostname == "docs.python.org" and path.startswith("/zh-cn/3.11/") and path.endswith(".html") and path not in ("/zh-cn/3.11/license.html", "/zh-cn/3.11/copyright.html")
    kubernetes = url.hostname == "kubernetes.io" and path.startswith("/zh-cn/docs/") and path.rstrip("/") != "/zh-cn/docs"
    if not (python or kubernetes):
        raise ValueError("仅允许登记的官方中文技术文档")
    expected = "PSF-2.0" if python else "CC-BY-4.0"
    if document["license"] != expected:
        raise ValueError("来源许可标记不匹配")
    version = document["version"].strip()
    if python and not re.fullmatch(r"3\.11(?:\.\d+)?", version):
        raise ValueError("Python版本必须为3.11系列")
    if kubernetes and not re.fullmatch(r"v?1\.\d+(?:\.\d+)?", version):
        raise ValueError("Kubernetes必须登记明确发布版本")
    if hashlib.sha256(document["text"].encode("utf-8")).hexdigest() != document["sha256"]:
        raise ValueError("原始正文SHA-256不匹配")
    return dict(document)


def _normalize(text):
    """生成仅用于去重比较的规范化文本，不修改原文。"""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def assign_splits(documents: list[dict], seed: int = 42) -> list[dict]:
    """按近重复连通组稳定分区，保持返回列表的输入顺序。"""
    if type(seed) is not int:
        raise ValueError("seed必须为整数")
    result = [validate_document(d) for d in documents]
    if len({d["id"] for d in result}) != len(result):
        raise ValueError("文档ID重复")
    parents = list(range(len(result)))
    normalized = [_normalize(d["text"]) for d in result]
    grams = [{text[i:i + 3] for i in range(max(1, len(text) - 2))} for text in normalized]

    def find(index):
        """查找并压缩连通组根节点。"""
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for i, doc in enumerate(result):
        for j in range(i):
            union = grams[i] | grams[j]
            similar = len(grams[i] & grams[j]) / len(union) if union else 1
            if doc["source_url"].rstrip("/") == result[j]["source_url"].rstrip("/") or normalized[i] == normalized[j] or similar >= 0.85:
                parents[find(i)] = find(j)
    groups = {}
    for i in range(len(result)):
        groups.setdefault(find(i), []).append(i)
    if len(groups) < 3:
        raise ValueError("至少需要三个独立文档组")
    identified = []
    for members in groups.values():
        identity = "\n".join(sorted(result[i]["id"] for i in members))
        group_id = hashlib.sha256(identity.encode()).hexdigest()
        order = hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()
        identified.append((order, group_id, members))
    identified.sort()
    held = max(1, math.floor(len(identified) * 0.1))
    train_end = len(identified) - held * 2
    for index, (_, group_id, members) in enumerate(identified):
        split = "train" if index < train_end else "validation" if index < train_end + held else "test"
        for member in members:
            result[member].update(group_id=group_id, split=split)
    return result
