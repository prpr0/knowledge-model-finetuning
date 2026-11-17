"""从锁定官方revision分段续传权重，复用已有连续前缀并核对官方LFS摘要。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen


def sha256(path):
    """分块计算文件摘要，避免整份权重进入内存。"""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fetch_part(task):
    """有限重取官方范围，避免缓存过期重定向，并校验真实字节数。"""
    url, start, end, destination, ranged = task
    expected = end - start + 1
    if destination.exists() and destination.stat().st_size == expected:
        return
    temporary = destination.with_suffix(".tmp")
    started = time.monotonic()
    for attempt in range(3):
        request = Request(url + f"?download=true&range_start={start}&range_end={end}&attempt={attempt}",
                          headers={"Range": f"bytes={start}-{end}"} if ranged else {})
        try:
            with urlopen(request, timeout=180) as response, temporary.open("wb") as stream:
                content_range = response.headers.get("Content-Range", "")
                if ranged and (response.status != 206 or not content_range.startswith(f"bytes {start}-{end}/")):
                    raise RuntimeError(f"官方响应范围不符：{content_range}")
                shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
            if temporary.stat().st_size != expected:
                raise RuntimeError("下载分段字节数不符")
            temporary.replace(destination)
            print(json.dumps({"part": destination.name, "bytes": expected, "seconds": time.monotonic() - started}), flush=True)
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    """并行下载剩余范围并按序合并，最终摘要不符则拒绝发布权重文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    if metadata["id"] != "Qwen/Qwen3-4B-Instruct-2507" or metadata["sha"] != "cdbee75f17c01a7cc42f958dc650907174af0554":
        raise ValueError("官方元数据与锁定基座不符")
    root = Path(args.output_dir)
    parts_root = root / "download-parts"
    parts_root.mkdir(exist_ok=True)
    tasks, assemblies = [], []
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if name == ".gitattributes":
            continue
        target = root / name
        size = item["size"]
        expected_hash = item.get("lfs", {}).get("sha256")
        url = f"https://huggingface.co/{metadata['id']}/resolve/{metadata['sha']}/{name}"
        if target.exists():
            if target.stat().st_size != size or (expected_hash and sha256(target) != expected_hash):
                raise ValueError(f"已有文件与官方摘要不符：{name}")
            continue
        prefix = root / (name + ".prefix")
        old_partial = root / (name + ".partial")
        if old_partial.exists():
            if prefix.exists():
                raise ValueError("同时存在两份前缀，无法判断连续性")
            old_partial.rename(prefix)
        offset = prefix.stat().st_size if prefix.exists() else 0
        if offset > size:
            raise ValueError("已有连续前缀超过官方文件大小")
        pieces = []
        part_size = 8 * 1024 * 1024 if expected_hash else size
        for start in range(offset, size, part_size):
            end = min(start + part_size, size) - 1
            part = parts_root / f"{name}.{start}-{end}.part"
            tasks.append((url, start, end, part, bool(expected_hash)))
            pieces.append(part)
        assemblies.append((item, prefix, pieces))
    print(json.dumps({"pending_parts": len(tasks), "remaining_bytes": sum(end - start + 1 for _, start, end, _, _ in tasks), "workers": args.workers}), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(fetch_part, tasks))
    for item, prefix, pieces in assemblies:
        target = root / item["rfilename"]
        temporary = target.with_name(target.name + ".assembling")
        with temporary.open("wb") as output:
            for source in ([prefix] if prefix.exists() else []) + pieces:
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
        actual = sha256(temporary)
        if temporary.stat().st_size != item["size"] or (item.get("lfs", {}).get("sha256") and actual != item["lfs"]["sha256"]):
            raise ValueError(f"完整文件不符合官方摘要：{item['rfilename']}")
        temporary.rename(target)
        for source in ([prefix] if prefix.exists() else []) + pieces:
            source.unlink()
    records = []
    for item in metadata["siblings"]:
        if item["rfilename"] == ".gitattributes":
            continue
        path = root / item["rfilename"]
        actual = sha256(path)
        expected = item.get("lfs", {}).get("sha256")
        if path.stat().st_size != item["size"] or (expected and actual != expected):
            raise ValueError(f"最终核验失败：{path.name}")
        records.append({"file": path.name, "bytes": path.stat().st_size, "sha256": actual, "official_lfs_sha256": expected, "source_url": f"https://huggingface.co/{metadata['id']}/resolve/{metadata['sha']}/{path.name}"})
    report = {"model": metadata["id"], "revision": metadata["sha"], "source": "official Hugging Face", "all_sizes_verified": True, "all_official_lfs_sha256_verified": True, "files": records}
    (root / "download-manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.rmtree(parts_root)
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
