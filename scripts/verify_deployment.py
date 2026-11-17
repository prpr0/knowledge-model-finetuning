"""实际核验Swift普通及SSE接口；基线释放GPU，最终模型可保留已验证服务。"""
from __future__ import annotations

import argparse
import json
import logging.config
import os
import signal
import socket
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from knowledge_model_finetuning.prompt import build_messages


def port_is_open(port):
    """检查本机回环端口是否被服务占用，避免借用已有模型验证。"""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def stop_server_group(process, port):
    """回收启动器及全部自有服务子进程，确认端口释放后才报告停止。"""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)
    deadline = time.monotonic() + 5
    while port_is_open(port):
        if time.monotonic() >= deadline:
            raise RuntimeError("自有模型服务的端口未释放，不能报告停止成功")
        time.sleep(0.05)


def request_json(url, payload=None):
    """发起真实HTTP请求，保留服务原始JSON响应。"""
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def main():
    """实测模型服务，失败始终回收进程；可显式保留成功验证的最终部署。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--keep-running", action="store_true", help="成功验证后保留服务供用户访问")
    args = parser.parse_args()
    logging.config.fileConfig(ROOT / "configs/logging.ini", disable_existing_loggers=False)
    logger = logging.LoggerAdapter(logging.getLogger("knowledge_finetuning"), {"run_id": "deployment-check"})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with open(args.documents, encoding="utf-8") as source:
        document = json.loads(next(source))
    evidence = [{"title": document["title"], "version": document["version"], "location": document["section"].replace("\n", " "), "content": document["text"][:800]}]
    messages = build_messages("仅根据S1，用一句话说明资料主题并标注引用。", evidence)
    command = [str(Path(sys.executable).parent / "swift"), "deploy", "--model", args.model,
               "--model_type", "qwen3_nothinking", "--template", "qwen3_nothinking",
               "--template_backend", "jinja", "--infer_backend", "pt", "--torch_dtype", "bfloat16",
               "--attn_impl", "sdpa", "--max_length", "4096", "--max_new_tokens", "1024",
               "--temperature", "0", "--max_batch_size", "1", "--host", "127.0.0.1",
               "--port", str(args.port), "--served_model_name", args.model_name, "--verbose", "false"]
    report = {"command": command, "human_review": "pending", "release_approved": False, "model": args.model_name}
    address = f"http://127.0.0.1:{args.port}"
    if port_is_open(args.port):
        raise RuntimeError(f"端口{args.port}已被占用，拒绝借用或终止未知服务")
    with (output / "server.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
        report["server_process_group"] = process.pid
        try:
            logger.info("开始部署验证 model=%s pid=%s", args.model_name, process.pid)
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"服务提前退出，退出码={process.returncode}；见server.log")
                try:
                    report["models"] = request_json(address + "/v1/models")
                    model_ids = {item["id"] for item in report["models"].get("data", [])}
                    if args.model_name not in model_ids:
                        raise RuntimeError(f"服务模型名称不匹配：{sorted(model_ids)}")
                    break
                except (URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("服务未在180秒内就绪")
                    time.sleep(1)
            payload = {"model": args.model_name, "messages": messages, "temperature": 0, "max_tokens": 128, "stream": False}
            report["request"] = payload
            started = time.perf_counter()
            regular = request_json(address + "/v1/chat/completions", payload)
            report["nonstream_seconds"] = time.perf_counter() - started
            if not regular.get("choices", [{}])[0].get("message", {}).get("content") or not regular.get("usage", {}).get("completion_tokens", 0):
                raise RuntimeError("普通响应缺少真实回答或usage")
            report["nonstream"] = regular
            streaming_payload = dict(payload, stream=True, stream_options={"include_usage": True})
            request = Request(address + "/v1/chat/completions", data=json.dumps(streaming_payload, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json"})
            started = time.perf_counter()
            chunks, content, first_token, done, usage = [], [], None, False, None
            with urlopen(request, timeout=120) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        done = True
                        break
                    chunk = json.loads(line[6:])
                    chunks.append(chunk)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        text = choice.get("delta", {}).get("content")
                        if text:
                            if first_token is None:
                                first_token = time.perf_counter() - started
                            content.append(text)
            if not done or not content or not usage or not usage.get("completion_tokens", 0):
                raise RuntimeError("SSE缺少正文、结束标记或真实usage")
            report.update(state="completed", stream_chunks=chunks, stream_text="".join(content), stream_usage=usage,
                          stream_first_token_seconds=first_token, stream_total_seconds=time.perf_counter() - started)
            logger.info("部署验证通过 model=%s", args.model_name)
        except BaseException as error:
            report.update(state="failed", error_type=type(error).__name__, error=str(error))
            logger.exception("部署验证失败 model=%s", args.model_name)
            raise
        finally:
            if args.keep_running and report.get("state") == "completed":
                report.update(server_stopped=False, server_pid=process.pid, base_url=address + "/v1")
            else:
                stop_server_group(process, args.port)
                report["server_stopped"] = True
            (output / "deployment-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
