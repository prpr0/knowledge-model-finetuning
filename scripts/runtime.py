"""运行真实本地权重推理、QLoRA冒烟及原精度合并；实验产物始终标记未经人工审核。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import logging.config
import math
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LOGGER = logging.getLogger("knowledge_finetuning")


def canonical(value):
    """编码稳定JSON，供请求、消息和配置的真实哈希使用。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    """计算规范JSON的SHA-256。"""
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    """分块读取文件，避免把模型权重整体复制到内存。"""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def utc_now():
    """记录带时区的真实时间。"""
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value, exclusive=False):
    """写入并同步运行记录，不覆盖要求独立保存的已有产物。"""
    with Path(path).open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path):
    """严格读取JSONL，拒绝空行和非对象记录，保留原始行号。"""
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{number}包含空行")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}必须是JSON对象")
            yield number, value


def checked_messages(value, training=False):
    """校验单轮system/user及可选assistant，不修改或截断内容。"""
    roles = ["system", "user", "assistant"] if training else ["system", "user"]
    if not isinstance(value, list) or len(value) != len(roles):
        raise ValueError(f"messages必须依次包含{roles}")
    for message, role in zip(value, roles):
        if not isinstance(message, dict) or message.get("role") != role:
            raise ValueError(f"messages角色必须依次为{roles}")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("消息正文必须为非空文本")
    return value


def fingerprint(path):
    """对实际加载的配置、分词器与权重逐文件计算SHA-256并锁定清单。"""
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"模型或adapter必须是本地目录：{root}")
    files = []
    for item in sorted(root.iterdir()):
        if item.name in {"merge-report.json", "smoke-report.json"}:
            continue
        if item.is_file() and (item.suffix in {".json", ".safetensors", ".bin", ".model", ".tiktoken"}
                               or item.name in {"merges.txt", "vocab.txt", "chat_template.jinja"}):
            files.append({"name": item.name, "bytes": item.stat().st_size, "sha256": file_hash(item)})
    if not any(item["name"].endswith((".safetensors", ".bin")) for item in files):
        raise ValueError(f"目录没有可加载的本地模型权重：{root}")
    return {"path": str(root), "files": files, "sha256": digest(files)}


def versions():
    """记录实际安装版本，不把规划版本写成运行事实。"""
    result = {"python": sys.version}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "ms-swift"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def environment(torch):
    """确认真实CUDA与BF16能力，固定本轮随机种子。"""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("运行要求支持BF16的真实CUDA GPU")
    from transformers import set_seed
    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    return {"versions": versions(), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "seed": 42, "base_revision_contract": BASE_REVISION,
            "experiment_mode": "automated_learning_experiment", "human_reviewed": False,
            "publication_approved": False}


def load_tokenizer(path):
    """只从本地加载原生聊天模板，并启用左侧padding。"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("本地分词器缺少原生chat_template")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("分词器同时缺少pad_token和eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prompt_ids(tokenizer, messages):
    """使用原生模板计算完整输入，超预算直接报错而不截断。"""
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not isinstance(ids, list) or not ids or any(type(token) is not int for token in ids):
        raise ValueError("原生模板没有返回完整token ID列表")
    if len(ids) > 3072:
        raise ValueError(f"输入有{len(ids)} tokens，超过3072；禁止截断")
    return ids


def load_base(path, torch, quantized=False, device="cuda"):
    """以本地BF16和SDPA加载基座，可选真实NF4双重量化。"""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    options = {"local_files_only": True, "trust_remote_code": False, "torch_dtype": torch.bfloat16,
               "attn_implementation": "sdpa", "device_map": {"": device}}
    if quantized:
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    return AutoModelForCausalLM.from_pretrained(str(path), **options)


def read_requests(path):
    """校验全部请求ID唯一；每条记录单独锁定完整请求哈希。"""
    records = []
    seen = set()
    for number, row in read_jsonl(path):
        identity = row.get("id")
        if not isinstance(identity, str) or not identity.strip() or identity in seen:
            raise ValueError(f"请求第{number}行id缺失或重复")
        checked_messages(row.get("messages"))
        seen.add(identity)
        records.append({"request": row, "request_hash": digest(row)})
    if not records:
        raise ValueError("请求文件为空")
    return records


def existing_outputs(path, records, run_hash):
    """续跑时逐条验证请求、配置和输出哈希，拒绝损坏或串用结果。"""
    expected = {item["request"]["id"]: item for item in records}
    done = set()
    for number, row in read_jsonl(path):
        identity = row.get("id")
        if identity not in expected or identity in done:
            raise ValueError(f"已有结果第{number}行id未知或重复")
        if row.get("request_hash") != expected[identity]["request_hash"] or row.get("run_hash") != run_hash:
            raise ValueError(f"已有结果第{number}行request hash或运行配置不匹配")
        request = expected[identity]["request"]
        if row.get("test_hash") != request.get("test_hash"):
            raise ValueError(f"已有结果第{number}行test_hash与请求不匹配")
        text = row.get("text")
        if not isinstance(text, str) or row.get("output_hash") != hashlib.sha256(text.encode("utf-8")).hexdigest():
            raise ValueError(f"已有结果第{number}行输出哈希不匹配")
        if (type(row.get("prompt_tokens")) is not int or row["prompt_tokens"] <= 0
                or type(row.get("generated_tokens")) is not int or row["generated_tokens"] <= 0
                or not isinstance(row.get("elapsed_seconds"), (int, float))
                or not math.isfinite(row["elapsed_seconds"]) or row["elapsed_seconds"] < 0):
            raise ValueError(f"已有结果第{number}行计数或耗时不合法")
        done.add(identity)
    return done


def generate(args, logger):
    """批量贪心推理并逐行flush，显式续跑绝不漏ID或静默截断。"""
    import torch
    records = read_requests(args.input)
    output = args.output
    metadata_path = Path(str(output) + ".meta.json")
    if not args.resume and (output.exists() or metadata_path.exists()):
        raise FileExistsError("输出或元数据已存在；只有显式--resume且哈希一致才可续跑")
    if args.resume and (not output.is_file() or not metadata_path.is_file()):
        raise FileNotFoundError("--resume要求结果JSONL和.meta.json同时存在")
    info = environment(torch)
    config = {"model": fingerprint(args.model), "adapter": fingerprint(args.adapter) if args.adapter else None,
              "input_sha256": file_hash(args.input), "requests_hash": digest([r["request_hash"] for r in records]),
              "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens, "seed": 42,
              "dtype": "bfloat16", "attention": "sdpa", "decoding": "greedy", "num_beams": 1,
              "padding_side": "left", "input_budget": 3072, "sequence_budget": 4096,
              "trust_remote_code": False, "local_files_only": True,
              "versions": info["versions"]}
    run_hash = digest(config)
    if args.resume:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("run_hash") != run_hash or metadata.get("config") != config:
            raise ValueError("续跑输入、模型、adapter、版本或生成配置发生变化")
        done = existing_outputs(output, records, run_hash)
    else:
        metadata = dict(info, config=config, run_hash=run_hash, started_at=utc_now(), state="running",
                        expected_count=len(records), timing_scope="CUDA同步后的整批生成耗时，每行共享该批耗时")
        done = set()
    tokenizer = load_tokenizer(args.model)
    pending = []
    for item in records:
        ids = prompt_ids(tokenizer, item["request"]["messages"])
        if len(ids) + args.max_new_tokens > 4096:
            raise ValueError(f"请求{item['request']['id']}输入加生成预算超过4096；禁止截断")
        if item["request"]["id"] not in done:
            pending.append(dict(item, input_ids=ids))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        save_json(metadata_path, metadata, exclusive=True)
    if not pending:
        metadata.update(state="completed", completed_count=len(done), finished_at=utc_now(), output_sha256=file_hash(output))
        save_json(metadata_path, metadata)
        logger.info("推理结果已完整 count=%d", len(done))
        return
    model = load_base(args.model, torch)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter), local_files_only=True, is_trainable=False)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    try:
        with output.open("a" if args.resume else "x", encoding="utf-8", newline="\n") as stream:
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset:offset + args.batch_size]
                encoded = tokenizer.pad({"input_ids": [item["input_ids"] for item in batch]},
                                        padding=True, return_attention_mask=True, return_tensors="pt")
                encoded = {key: tensor.to("cuda") for key, tensor in encoded.items()}
                width = encoded["input_ids"].shape[1]
                if width + args.max_new_tokens > 4096:
                    raise ValueError("左padding后的批次超过4096序列预算")
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    outputs = model.generate(**encoded, max_new_tokens=args.max_new_tokens,
                                             do_sample=False, num_beams=1, repetition_penalty=1.0,
                                             pad_token_id=tokenizer.pad_token_id, use_cache=True)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                generated = outputs[:, width:].tolist()
                for item, tokens in zip(batch, generated):
                    for position, token in enumerate(tokens):
                        if token in eos_ids:
                            tokens = tokens[:position + 1]
                            break
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                    row = {"id": item["request"]["id"], "text": text,
                           "prompt_tokens": len(item["input_ids"]), "generated_tokens": len(tokens),
                           "elapsed_seconds": elapsed, "batch_elapsed_seconds": elapsed, "batch_size": len(batch),
                           "request_hash": item["request_hash"], "run_hash": run_hash,
                           "prompt_token_ids_hash": digest(item["input_ids"]), "generated_token_ids_hash": digest(tokens),
                           "output_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                           "finish_reason": "eos" if tokens and tokens[-1] in eos_ids else "length",
                           "generated_at": utc_now(), "human_reviewed": False}
                    if "test_hash" in item["request"]:
                        row["test_hash"] = item["request"]["test_hash"]
                    stream.write(canonical(row).decode("utf-8") + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    done.add(row["id"])
                logger.info("真实推理已完成 count=%d total=%d batch_seconds=%.3f", len(done), len(records), elapsed)
        if len(done) != len(records):
            raise RuntimeError("输出ID数量与请求数量不一致")
        metadata.update(state="completed", completed_count=len(done), finished_at=utc_now(),
                        output_sha256=file_hash(output), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save_json(metadata_path, metadata)
    except BaseException as error:
        metadata.update(state="failed", completed_count=len(done), failed_at=utc_now(), error_type=type(error).__name__)
        save_json(metadata_path, metadata)
        raise


def training_tokens(tokenizer, messages):
    """以原生模板前缀精确掩码，仅assistant正文和结束符参与loss。"""
    checked_messages(messages, training=True)
    prefix = prompt_ids(tokenizer, messages[:2])
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if not isinstance(full, list) or len(full) > 4096 or len(full) <= len(prefix):
        raise ValueError("训练完整序列为空、没有assistant监督或超过4096；禁止截断")
    if full[:len(prefix)] != prefix:
        raise ValueError("原生模板的生成前缀与完整消息不一致，无法安全建立assistant-only标签")
    labels = [-100] * len(prefix) + full[len(prefix):]
    if any(value != -100 for value in labels[:len(prefix)]) or not any(value != -100 for value in labels):
        raise RuntimeError("system/user标签掩码校验失败")
    return {"input_ids": full, "labels": labels, "prompt_tokens": len(prefix), "full_tokens": len(full)}


def smoke_samples(args, tokenizer):
    """只从train关联真实自动质检通过样本，覆盖类型、拒答子类及最长样本。"""
    cases = {}
    for _, case in read_jsonl(args.cases):
        if case.get("split") != "train":
            continue
        key = case.get("messages_hash")
        if not isinstance(key, str) or key in cases:
            raise ValueError("train cases缺少唯一messages_hash")
        cases[key] = case
    candidates = []
    seen = set()
    for number, row in read_jsonl(args.train):
        messages = checked_messages(row.get("messages"), training=True)
        key = digest({"messages": messages})
        case = cases.get(key)
        if key in seen or case is None:
            raise ValueError(f"train第{number}行重复或无法按messages_hash关联cases")
        if (case.get("automatic_quality", {}).get("status") != "passed"
                or case.get("experiment_mode") != "automated_learning_experiment"):
            raise ValueError(f"train第{number}行没有真实实验自动质检通过记录")
        if case.get("review", {}).get("status") != "pending":
            raise ValueError("自动学习实验必须明确保留pending人工审核状态")
        tokens = training_tokens(tokenizer, messages)
        quality = case["automatic_quality"]
        if quality.get("input_tokens") != tokens["prompt_tokens"] or quality.get("full_tokens") != tokens["full_tokens"]:
            raise ValueError(f"train第{number}行质检token计数与真实模板不一致")
        identity = case.get("id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("cases样本缺少id")
        candidates.append(dict(tokens, id=identity, messages_hash=key, task_type=case.get("task_type"),
                               refusal_type=case.get("refusal_type"), train_line=number))
        seen.add(key)
    if len(candidates) < 20:
        raise ValueError("train中真实质检样本不足20条")
    selected = []
    used = set()

    def choose(item):
        """按消息哈希去重加入覆盖样本。"""
        if item["messages_hash"] not in used:
            selected.append(item)
            used.add(item["messages_hash"])

    choose(max(candidates, key=lambda item: item["prompt_tokens"]))
    choose(max(candidates, key=lambda item: item["full_tokens"]))
    for name in ("task_type", "refusal_type"):
        groups = sorted({item[name] for item in candidates if item[name]})
        for group in groups:
            choose(max((item for item in candidates if item[name] == group), key=lambda item: item["full_tokens"]))
    if len(selected) > 20:
        raise ValueError("训练类型覆盖超过20条容量，不能静默遗漏类型")
    remainder = [item for item in candidates if item["messages_hash"] not in used]
    random.Random(42).shuffle(remainder)
    selected.extend(remainder[:20 - len(selected)])
    return selected, len(candidates)


def smoke(args, logger):
    """执行20条真实样本上的10个QLoRA优化步骤并实际保存和重载adapter。"""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training
    info = environment(torch)
    if args.output_dir.exists():
        raise FileExistsError("smoke输出目录必须独立且不存在")
    tokenizer = load_tokenizer(args.model)
    samples, train_count = smoke_samples(args, tokenizer)
    config = {"model": fingerprint(args.model), "train_sha256": file_hash(args.train),
              "cases_sha256": file_hash(args.cases), "sample_count": 20, "optimizer_steps": 10,
              "max_seq_length": 4096, "batch_size": 1, "gradient_accumulation_steps": 16,
              "learning_rate": 1e-4, "seed": 42, "data_seed": 42, "quant_type": "nf4",
              "double_quant": True, "compute_dtype": "bfloat16", "attention": "sdpa",
              "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
              "target_modules": TARGET_MODULES, "gradient_checkpointing": True,
              "loss": "assistant_only", "packing": False, "validation_loaded": False, "test_loaded": False}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "smoke-report.json"
    report = dict(info, config=config, config_hash=digest(config), state="running", started_at=utc_now(),
                  train_count=train_count, selected_samples=[{key: value for key, value in item.items()
                  if key not in {"input_ids", "labels"}} for item in samples],
                  system_user_labels_all_minus_100=True, optimizer_steps=[])
    save_json(report_path, report, exclusive=True)
    torch.cuda.reset_peak_memory_stats()
    try:
        model = load_base(args.model, torch, quantized=True)
        if not getattr(model, "is_loaded_in_4bit", False):
            raise RuntimeError("基座没有实际加载为4-bit")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                 target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM"))
        model.config.use_cache = False
        model.train()
        named_trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not named_trainable or any("lora_" not in name for name, _ in named_trainable):
            raise RuntimeError("可训练参数不是预期的LoRA参数集合")
        report["trainable_parameters"] = sum(parameter.numel() for _, parameter in named_trainable)
        optimizer = torch.optim.AdamW([parameter for _, parameter in named_trainable], lr=1e-4, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        order = list(range(len(samples)))
        randomizer = random.Random(42)
        randomizer.shuffle(order)
        position = 0
        with (args.output_dir / "steps.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for step in range(1, 11):
                losses = []
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(16):
                    if position == len(order):
                        randomizer.shuffle(order)
                        position = 0
                    sample = samples[order[position]]
                    position += 1
                    ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device="cuda")
                    labels = torch.tensor([sample["labels"]], dtype=torch.long, device="cuda")
                    if not torch.all(labels[:, :sample["prompt_tokens"]] == -100).item():
                        raise RuntimeError("真实GPU上的system/user标签不是-100")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        result = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
                        loss = result.loss
                    if not torch.isfinite(loss).item():
                        raise RuntimeError("QLoRA出现非有限loss")
                    losses.append(float(loss.detach().item()))
                    (loss / 16).backward()
                    del result, loss, ids, labels
                gradients = [parameter.grad for _, parameter in named_trainable if parameter.grad is not None]
                if len(gradients) != len(named_trainable):
                    raise RuntimeError("存在未获得梯度的LoRA可训练参数")
                if any(not torch.isfinite(gradient).all().item() for gradient in gradients):
                    raise RuntimeError("LoRA出现非有限梯度")
                norm = torch.nn.utils.clip_grad_norm_([parameter for _, parameter in named_trainable], 1.0,
                                                       error_if_nonfinite=True)
                grad_norm = float(norm.item())
                if not math.isfinite(grad_norm) or grad_norm <= 0:
                    raise RuntimeError("LoRA梯度范数不是有限非零值")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if any(not torch.isfinite(parameter).all().item() for _, parameter in named_trainable):
                    raise RuntimeError("优化器更新后LoRA参数不是有限值")
                torch.cuda.synchronize()
                record = {"optimizer_step": step, "microsteps": 16, "mean_loss": sum(losses) / len(losses),
                          "microstep_losses": losses, "lora_grad_norm_before_clip": grad_norm,
                          "all_lora_gradients_finite": True, "elapsed_seconds": time.perf_counter() - started}
                report["optimizer_steps"].append(record)
                stream.write(canonical(record).decode("utf-8") + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                logger.info("QLoRA冒烟 step=%d loss=%.6f grad_norm=%.6f", step, record["mean_loss"], grad_norm)
        adapter_path = args.output_dir / "adapter"
        model.save_pretrained(str(adapter_path), safe_serialization=True)
        tokenizer.save_pretrained(str(adapter_path))
        expected_state = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}
        report.update(training_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      training_peak_reserved_bytes=torch.cuda.max_memory_reserved(), adapter=fingerprint(adapter_path))
        del optimizer, named_trainable, gradients, model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        reloaded = load_base(args.model, torch, quantized=True)
        reloaded = PeftModel.from_pretrained(reloaded, str(adapter_path), local_files_only=True, is_trainable=False)
        actual_state = get_peft_model_state_dict(reloaded)
        if actual_state.keys() != expected_state.keys() or any(
                not torch.equal(actual_state[key].detach().cpu(), expected_state[key]) for key in expected_state):
            raise RuntimeError("adapter保存后重载权重与保存前不一致")
        reloaded.eval()
        sample = samples[0]
        ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device="cuda")
        labels = torch.tensor([sample["labels"]], dtype=torch.long, device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            reload_loss = reloaded(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss
        if not torch.isfinite(reload_loss).item():
            raise RuntimeError("adapter重载后的真实前向loss不是有限值")
        report.update(state="completed", finished_at=utc_now(), adapter_reload_equal=True,
                      reload_forward_loss=float(reload_loss.item()),
                      reload_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      reload_peak_reserved_bytes=torch.cuda.max_memory_reserved(), microsteps_completed=160)
        save_json(report_path, report)
    except BaseException as error:
        report.update(state="failed", failed_at=utc_now(), error_type=type(error).__name__)
        save_json(report_path, report)
        raise


def merge(args, logger):
    """从原始BF16基座加载adapter并安全合并，保存safetensors与原生分词器。"""
    import torch
    from peft import PeftModel
    if args.output_dir.exists():
        raise FileExistsError("merge输出目录必须独立且不存在")
    info = environment(torch) if args.device == "cuda" else {
        "versions": versions(), "experiment_mode": "automated_learning_experiment", "human_reviewed": False,
        "publication_approved": False, "base_revision_contract": BASE_REVISION}
    config = {"model": fingerprint(args.model), "adapter": fingerprint(args.adapter), "dtype": "bfloat16",
              "attention": "sdpa", "device": args.device, "safe_serialization": True, "safe_merge": True}
    tokenizer = load_tokenizer(args.model)
    model = load_base(args.model, torch, device=args.device)
    if getattr(model, "is_quantized", False) or getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("禁止从量化模型合并，必须使用原始BF16基座")
    model = PeftModel.from_pretrained(model, str(args.adapter), local_files_only=True, is_trainable=False)
    merged = model.merge_and_unload(safe_merge=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "merge-report.json"
    report = dict(info, config=config, config_hash=digest(config), state="running", started_at=utc_now())
    save_json(report_path, report, exclusive=True)
    try:
        merged.save_pretrained(str(args.output_dir), safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(str(args.output_dir))
        report.update(state="completed", finished_at=utc_now(), merged_model=fingerprint(args.output_dir))
        save_json(report_path, report)
        logger.info("原始BF16基座与adapter已合并 output=%s", args.output_dir)
    except BaseException as error:
        report.update(state="failed", failed_at=utc_now(), error_type=type(error).__name__)
        save_json(report_path, report)
        raise


def positive_int(value):
    """拒绝非正整数运行参数。"""
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return result


def parser():
    """声明已实现且能真实执行的推理、冒烟和合并命令。"""
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    inference = commands.add_parser("generate", help="本地BF16基座及可选adapter真实贪心推理")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--adapter", type=Path)
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--batch-size", type=positive_int, default=1)
    inference.add_argument("--max-new-tokens", type=positive_int, default=1024)
    inference.add_argument("--resume", action="store_true")
    experiment = commands.add_parser("smoke", help="20条真实train自动质检样本，10个QLoRA优化器步骤")
    experiment.add_argument("--model", type=Path, required=True)
    experiment.add_argument("--train", type=Path, required=True)
    experiment.add_argument("--cases", type=Path, required=True)
    experiment.add_argument("--output-dir", type=Path, required=True)
    export = commands.add_parser("merge", help="原始BF16基座加adapter安全合并为safetensors")
    export.add_argument("--model", type=Path, required=True)
    export.add_argument("--adapter", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return result


def main(argv=None):
    """读取独立日志配置，运行指定真实操作并以非零退出码报告失败。"""
    args = parser().parse_args(argv)
    logging.config.fileConfig(ROOT / "configs" / "logging.ini", disable_existing_loggers=False)
    logger = logging.LoggerAdapter(LOGGER, {"run_id": uuid4().hex})
    logger.info("云端模型流程开始 command=%s", args.command)
    try:
        {"generate": generate, "smoke": smoke, "merge": merge}[args.command](args, logger)
    except Exception as error:
        logger.error("云端模型流程失败 command=%s error_type=%s", args.command, type(error).__name__)
        raise
    logger.info("云端模型流程完成 command=%s", args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
