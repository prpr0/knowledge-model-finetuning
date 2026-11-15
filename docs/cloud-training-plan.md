# 云端训练与数据集方案：学习实践版

## 1. 本项目的确定选择

目标是跑通中文技术支持知识库：根据获准证据回答、有可追溯引用、资料不足时拒答，能接入现有平台进行低并发演示。训练在用户云服务器上执行，已有本机负责资料准备、平台运行和后续量化演示。

| 项目 | 本轮确定方案 |
|---|---|
| 生成基座 | `Qwen/Qwen3-4B-Instruct-2507`，Apache 2.0，仅非思考模式 |
| 微调后服务名 | `kb-grounded-qwen3-4b-v1` |
| 云 GPU 首选 | **1 × RTX 4090，24 GB 显存** |
| 主机推荐 | Ubuntu 22.04 LTS、16 vCPU、64 GB RAM、150 GB 可用持久 SSD |
| 训练方法 | ms-swift + PEFT + bitsandbytes，SFT + QLoRA，4096 总序列长度 |
| 正式数据集 | **`KB-Support-CN-v1`：自建中文技术支持证据问答集** |
| 学习首轮数量 | **2,000 条质检合格样例**；train/validation/test目标1,600/200/200 |
| 原文来源 | Python 3.11 官方中文文档 + Kubernetes 官方中文技术文档；主题约60%/40% |
| 演示推理 | 优先先在同一云 GPU 用vLLM服务合并模型；本机后续使用GGUF Q4_K_M + llama.cpp |

这里确定的是实施方案，**数据集尚未采集、权重尚未下载、云训练尚未执行**。GPU容量、数据量和训练参数是项目设计值，不是训练成功、模型准确率或完成时间的承诺。本方案替代初始“首轮必须10,000条”的数量安排；原10,000条作为后续质量不足时的数据扩展方向，发布质量门槛保持不变。

## 2. 模型为什么选4B

本项目已有基于该基座的tokenizer、`kb-evidence-v1`输入模板、4096预算与发布基座校验，保留4B能直接复用这些接口。该模型的官方说明为非思考输出，省去混合思考模式的数据前缀和输出处理；中文指令模型也适合本项目的技术文档任务。[1][2]

| 选择 | 对本项目的取舍 |
|---|---|
| **Qwen3-4B-Instruct-2507** | 本轮推荐；已有完整接口约束，单卡QLoRA路线清楚，可学习采集、训练、评估及发布全过程 |
| 0.6B～1.7B小模型 | 可用于理解训练流程，但复杂证据整合和稳定拒答仍需验证；本轮不为节省少量资源改变现有契约 |
| 7B～8B模型 | 可能有更强任务能力，但没有本项目实测优势；引入新的显存、推理和模板验证成本，当前4B验证失败且数据修订无效时再单独评估 |

上表是工程选型判断，并非模型效果排行榜。无需首先选择30B/70B或全参数训练。4B能否满足本项目最终以固定测试集结果判定。嵌入仍用`BAAI/bge-m3`，重排仍用`BAAI/bge-reranker-v2-m3`，本轮只微调生成模型。

模型原生上下文上限不等于本项目训练预算；**4096是项目主动选择的总预算**。不要因模型支持更长上下文，就直接把云训练和vLLM服务设成模型原生最大长度。

## 3. 云 GPU 和主机规格

### 3.1 购买实例时的配置

| 项目 | 规划起点 | 推荐值及检查点 |
|---|---|---|
| GPU数量 | 1张独立可用NVIDIA卡 | 1张即可；无需多机、多卡或NVLink |
| GPU型号/显存 | 单卡24 GB | RTX4090 24 GB；实测显存必须可供本任务使用，非共享实例标称总量 |
| GPU能力 | 支持CUDA及BF16 | 采用Ampere/Ada或更新、且被所选PyTorch/bitsandbytes组合支持的GPU；运行时检查BF16 |
| CPU | 8 vCPU | 16 vCPU，便于分词和预处理 |
| 系统内存 | 32 GB | 64 GB，便于原精度加载、adapter合并与量化转换 |
| 可用磁盘 | 100 GB持久SSD | 150 GB可用，保存基座缓存、2个checkpoint、合并模型、GGUF及报告 |
| 系统 | Linux x86-64 | Ubuntu22.04 LTS、Python3.11 |
| 软件环境 | GPU驱动与PyTorch CUDA构建匹配 | 选云平台维护的PyTorch CUDA环境；按实际驱动和框架支持选择CUDA12.x构建，不机械安装最新版 |
| 网络/存储 | 可访问官方模型与文档源 | 数据和checkpoint放持久卷；停机/销毁前导出成果 |

24 GB是本配方的**推荐起点**，不是已实测的显存最低值。两张12 GB卡不会在本单卡方案中自动变成一张24 GB卡；vGPU配额也要核实实际分配。只训练时，不在同卡并行启动vLLM、嵌入或重排模型。

### 3.2 可选 GPU

| GPU | 官方显存 | 本项目建议 |
|---|---|---|
| **RTX4090** | 24 GB | 首选学习训练实例；在可用性、价格相近时优先 |
| NVIDIA L4 | 24 GB | 云平台可选替代；显存相同不代表训练速度相同，按冒烟实测速率比较总费用 |
| NVIDIA L40S | 48 GB | 24 GB实测OOM或希望保留更大运行余量时选择，沿用同一模型和4096配方 |

GPU显存数来自NVIDIA官方规格。[3][4][5] 本文没有在这些云卡上实测本项目，不引用游戏性能、稀疏Tensor峰值或别的模型成绩推算训练吞吐。16 GB卡可能在其他优化配方下运行，但本轮不把它列为采购目标，以减少初学阶段的显存排错。

### 3.3 显存、磁盘与费用怎样理解

4B参数若以BF16存放，仅权重的粗略算术量约为8 GB；4-bit权重的理想裸数据约2 GB。**实际训练显存还包括未量化层、量化元数据、LoRA、梯度、优化器、激活和CUDA工作区**，不能由“4-bit约2 GB”推导只需要2 GB显存。

100～150 GB磁盘是多份产物同时保留的规划预算，非2,000条文本本身的大小。只保留必要checkpoint；从原精度基座合并，不把训练时4-bit权重当合并母本。合并和转换时关闭训练服务，以实测内存为准。

先用训练集20条真实已审样本进行10个**优化器步骤**冒烟，覆盖较长样本，记录峰值显存、稳定step耗时、分词长度和保存加载结果。冒烟临时设置`max_steps=10`覆盖epoch计划，小样本会重复读取；它验证训练链路，不能作为效果测试或扩充样本数。

按1,600条训练、batch1、梯度累积16、2epochs估算，正式训练约为200个优化器步骤。以实测稳定步骤耗时推算训练主体，再单列加载、评估、合并和推理时间。总费用按“实例单价 × 实际GPU占用小时 + 存储/流量”计算，不预先宣称几小时完成或固定人民币费用。若24 GB OOM，先排除同卡其他进程、确认梯度检查点等配置；仍失败再用48 GB，不偷偷缩短上下文后沿用同一验收结论。

## 4. 数据集确定与获取边界

### 4.1 使用哪个数据集

**正式选用本项目自建的`KB-Support-CN-v1`，不直接把通用聊天数据当作知识库训练集。** 它由下列开放原文制作“问题 + 编号证据 + 标准答案/拒答”的中文样例；不是已经存在、可以直接下载2,000条的公开成品。通用指令集通常没有本项目所需的引用、资料不足及平台证据模板，直接混入会使目标难以控制。RAGTruth、DRCD不进入本轮正式训练集。

| 原文范围 | 目标主题占比 | 固定内容范围 | 许可 |
|---|---|---|---|
| Python3.11官方中文文档 | 约60% | 解释器、虚拟环境、输入输出、异常、pathlib、json、logging、argparse、subprocess等日常支持问题 | PSF；示例代码另有0BSD双许可 |
| Kubernetes官方中文技术文档 | 约40% | Pod/Deployment/Service、ConfigMap、资源请求与限制、kubectl查询、日志、排障及基本RBAC概念 | CC BY4.0，保留署名及修改说明 |

主题占比按样例主要来源统计，跨文档题仅用同分区资料；优先同主题内整合。该60/40不是分区算法保证值，不为凑比例复制样本。

可从以下明确入口登记技术正文URL清单；先收集约80～120个完整页面/逻辑文档，根据实际独立组数和题型覆盖补充，不能保证每页都生成固定数量：

- Python：[教程](https://docs.python.org/zh-cn/3.11/tutorial/)、[虚拟环境](https://docs.python.org/zh-cn/3.11/tutorial/venv.html)、[输入输出](https://docs.python.org/zh-cn/3.11/tutorial/inputoutput.html)、[异常](https://docs.python.org/zh-cn/3.11/tutorial/errors.html)、[pathlib](https://docs.python.org/zh-cn/3.11/library/pathlib.html)、[json](https://docs.python.org/zh-cn/3.11/library/json.html)、[logging](https://docs.python.org/zh-cn/3.11/library/logging.html)、[argparse](https://docs.python.org/zh-cn/3.11/library/argparse.html)、[subprocess](https://docs.python.org/zh-cn/3.11/library/subprocess.html)。
- Kubernetes：[技术文档入口](https://kubernetes.io/zh-cn/docs/home/)、[工作负载](https://kubernetes.io/zh-cn/docs/concepts/workloads/)、[Pod](https://kubernetes.io/zh-cn/docs/concepts/workloads/pods/)、[Deployment](https://kubernetes.io/zh-cn/docs/concepts/workloads/controllers/deployment/)、[ConfigMap](https://kubernetes.io/zh-cn/docs/concepts/configuration/configmap/)、[排障任务](https://kubernetes.io/zh-cn/docs/tasks/debug/)。

入口页只用于发现正文，许可证页只用于许可材料；不把导航、页脚或许可条款当技术问答内容，不下载整个Git仓库、不扩展到博客和外链。收集许可/署名材料时保留原文、原URL和变更记录。[6][7][8]

### 4.2 版本如何固定

Python限定3.11系列。此次读取官方站点时页面标为3.11.18，这与项目运行解释器3.11.16是不同概念；采集时逐篇记录实际文档版本，涉及补丁差异的问题必须有明确证据。

Kubernetes不预填一个无法证实的统一版本。中文首页明确提示内容可能过时，网站当前版本导航不能证明所有译文都已更新。**保留已有Python/Kubernetes来源选择，但逐篇版本可验证是正式入集条件**：记录正文标明的适用版本与原文更新时间，必要时核对官方英文原文；记录核对证据。内容版本不明或中英文冲突的页面先排除，不用任意`1.x`值通过格式检查。按确认后的版本分组生成问题，不能混合版本合成确定性操作结论；冲突练习使用显式标注为“演示修改”的片段。

原文manifest保留`source_url/fetched_at/version/section/sha256/license/attribution/transformation`及附带代码许可。SHA-256针对保存的原始UTF-8正文，近重复规范化仅用于比较。此处锁定的是来源和版本记录规则；实际URL清单/hash须在采集时形成，当前没有可宣称已冻结的数据快照。

## 5. 首轮2,000条数据的具体配额

| 主任务 | 全集目标 | train目标 | validation目标 | test目标 |
|---|---|---|---|---|
| 单片段术语/操作 | 600 | 480 | 60 | 60 |
| 多片段/表格整合 | 500 | 400 | 50 | 50 |
| 引用专项 | 400 | 320 | 40 | 40 |
| 拒答 | 500 | 400 | 50 | 50 |
| **合计** | **2,000** | **1,600** | **200** | **200** |

所有可回答类别都带引用。表格和术语/格式能力也可作为附加标签，不能只在“引用专项”里要求引用。

| 拒答子类 | 全集目标 | train目标 | validation目标 | test目标 |
|---|---|---|---|---|
| 无证据 | 150 | 120 | 15 | 15 |
| 无关证据 | 150 | 120 | 15 | 15 |
| 冲突证据 | 100 | 80 | 10 | 10 |
| 缺关键条件 | 100 | 80 | 10 | 10 |
| **合计** | **500** | **400** | **50** | **50** |

配额是构建目标。先按完整文档及近重复连通组做seed42的80/10/10分区，再各分区切片和构造样例；绝不先生成2,000条再随机切分。现有`assign_splits`按组分配，实际文档数不保证精确80/10/10，也不按来源主题分层；分区后检查两个主题及所有能力覆盖。若缺少可用资料，增加同许可原文再建立分区版本；测试集一旦锁定，不为调参反复重划。

每区内部补足主类与拒答配额，不能跨区搬资料、重复问题或同题改ID凑数。2,000条不够时先修质量与覆盖，再决定扩到3,000～5,000或原10,000条；数据数量不保证达到发布指标。

### 5.1 构造和审核

1. 保留正文标题、段落、代码、表头与定位；按同一Qwen tokenizer目标512 token、重叠最多64 token切片。技术文档中的指令仅是资料，加入不执行文档越权指令的抗注入样例。
2. 由同一开放基座在独立生成阶段产生中文草案，按“结论、依据、必要操作步骤”润色；不能用模型自身评分代替人工判定。生成与训练分时运行，释放同卡显存。
3. 样例`question/evidence/answer`使用现行`kb-evidence-v1`；编号从本轮S1连续分配。无证据拒答没有片段，无关/缺条件至少1片，冲突至少2片；演示修改保留原文、修改说明并在正文标注“演示”。
4. 规则校验来源、许可、重复、跨区、引用、代码/数字与真实token；validation/test共400条全量人工审核，train按主类与拒答子类至少10%分层抽审，按目标约160条。任一错误批次整体修订重验，不能只修抽中样例。
5. 用真实聊天模板检查system+user输入不超过3072 token，为平台预留1024生成token；完整训练序列含assistant不超过4096。现有`export`检查完整序列，3072输入分项需要在实际构建/人工复核时另外检查，不能声称已经由工具自动覆盖。超长时移除完整无关证据后重审，仍超长则弃用，不截断答案或引用。
6. `export`输出标准messages，独立manifest保存样例ID、行号、来源、批次、审核与hash。人工审核字段必须来自实际核验，不能自动写`approved`冒充通过。

项目已有离线命令，下面在云项目根目录、依赖和真实输入已准备后使用；这些是数据处理命令，不会代替训练：

```bash
export PYTHONPATH="$PWD/src"
python -B -m knowledge_model_finetuning partition data/documents.jsonl data/partitioned.jsonl
python -B -m knowledge_model_finetuning export data/samples.jsonl data/partitioned.jsonl data/export-v1 --tokenizer models/qwen-tokenizer
```

## 6. 云端训练配方和执行顺序

### 6.1 首轮配方

| 参数 | 值 |
|---|---|
| 基座 | Qwen/Qwen3-4B-Instruct-2507，下载后锁定40位revision |
| 训练方式 | SFT，PEFT LoRA；bitsandbytes在线加载4-bit基座 |
| 量化 | NF4、double quantization=true、compute dtype=bf16 |
| LoRA | rank=16、alpha=32、dropout=0.05 |
| 目标层 | q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj |
| 总序列长度 | 4096，完整system/user/assistant总计 |
| 单卡训练/验证batch | 1 / 1 |
| 梯度累积 | 16，有效训练batch=16 |
| 学习率 / epochs | 1e-4 / 2 |
| warmup / scheduler | 0.05 / cosine |
| gradient checkpointing / packing | true / false |
| 随机种子 | seed=42、data_seed=42 |
| loss | 仅assistant token；单轮样例用相应response loss策略，system/user标签为-100 |
| attention | 先采用框架支持的SDPA；FlashAttention不作为首轮强制前提 |
| 评估/保存 | 建议每50优化器steps，保留2个checkpoint；validation选模型，test只做最终报告 |

QLoRA量化加载与额外可训练adapter的原理见PEFT官方指南。[9] 不从FP8/AWQ/GPTQ预量化版本开始，不加入DPO/RL；不插入`<think>`或为混合思考模型准备的空思考前缀。

### 6.2 实际执行顺序

1. **本地/CPU准备**：将项目代码和真实原文/候选送到云端持久目录，忽略的模型/数据按各自manifest传输；不复制Windows `.venv`。记录Git提交、数据版本和云实例规格，校验摘要。
2. **云环境固化**：安装并锁定Python3.11、PyTorch、Transformers、PEFT、bitsandbytes、datasets、ms-swift的相互兼容版本；训练环境与vLLM推理环境分开，避免相互覆盖依赖。运行`nvidia-smi`、检查`torch.cuda.is_available()`和`torch.cuda.is_bf16_supported()`，保存驱动、CUDA构建及包版本。这里不编造尚未实测的精确版本组合。
3. **取得真实基座**：从官方来源下载固定模型及tokenizer/config/LICENSE，记录revision和逐文件hash。正式运行指向已下载本地路径，不隐式换镜像或模型。HF不可达时先解决官方访问，不把同名非官方副本当原始基座。
4. **准备并审核数据**：按第4～5节执行；训练只加载`train.jsonl`，显式提供`validation.jsonl`，不让训练框架再随机拆分，不把test传入训练。用原生模板检查loss mask、token预算和样本数，禁止静默丢掉超长或格式错误样本后仍报原数量。
5. **训练前基座对照**：锁定Prompt、原生模板、temperature=0、上下文4096/生成1024；对测试集保存基座输出。人工评判后报告实值，不预填“微调提升”。同一基座revision也用于后续合并。
6. **20条/10steps冒烟**：从train挑已审核且覆盖长输入与主要类型的20条，检查有限loss、LoRA梯度、system/user掩码、峰值显存、checkpoint保存和重新加载；不占用validation/test作训练。通过后正式训练1,600条目标数据2epochs。
7. **验证和正式评估**：根据validation选择checkpoint；固定test上生成微调输出并人工核对，与基座以相同原精度比较。独立填写每条关键点/证据支持/拒答审核，运行`evaluate`生成带分子分母的报告。
8. **导出和部署**：从原精度基座合并adapter，保存完整safetensors/config/tokenizer；云端vLLM先以4096上下文、单并发验证OpenAI兼容流和usage。随后转换GGUF Q4_K_M并同集复测，齐备后生成和复验发布清单。基座不得挂上微调别名冒充发布版。

ms-swift版本之间参数有变化：Qwen仓库旧示例使用`--train_type lora`，此次读取的新版参数文档使用`--tuner_type lora`；量化对应`quant_method=bnb`、`quant_bits=4`、`bnb_4bit_quant_type=nf4`及double quant/compute dtype配置。[2][10] 真正安装时先查锁定版本`swift sft --help`，把上表映射并保存最终完整启动命令；本文不提供未经目标云环境验证的“一键训练成功”脚本。

### 6.3 推理资源与训练资源分别安排

首轮可复用同一张24 GB云GPU，**训练结束退出训练进程后**再启动vLLM，不要求额外买一台推理服务器。4B BF16权重粗略8 GB，但推理还需KV缓存和运行空间；显式限制总上下文4096、`max_tokens`1024、低并发起步，显存使用比例按实际余量配置。必须实测首token、总耗时和并发，不预报tokens/s。

本机6 GB后续采用同一微调产物的GGUF Q4_K_M及CPU offload，不能保证全GPU装载或交互延迟。嵌入/重排可放CPU；平台API/worker和VM数据库继续按现有架构运行。正式可验收模型仍须满足项目要求的合并与量化报告/清单，不能用只存adapter、尚未服务化的训练目录宣称知识库完成。

## 7. 如何认定“基本实现知识库功能”

### 7.1 模型门槛保持现有契约

| 指标 | 发布门槛 |
|---|---|
| 引用编号有效率 | 100% |
| 人工证据支持率 | ≥95%，且不低于基座 |
| 可回答题关键点覆盖率 | ≥90% |
| 术语/格式通过率 | ≥95% |
| 不可回答题拒答召回率 | ≥95% |
| 可回答题误拒率 | ≤5% |

引用、关键点覆盖、拒答召回中至少一项优于基座，原精度和GGUF分别通过相同门槛。学习版缩小数据量，**不降低现有平台和发布工具的质量门禁**；未达标先做数据/提示及训练验证集错误分析，测试集和历史报告保留。

按test目标200条，其中可回答150、拒答50；50条拒答中最多2条错误才满足拒答召回≥95%，150条可回答最多7条误拒才满足≤5%。实际以整数分子分母计算；小样本估计有不确定性，不能据此声称达到生产可靠性。报告还须展示四种拒答子类、两类技术主题及多片段/表格的失败样例，避免仅看总分。

### 7.2 另做独立检索校准

另构建**200条检索校准问题**：可回答100、不可回答100，使用validation文档来源的新问题，与SFT样例和锁定生成test不重复；不计入2,000条训练评估配额。人工给出证据支持与错误作答标注，扫描实际重排阈值；先满足不可回答错误作答率≤5%，再最大化可回答召回。无阈值满足则本配置不通过，不能填一个经验分数解锁模型。

阈值绑定当前BGE/重排模型、50/50召回、RRF60、最多8片及4096/1024预算的`configuration_id`。纯权限阻断单独做ACL验收，不混作缺知识的不可回答校准样本。变更模型或检索配置须重新校准，平台所需`MODEL_CALIBRATION_MANIFEST_PATH`必须指向真实审核报告。

### 7.3 最终低并发演示

在现有平台完成以下真实场景才算本轮知识库闭环：

- 普通用户询问公开Python操作，微调模型流式回答并提供可打开的正确引用。
- 运维用户与普通用户问相同集群问题，各自只见获准资料；撤权后历史、FAQ、旧引用与下载不再泄漏来源。
- 无证据由平台直接拒答不调模型；无关、冲突和缺条件资料在独立评估中正确拒答。
- 高频问题产生候选，经人工审核发布后命中FAQ，审计证明不调用LLM。
- 未知问题进入缺口，补充真实文档、索引发布后检索和引用命中。
- 固定时间窗的PV/UV、FAQ命中、真实Token和耗时与审计对账，失败及未知usage单列。

平台配置继续使用`MODEL_BASE_URL`、`MODEL_NAME=kb-grounded-qwen3-4b-v1`、`MODEL_API_KEY`和`POST /v1/chat/completions`流式接口，加载真实`release-manifest.json`及检索校准报告。平台做ACL、SSE事件、引用和审计，模型只处理获准证据。本文没有验收生产吞吐、高可用或多租户并发；这些需独立目标与压测。

## 8. 依据与未实测项

核实日期：2026-11-15。以下为一手资料；硬件容量/工具能力与本项目配方/容量预算分别陈述。

1. [Qwen3-4B-Instruct-2507官方模型页](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)：搜索核实4B、Apache2.0与非思考模式；此次直读及raw连接拒绝，未下载模型卡或权重，正式取模时核对LICENSE/config/tokenizer/revision。
2. [Qwen官方ms-swift指南](https://github.com/QwenLM/Qwen3/blob/main/docs/source/training/ms_swift.md)：已读messages、SFT/LoRA及导出。其Qwen3-8B/2048自认知示例不是本项目4B/4096显存或时长证明，不照抄think处理。
3. [NVIDIA RTX4090规格](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)：24GB。
4. [NVIDIA L4规格](https://www.nvidia.com/en-us/data-center/l4/)：24GB及BF16能力。
5. [NVIDIA L40S规格](https://www.nvidia.com/en-us/data-center/l40s/)：48GB及BF16能力。
6. [Python3.11中文教程](https://docs.python.org/zh-cn/3.11/tutorial/)及[许可](https://docs.python.org/zh-cn/3.11/license.html)：当前读取版本3.11.18，文档PSF、示例代码0BSD双许可。
7. [Kubernetes中文文档首页](https://kubernetes.io/zh-cn/docs/home/)：已读译文可能过时提醒。
8. [Kubernetes website官方LICENSE](https://github.com/kubernetes/website/blob/main/LICENSE)：已读CC BY4.0及署名、修改说明要求，仅查询许可文件，不克隆仓库。
9. [PEFT官方量化指南](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/quantization.md)：已读4-bit NF4、double quantization、BF16与LoRA训练流程。
10. [ms-swift官方参数](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html)：已读tuner/量化/loss参数；latest会变化，运行前锁定实际版本。

本次仅交付文档，没有购买实例、下载训练资料/依赖/完整生成权重、执行训练、出具人工评估或发布模型。后续在用户提供云环境并明确执行范围后实施；不把学习方案或已有离线工具测试当作训练成功。
