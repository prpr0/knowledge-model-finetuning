# 知识库模型微调

面向中文技术支持与操作知识库的证据问答模型微调项目，当前提供可执行的离线语料分区、正式样例质检导出、对照评估与发布文件校验工具。

**正式发布仍要求真实人工审核、质量门禁及平台最终联调。** 离线单元测试不构成模型成果；云端自动学习实验及其运行产物与正式发布分开记录，不把自动质检当作人工批准。

本项目对应 `real/4.1.md` 的术语格式、忠实回答、多片段整合、引用和拒答能力需求，为独立项目 `knowledge-base-platform` 交付生成模型；平台的权限、检索与会话仍由平台实现。 两项目通过 OpenAI 兼容服务契约协作，不复制代码或导入兄弟目录。微调模型发布前，基座对照实验不计为最终联调成功。

详细技术栈、架构、需求覆盖、实施顺序、统一契约与未来验收见 [实现方案](docs/implementation-plan.md)。

用户云端学习训练的确定方案见[云端训练与数据集方案](docs/cloud-training-plan.md)：`Qwen3-4B-Instruct-2507` + 单卡RTX4090 24GB + SFT/QLoRA；推荐16vCPU、64GB RAM、150GB可用持久盘。首轮使用自建`KB-Support-CN-v1`，来自Python3.11/Kubernetes官方中文正文，目标2,000条，train/validation/test目标1,600/200/200。原10,000条改作后续扩展项；数据与模型质量仍须实测，不降低发布门槛。

## 云端自动学习实验

`scripts/`提供真实官方语料采集、基座推理、QLoRA冒烟及原精度合并。学习实验采用`automated_learning_experiment`，所有人工审核状态保持`pending`，不调用正式`export`绕过人工门禁。

- `prepare_experiment.py collect/prepare`保存官方HTML、许可、版本证据并先按完整文档分区；Python限定3.11，Kubernetes逐页核验官方版本快照。
- `prepare_extractive.py build`读取`generation-plan.jsonl`、`partitioned.jsonl`、`chunks.jsonl`及本地tokenizer，构造`extractive-plan.jsonl`和`question-requests.jsonl`。基座仅出题，参考答案由已核验原文摘录或受控拒答派生；代码不把这些参考答案冒称为模型自由生成。
- `runtime.py generate`批量保存真实输出与token/哈希；`prepare_extractive.py assemble`保留原始出题输出并组装样例，再用`prepare_experiment.py finalize`做2000条完整质检及测试集锁定。失败项用`prepare_extractive.py repair`有限重生成，不静默缩量。
- 对`test-requests.jsonl`先生成基座结果，再运行`runtime.py smoke`检查真实梯度、assistant-only掩码和保存重载。正式训练显式传入train/validation并按validation loss选模型，test不参与训练；随后`runtime.py merge`从原始BF16基座合并。
- `verify_deployment.py`实际核验Swift的普通及SSE OpenAI接口和usage，启动前检查端口空闲、就绪后校验模型名，默认回收整个自有进程组并确认端口释放；最终模型可用`--keep-running`保留已验证的回环服务。`download_model.py`按锁定官方revision续传并核验官方LFS SHA-256。

`prepare_experiment.py compare`输出同一固定测试集的逐题对照和自动代理指标，包括引用编号、字面关键点、启发式拒答及批次分摊耗时。代理指标不等同人工证据支持、语义正确率或独立单请求延迟，不产生正式发布通过结论。摘录能逐字追溯，不代表问题和关键点完整性已经人工审核。具体环境、命令和结果以运行目录为准；本轮云端沿用已实测支持RTX5090的Python3.12.3及PyTorch2.12.1/CUDA13.0，核心离线工具仍兼容Python3.11。

### Git归档范围

本仓库保留微调工作流代码、测试代码和[本轮对比报告](results/cloud-run-20261117/comparison.md)。两次原始模型输出分别为[基线结果](results/cloud-run-20261117/baseline.jsonl)和[微调结果](results/cloud-run-20261117/finetuned.jsonl)，另有[逐题对比](results/cloud-run-20261117/comparison.jsonl)与[统计汇总](results/cloud-run-20261117/summary.json)。

Git不包含模型权重、adapter、tokenizer、训练参数文件、训练数据、测试输入、原始文档、环境快照或压缩备份；这些本机文件由`.gitignore`排除。逐题对比只保留样本ID、分类、双侧输出与代理指标，移除了问题、证据、参考答案和关键点原文。运行代码时需自行提供模型、数据和训练参数；`configs/logging.ini`仅用于程序日志。

本轮200题的输入token哈希一致，结果属于未经人工审核的自动代理评估。旧基座服务子进程曾驻留，已回收并修复进程组管理；耗时只作为批次运行记录，不作为资源隔离条件下的性能结论。


## 运行与验证

Python 3.11.x、uv；核心工具仅用标准库，`dependencies=[]`，锁文件保留。`export`额外要求Transformers及现有本地Qwen3 tokenizer；云端训练依赖在独立环境安装，实际版本保存到实验的`environment-freeze.txt`。其他机器可替换为本地已装3.11解释器。

```powershell
uv --offline --no-python-downloads venv --python D:/uv/python/cpython-3.11.16-windows-x86_64-none/python.exe .venv
$env:PYTHONPATH = (Join-Path $PWD 'src')
.venv/Scripts/python.exe -B -m knowledge_model_finetuning --help
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

命令在项目根运行。包采用src布局且`[tool.uv] package=false`，通过PYTHONPATH使用，不宣称已安装分发包。CLI日志由`configs/logging.ini`独立配置，stderr记录run_id、阶段及错误，stdout输出JSON摘要。失败非零退出，不输出成功状态；现有输出文件不覆盖。

## 离线操作

以下路径为准备好真实资料后的输入约定，仓库不预置假数据或模型。数据与模型均在被忽略的目录。

```powershell
# 先验证原文及许可，再以近重复连通组分区。
.venv/Scripts/python.exe -B -m knowledge_model_finetuning partition data/documents.jsonl data/partitioned.jsonl
# 使用已配置Transformers的解释器；缺少包或本地模型时明确失败，不联网。
python -B -m knowledge_model_finetuning export data/samples.jsonl data/partitioned.jsonl data/export-v1 --tokenizer models/qwen-tokenizer
# 输入基座和微调模型在同一锁定测试集上的真实输出及人工审核记录。
.venv/Scripts/python.exe -B -m knowledge_model_finetuning evaluate data/evaluation-records.jsonl outputs/evaluation.json
# root内需已存在所有真实模型文件和原精度/量化报告；规格文件只列相对路径。
.venv/Scripts/python.exe -B -m knowledge_model_finetuning release artifacts/release-v1 configs/release-specification.json
.venv/Scripts/python.exe -B -m knowledge_model_finetuning verify-release artifacts/release-v1/release-manifest.json
```

`evaluate`保留不通过的完整报告并返回2；缺少审核或输入格式错误直接失败。发布需使用新版本目录，文件变更后旧清单摘要失效。发布清单生成只证明门禁和文件一致性，不能凭文件名证明训练真实完成。

## 原文与样例输入

`documents.jsonl`每行是一个完整逻辑文档，必需字段如下：

| 字段 | 约束 |
|---|---|
| `id`、`source_url` | 唯一ID；官方HTTPS正文URL，限`docs.python.org/zh-cn/3.11/`和`kubernetes.io/zh-cn/docs/` |
| `fetched_at`、`version`、`section` | 带时区ISO8601时间；Python为3.11系列，Kubernetes为明确1.x发布版本；保留章节 |
| `license`、`attribution`、`transformation` | 分别为`PSF-2.0`或`CC-BY-4.0`、实际署名、正文清洗/转换说明；附带许可可额外保存 |
| `text`、`sha256` | 原始提取正文与其UTF-8 SHA-256；不以去重规范化文本代替原文 |

分区前按NFKC、空白折叠、casefold规范化做比较；同源、全文相同或字符三元组Jaccard≥0.85合并连通组。seed42稳定排序，按组近似80/10/10，每区至少一组；不足3组拒绝。组大小影响文档比例，summary报告实际数量。算法全对比较O(n²)，不声称语义去重或大规模性能已验收。新增原文需建立新版本分区，不重写已锁定测试集。

`samples.jsonl`每行包含：

| 字段 | 约束 |
|---|---|
| `id`、`split`、`batch_id` | 唯一ID；`train/validation/test`；可追踪生成批次 |
| `task_type`、`refusal_type` | 主类`single/multi/citation/refusal`；拒答另填`no_evidence/irrelevant/conflict/missing_conditions` |
| `question`、`answer` | 非空问题和审核答案；可回答类必须引用本轮`[S1]`等，重复问题拒绝 |
| `evidence` | 列表项含`document_id/title/version/location/content`；必须对应同分区、同版本的原文及章节；多片段至少2片 |
| `review` | `status=pending/approved/rejected`；批准记录含非空`reviewer`及带时区`reviewed_at` |

无证据拒答必须为空证据；无关及缺条件须有资料，冲突至少2片。演示修改仅用于冲突题，证据额外保留`original_content/transformation`，修改说明和模型可见正文均标明“演示”，不能冒充官方原文。

每分区必须覆盖全部主类及四种拒答子类；validation/test全量批准，train各层至少向上取整10%批准，任一拒审样例阻止其批次导出。规则校验命令/代码和数字是否见于证据、引用编号、来源、跨区污染、真实token计数及4096预算；这些字面规则不能证明语义正确，也不自动替人工签名。超长样例拒绝，不静默截断答案。

成功后一次发布新目录：`train.jsonl/validation.jsonl/test.jsonl`每行仅含标准`messages`，另有`manifest.jsonl`与`summary.json`。manifest保存分区内从1开始的行号、来源及许可/hash、证据、批次、审核、模板版本、消息及样例hash、真实token数；这些信息不进入system/user。输入或门禁失败不留下部分可消费数据集。

## 共用模板与项目边界

`prompt.py`的`kb-evidence-v1`逐字符对齐平台现行输入：`问题：<问题>`、空行、`编号证据：`，每片采用`[S1] 文档：<标题>；版本：<版本>；出处：<定位>`及换行正文。使用固定中文system约束和原生Qwen聊天模板，空证据保留空区域。该格式修正初始方案的多行出处展示；不复制平台代码。后端负责ACL，模型不接收受限片段或权限标签。

`_token_counter`只加载本地模型文件，`local_files_only=True`、`trust_remote_code=False`，明确启用离线模式。读取实际`input_ids`长度，不把BatchEncoding字段数当token数。核心Python函数允许注入计数器；调用方负责保证它确为真实tokenizer，测试用固定计数回调只验证预算拒绝路径。

核心包包括模板、语料分区、正式数据导出、人工评估、发布及CLI，云端学习脚本位于`scripts/`；`configs/logging.ini`集中配置日志。模型服务复用Swift，不保存真实`.env`或凭证。正式SFT/QLoRA在用户云服务器运行，本机用于资料、结果和后续GGUF演示。沿用Qwen3-4B、原精度合并再量化及正式质量门槛，基座或未经人工审核的学习实验均不等于平台最终验收。

原始需求为工作区只读资料`real/2.9.md`与`real/4.1.md`。正式人工审核、量化发布及平台最终联调仍需按方案验收；云端学习执行记录保存实际采集、权重、训练和推理证据。外部下载须在用户授权范围内执行。

## 评估记录与报告

`evaluation-records.jsonl`每行包含`id/task_type/refusal_type/answerable/evidence_labels/test_case_hash/base/finetuned`。主任务与拒答分类沿用样例schema，编号列表从S1连续分配，证据数量满足对应类型。`test_case_hash`必须是锁定问题、完整证据及预定关键点/拒答基准的规范JSON SHA-256，不是ID或模型输出hash。调用方保留原始用例；工具只能核对hash格式与一致性，不能凭hash证明人工判断正确。

`base`与`finetuned`各包含真实`text`和独立`review`。审核字段为`reviewer/reviewed_at/supported_claims/total_claims/covered_keypoints/total_keypoints/terminology_format/refused/refusal_correct`。前四个计数必须为非负整数且分子不超过分母；关键点总数由锁定用例预先确定，两模型一致。后三个状态为布尔值，拒答正确仅用于不可回答且确实拒答的输出。

评估自动检查引用；没有任何有效本轮引用的事实支持数按0计，可回答题无引用另计引用失败。六指标按方案固定口径计算，报告包含分子/分母、两模型实际比例、差值、能力与拒答子类分组、逐条输出及审核、失败原因。空分母、缺覆盖、忠实性退步或无改善均不发布；对比门槛使用精确整数比例，避免浮点舍入放过未达标计数。`validate_report`从逐条记录重建，不能只改`passed`或聚合比例骗过门禁。

## 发布规格

`release`的规格JSON包含`model_version`、40位不可变`base_revision`、`evaluation_report`、`quantized_evaluation_report`及`artifact_groups`。两个报告必须是发布根目录内不同文件，使用相同`test_set_hash`并分别通过同一门槛。组名固定且每组非空：

| 分组 | 必需文件与职责 |
|---|---|
| `adapter` | LoRA的`.safetensors`和`adapter_config.json` |
| `merged_model` | 从原精度基座合并后的`.safetensors`及相应模型配置/索引 |
| `gguf` | 复测通过的Q4_K_M `.gguf`及实际量化记录 |
| `tokenizer` | tokenizer文件及真实原生chat template，可由`tokenizer_config.json`或`chat_template.jinja`提供 |
| `training_config` | 实际训练配置、锁定工具/环境与运行记录 |
| `data_manifest` | 数据分区、来源及审核清单 |
| `licenses` | 实际基座/原文适用许可、署名及转换说明 |

每组值为相对文件路径列表。工具检查所需后缀/配置和非空文件，不分析权重内部语义，因此仍需真实加载、推理、人工评估及完整产物检查。清单固定`model_name=kb-grounded-qwen3-4b-v1`、`base_model=Qwen/Qwen3-4B-Instruct-2507`，包含平台所需顶层字段和逐文件SHA-256，附模板版本及分组；报告本身也受摘要保护。路径仅允许可移植相对路径，解析后仍限根目录，拒绝Windows盘符/设备名/ADS、目录穿越及重复文件别名；权重分块读取，避免整体读入内存。
