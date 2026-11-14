# 知识库模型微调

面向知识库证据问答的开源模型微调初始化与实现方案。领域固定为中文技术支持与操作知识库。

**当前仅完成初始化与实现方案，业务功能尚未开发。** 没有可启动业务服务、前端控制台、数据集、已训练模型或业务验收结果；训练和推理栈仅为后续设计，未安装。

本项目对应 `real/4.1.md` 的术语格式、忠实回答、多片段整合、引用和拒答能力需求，为独立项目 `knowledge-base-platform` 交付生成模型；平台的权限、检索与会话仍由平台实现。 两项目通过 OpenAI 兼容服务契约协作，不复制代码或导入兄弟目录。微调模型发布前，基座对照实验不计为最终联调成功。

详细技术栈、架构、需求覆盖、实施顺序、统一契约与未来验收见 [实现方案](docs/implementation-plan.md)。

## 初始化与验证

使用 Python 3.11.x、uv；本次选定已安装的 3.11.16，PEP 621 元数据声明无业务依赖，`uv.lock` 保留在版本管理中。以下命令在项目目录执行，明确使用离线模式并禁止下载 Python，不安装包：

```bash
uv --offline --no-python-downloads venv --python D:/uv/python/cpython-3.11.16-windows-x86_64-none/python.exe .venv
uv --offline --no-python-downloads lock --python D:/uv/python/cpython-3.11.16-windows-x86_64-none/python.exe
.venv/Scripts/python.exe -B -c 'import sys,tomllib; from pathlib import Path; p=tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8")); assert p["project"]["name"]=="knowledge-model-finetuning"; assert p["project"]["dependencies"]==[]; sys.path.insert(0,"src"); import knowledge_model_finetuning; assert knowledge_model_finetuning.__doc__=="知识库模型微调项目包。"; print("knowledge-model-finetuning 初始化验证通过")'
```

其他机器需将解释器路径替换为当地已装的 Python 3.11.x；目标版本见 `.python-version`。命令仅证明环境、项目元数据与包导入，不证明业务功能。包使用 `src/` 布局，本次没有构建后端且 `[tool.uv] package = false`，验证显式加入 `src`，不宣称已安装为分发包。不运行可能安装依赖的 `uv run` 或 `uv sync`。

## 当前文件

- `pyproject.toml`、`.python-version`、`uv.lock`：项目元数据及离线环境解析信息。
- `src/knowledge_model_finetuning/__init__.py`：仅中文包文档字符串。
- `docs/implementation-plan.md`：后续实施设计与验收规范。
- `.gitignore`：隔离本地环境、凭证、数据、模型及输出；锁文件可跟踪。

当前无需运行服务环境变量，不生成 `.env` 或 `.env.example`。模型服务配置、独立日志配置及业务依赖在后续实现时按方案加入。原始需求位于工作区只读资料 `real/2.9.md` 与 `real/4.1.md`，本项目不复制其他项目文件。
