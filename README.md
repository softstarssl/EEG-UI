# EEG 疾病分类研究演示界面

> **研究原型 / Mock 模式**：本项目当前没有接入真实 REVE 模型、疾病分类头、模型权重（checkpoint）或正式训练数据。页面展示的预测值是为打通交互流程而生成的稳定模拟结果，不能解释为真实疾病预测，也绝不能用于医学诊断或临床决策。

`EEG-UI` 是一个本地运行的 Gradio 应用，用于演示 EEG 疾病分类研究中的完整交互链路：上传一位受试者的 EEG 文件、选择分类任务、校验并读取基础元数据、调用推理服务，以及呈现标签、概率、日志和免责声明。

目前支持两个**彼此独立**的二分类任务：

- `Depression`：抑郁症二分类；
- `ADHD`：注意缺陷多动障碍（ADHD）二分类。

项目的目标是为后续接入真实 REVE 编码器和分类器保留清晰、可测试的接口，而不是训练或发布医学模型。

## 目录

- [项目状态与边界](#项目状态与边界)
- [功能概览](#功能概览)
- [环境安装与启动](#环境安装与启动)
- [页面使用流程](#页面使用流程)
- [分类结果的正确理解](#分类结果的正确理解)
- [支持的文件格式与读取策略](#支持的文件格式与读取策略)
- [配置说明](#配置说明)
- [Mock 推理的工作方式](#mock-推理的工作方式)
- [错误提示与排查](#错误提示与排查)
- [项目结构与模块职责](#项目结构与模块职责)
- [测试](#测试)
- [接入真实 REVE 的准备事项](#接入真实-reve-的准备事项)
- [数据与医学免责声明](#数据与医学免责声明)

## 项目状态与边界

这是第一阶段的本地科研 Demo。它已经实现上传、校验、元数据展示、Mock 推理、结果展示和测试；但下列功能明确**不在当前阶段范围内**：

| 当前已具备 | 当前不具备 |
| --- | --- |
| Gradio 单页交互界面 | 真实 REVE 模型推理 |
| Depression / ADHD 两个独立二分类入口 | 模型训练、微调或评估 |
| 可替换的推理服务接口 | checkpoint 下载或加载 |
| 文件扩展名、空文件和大小校验 | 自动下载 EEG 数据集 |
| 尽力读取部分文件的基础元数据 | 临床诊断、患者管理、数据库或云端部署 |
| 稳定的 Mock 结果和日志 | 自动生成或伪造电极三维坐标 |

默认配置只监听本机回环地址 `127.0.0.1`，且 `share: false`。项目不会自行上传数据、下载模型、下载 checkpoint 或下载 EEG 数据集。

## 功能概览

- 支持拖拽或选择 `.edf`、`.set`、`.fif`、`.mat`、`.npy`、`.npz` 文件；
- 运行前校验扩展名、文件存在性、空文件和文件大小；默认大小上限为 `1024 MB`；
- 上传后展示文件名、扩展名、大小、校验状态、信号形状、通道数、时间点、采样率和通道名数量；不能可靠获取时显示 `Unknown`；
- 尽力解析可用的基础元数据，但不会编造采样率、通道名、信号含义或电极坐标；
- 提供 `Depression`、`ADHD` 两个独立任务，并输出 `0/1` 标签、Positive/Negative 判定、正类概率和阈值；
- 全程显示运行阶段日志、Mock 标志和医学免责声明；
- Mock 与 Real REVE 后端均通过统一接口创建，真实后端未配置时会明确失败，不会静默伪装为 Mock 结果；
- 提供配置、文件校验、推理工厂、Mock 稳定性和 UI 回调的自动化测试。

## 环境安装与启动

### 1. 前置条件

- Python `3.10` 或更高版本；
- `pip`；
- 建议使用独立虚拟环境；
- 启动 Web 界面时需要安装 `requirements.txt` 中的依赖。

主要依赖包括 Gradio、NumPy、PyYAML、SciPy、MNE 和 pytest。MNE/SciPy 用于尽力读取部分 EEG 文件的基础元数据；即便某个文件无法解析，Mock 演示流程仍可继续。

### 2. 创建环境

```bash
git clone https://github.com/softstarssl/EEG-UI.git
cd EEG-UI
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

Windows CMD：

```bat
.venv\Scripts\activate.bat
```

macOS / Linux：

```bash
source .venv/bin/activate
```

如果 Windows PowerShell 因本地执行策略拒绝激活脚本，可在当前终端临时允许：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 3. 安装与运行

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

正常启动后，在浏览器打开终端显示的地址；默认是：

```text
http://127.0.0.1:7860
```

按 `Ctrl+C` 停止本地服务。

> 若出现 `Gradio is not installed`，说明当前 Python 环境尚未执行依赖安装，运行 `python -m pip install -r requirements.txt` 后重新启动即可。

## 页面使用流程

1. 启动应用后，页面顶部会显示 **Mock mode: no real model is connected**，这是预期行为。
2. 在 `Task` 下拉框中选择 `Depression` 或 `ADHD`。
3. 将一个 EEG 文件拖入上传框，或点击上传框选择本地文件。
4. 查看右侧 `File information`：系统会显示已知元数据和无法确定的字段。此处不展示完整临时路径。
5. 点击 `Run Prediction`。
6. 在 `Run status and log` 中查看四个阶段：文件校验、元数据读取、推理调用、结果格式化。
7. 在 `Prediction result` 中查看任务、标签、判定、正类概率、阈值和后端；结果区域会持续明确标记当前是 Mock 模式。
8. 需要开始新一次演示时，点击 `Reset` 清空文件、日志和结果。

应用对运行请求设置了单次并发限制，重复点击会排队而不是并行执行同一份推理流程。

### 文件信息字段

| 字段 | 含义 |
| --- | --- |
| `File name` | 上传文件的文件名，不显示完整临时路径。 |
| `Extension` | 规范化为小写的文件扩展名。 |
| `File size` | 上传文件的实际大小。 |
| `Extension validation` | 是否通过允许扩展名校验。 |
| `Signal shape` | 若读取器能安全获得数组/原始记录的形状，则显示为 `(channels, time)`；否则为 `Unknown`。 |
| `Channels` / `Time points` | 仅在已有二维形状时显示。 |
| `Sampling rate` | 读取器明确获取到的采样率；不会凭空填充。 |
| `Channel-name count` | 读取器明确获取到的通道名数量；未知时为 `Unknown`。 |

## 分类结果的正确理解

两个任务是**分别运行的二分类器接口**，不是三分类，也不会在同一次结果中比较两种疾病。

| 任务 | `0` | `1` |
| --- | --- | --- |
| `Depression` | `Depression Negative` | `Depression Positive` |
| `ADHD` | `ADHD Negative` | `ADHD Positive` |

结果字段说明：

| 字段 | 含义 |
| --- | --- |
| `Task` | 本次选择的任务。 |
| `Prediction` | 二值标签 `0` 或 `1`。 |
| `Decision` | 与标签对应的 `Negative` 或 `Positive` 文本。 |
| `Positive-class probability` | 当前后端给出的正类概率；在当前版本中为 Mock 概率。 |
| `Threshold` | 二值化阈值，默认 `50.0%`。 |
| `Backend` | 当前推理后端，例如 `mock_reve`。 |

例如，`Depression Negative` 的含义只是“抑郁症任务没有得到阳性标签”。它**不**代表受试者是健康人，也不代表其不存在其他疾病、共病或任何临床风险。

## 支持的文件格式与读取策略

上传入口允许以下扩展名（扩展名大小写不敏感）：

| 格式 | 尝试使用的读取方式 | 当前可可靠展示的内容 | 限制 |
| --- | --- | --- | --- |
| `.edf` | MNE `read_raw_edf` | 通道数、时间点、采样率、通道名（读取成功时） | 损坏、非兼容或依赖缺失时仅显示清晰提示，Mock 仍可运行。 |
| `.set` | MNE `read_raw_eeglab` | 通道数、时间点、采样率、通道名（读取成功时） | 某些 EEGLAB 数据还依赖配套 `.fdt`，需要文件结构完整。 |
| `.fif` | MNE `read_raw_fif` | 通道数、时间点、采样率、通道名（读取成功时） | 仅尝试元数据读取，不在此阶段执行 REVE 预处理。 |
| `.mat` | SciPy `loadmat` | MATLAB 变量名（读取成功时） | `.mat` 没有统一 EEG schema；不会猜测哪个变量是信号或采样率。 |
| `.npy` | NumPy `load` | 数组形状；若恰为二维，会作为候选 `[channels, time]` 数组保留 | 二维数组的 EEG 语义、采样率和通道名仍未验证。 |
| `.npz` | NumPy `load` | 压缩包中的数组名和形状（读取成功时） | 不会擅自选择其中某个数组作为 EEG 信号。 |

这意味着“文件扩展名被允许上传”不等于“已经通用、无损地理解该格式内部的 EEG 数据”。对于无统一 schema 的 `.mat/.npy/.npz`，应在真实模型接入阶段为实际数据集实现专用 loader。

### 文件校验规则

- 只允许配置文件中列出的扩展名；不支持的扩展名会在运行前拒绝；
- 扩展名会转换为小写后再比较，因此 `record.NPY` 与 `record.npy` 都可通过；
- 空文件会被拒绝；
- 默认最大大小为 `1024 MB`；
- 文件在上传后若不可访问，页面会要求重新上传；
- 元数据解析失败不会泄露 Python traceback 到页面，且在 Mock 模式下仍可继续演示。

## 配置说明

默认配置文件为 [`configs/default.yaml`](configs/default.yaml)。关键配置如下：

```yaml
app:
  title: "EEG Disease Classification Demo"
  host: "127.0.0.1"
  port: 7860
  share: false

inference:
  backend: "mock_reve"
  threshold: 0.5
  device: "auto"

files:
  allowed_extensions: [".edf", ".set", ".fif", ".mat", ".npy", ".npz"]
  max_size_mb: 1024
```

### 配置字段

| 配置 | 作用 | 当前注意事项 |
| --- | --- | --- |
| `app.host` / `app.port` | Gradio 本地服务地址与端口 | 默认仅绑定 `127.0.0.1:7860`。 |
| `app.share` | 是否启用 Gradio 分享链接 | 默认 `false`，应在处理真实 EEG 前谨慎评估隐私和合规性。 |
| `inference.backend` | 推理服务类型 | 只能使用 `mock_reve` 或预留的 `real_reve`。 |
| `inference.threshold` | 正类判定阈值 | 必须在 `[0, 1]` 内，默认 `0.5`。 |
| `files.allowed_extensions` | 可上传扩展名列表 | 每个扩展名都必须以 `.` 开头。 |
| `files.max_size_mb` | 单文件大小上限 | 必须为正数。 |
| `reve.*` | 后续真实 REVE 预处理的占位配置 | 当前不触发真实预处理或模型加载。 |

若将 `backend` 改成 `real_reve`，应用会调用真实后端占位类，并返回：

```text
Real REVE backend is not available.
Please provide a model checkpoint and classifier configuration.
```

这是有意设计：系统不会在真实后端不可用时私自回退到 Mock 并将结果标为真实推理。

## Mock 推理的工作方式

`mock_reve` 的职责只有一个：在没有模型、权重和训练数据时，验证 UI 与服务接口是否能够端到端工作。它不使用 EEG 信号进行学习或预测。

当前 Mock 概率由以下内容经 SHA-256 生成：

```text
文件名 + 文件大小 + 任务名
```

因此：

- 同一文件名、相同文件大小和相同任务，会得到相同的模拟概率；
- 切换 `Depression` 与 `ADHD` 会改变任务名，通常会产生不同的概率；
- 内容不同但文件名和大小同时相同的两个文件，也可能得到相同 Mock 结果；这是演示用确定性规则，不是模型特征；
- 标签的计算方式为 `label = int(positive_probability >= threshold)`；
- 输出始终带有 `backend="mock_reve"`、`is_mock=True` 和“没有连接真实模型”的消息。

请勿将 Mock 概率用于数据分析、患者筛查、模型对比或任何医学解释。

## 错误提示与排查

| 页面提示或现象 | 可能原因 | 处理方式 |
| --- | --- | --- |
| `Upload an EEG file before running prediction.` | 未上传文件就点击了运行。 | 上传一个允许格式的非空文件。 |
| `Unsupported file type` | 文件扩展名不在允许列表。 | 使用允许格式，或在 `allowed_extensions` 中增加并实现相应 loader。 |
| `The uploaded file is empty` | 上传的文件大小为 0。 | 重新导出或选择有效文件。 |
| 文件超过大小上限 | 文件大于 `max_size_mb`。 | 调整配置中的上限，或使用更小文件；注意本机资源与隐私风险。 |
| 元数据无法读取 | 文件损坏、格式不兼容、缺少 MNE/SciPy 或无统一内部 schema。 | 检查原始文件及依赖；Mock 演示仍可继续。真实接入时实现数据集专用 loader。 |
| `Real REVE backend is not available` | 配置了 `real_reve`，但尚未实现/提供模型。 | 提供 checkpoint、分类头和模型配置后实现 `real_reve.py`。 |
| `Gradio is not installed` | 当前环境未安装运行依赖。 | 激活虚拟环境并执行 `python -m pip install -r requirements.txt`。 |
| 端口无法启动 | 默认端口 `7860` 被占用。 | 修改 `configs/default.yaml` 的 `app.port`，然后重新启动。 |

开发者日志可以保留异常信息；面向使用者的页面不会展示完整 Python traceback。

## 项目结构与模块职责

```text
EEG-UI/
├── app.py                         # 加载配置、创建并启动 Gradio 应用
├── configs/
│   └── default.yaml               # 应用、文件、推理和 REVE 占位配置
├── src/eeg_ui/
│   ├── config.py                  # YAML 配置加载和校验
│   ├── schemas.py                 # 任务、EEGRecord、PredictionResult 等数据结构
│   ├── ui/
│   │   └── gradio_app.py          # 页面组件、回调、日志和结果格式化
│   ├── io/
│   │   ├── base.py                # 文件扩展名、大小、空文件校验
│   │   ├── registry.py            # 按扩展名选择读取器
│   │   ├── mne_loader.py          # EDF/SET/FIF 基础元数据读取
│   │   ├── mat_loader.py          # MATLAB 变量信息读取
│   │   └── numpy_loader.py        # NPY/NPZ 基础信息读取
│   ├── preprocessing/
│   │   └── reve_preprocessor.py   # 真实 REVE 输入完整性检查占位
│   └── inference/
│       ├── base.py                # 统一推理抽象接口
│       ├── factory.py             # 依据配置创建后端
│       ├── mock_reve.py           # 稳定 Mock 推理
│       └── real_reve.py           # 真实后端占位与明确错误
├── tests/                         # pytest 自动化测试
├── requirements.txt
├── .gitignore
└── plan.md                        # 第一阶段实施计划与验收标准
```

依赖关系保持为三层：

```text
Gradio UI  →  EEG I/O / 元数据读取  →  推理服务接口
```

页面不直接读取 checkpoint 或创建 PyTorch 模型；真实模型逻辑应集中在推理和预处理层，以避免重写前端。

## 测试

在已激活、依赖已安装的环境中执行：

```bash
python -m pytest -q
```

测试覆盖范围：

- 默认配置加载、可选字段默认值与非法阈值拒绝；
- 支持/不支持扩展名、大小写扩展名和空文件校验；
- Mock 输出字段、概率范围、标签阈值规则和同一输入的稳定性；
- Mock、Real 占位和未知后端的工厂行为；
- 页面回调的日志、Mock 标志与文件信息；
- 安装 Gradio 后的应用创建 smoke test（不长期占用端口）。

如只修改了配置、I/O 或推理逻辑，请在提交前至少运行一次完整测试集。

## 接入真实 REVE 的准备事项

真实模型接入应尽量集中修改以下文件：

- `src/eeg_ui/preprocessing/reve_preprocessor.py`：实现与训练时一致的输入检查、重采样、滤波、归一化、分窗和通道处理；
- `src/eeg_ui/inference/real_reve.py`：加载模型与分类头，执行推理并返回标准化 `PredictionResult`；
- `configs/default.yaml`：填写已确认的 checkpoint、模型大小、设备和真实预处理参数；
- 如实际数据格式有专有字段，则在 `src/eeg_ui/io/` 增加对应的数据集 loader。

在开始实现前，应由模型负责人明确提供：

1. 使用的 REVE 版本（Small / Base / Large）及 encoder 权重来源；
2. Depression 和 ADHD 是共享 encoder 配独立分类头，还是两套完整模型；
3. checkpoint 格式、路径、加载方式和设备要求；
4. 训练时的采样率、滤波参数、归一化方式、截断规则；
5. 输入窗口长度、重叠、patch 策略和多窗口到受试者级结果的汇总规则；
6. 支持的通道集合、通道重排策略、缺失通道策略；
7. 电极三维坐标来源及坐标系；
8. 模型输出是 logits、概率还是 embedding，以及分类头的输入输出定义；
9. 各任务的标签定义、阈值来源及是否需要不确定性估计；
10. 真实推理的验证集指标、适用人群和已知限制。

`EEGRecord` 已为未来接口预留 `signal`、`sampling_rate`、`channel_names`、`coordinates` 和元数据字段。真实 REVE 输入通常至少需要可靠的二维信号 `[channels, time]`、采样率、通道名和电极坐标；当前阶段不会伪造这些缺失信息。

## 数据与医学免责声明

- 本项目仅用于科研界面、工程接口和 Mock 流程演示；
- 不可用于临床诊断、筛查、治疗建议、紧急医疗用途或任何个体层面的医学决定；
- 当前所有概率、标签和 Positive/Negative 文本都是 Mock 输出，不反映受试者的真实健康状况；
- 使用真实 EEG 数据前，请自行确认知情同意、数据脱敏、访问控制、存储安全、机构审批和适用法律法规；
- 只有在模型、数据、验证、偏倚评估、性能边界和监管要求都得到充分确认后，才应讨论真实模型的研究用途；即使如此，也不等同于临床可用性。
