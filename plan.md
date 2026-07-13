# EEG-UI 第一阶段开发计划

## 1. 项目背景

仓库：`https://github.com/softstarssl/EEG-UI`  
本地目录：`EEG-UI/`

本项目计划制作一个面向 EEG 疾病分类研究的可交互界面。用户将某一位受试者的 EEG 文件拖入网页，选择疾病分类任务，点击运行后，由后台模型给出二分类结果和概率。

当前阶段尚未准备好真实 REVE 模型、分类头、checkpoint 和正式数据，因此先完成：

- Gradio 前端；
- EEG 文件上传与基础校验；
- 抑郁症、ADHD 两个独立二分类任务；
- Mock REVE 推理后端；
- 真实 REVE 推理接口占位；
- 日志、错误提示、测试和使用说明。

本阶段必须做到：**没有真实模型、没有 checkpoint、没有训练数据，也可以完整启动和演示。**

---

## 2. 第一阶段目标

实现一个可运行的科研 Demo：

1. 用户选择分类任务：
   - `Depression`：抑郁症二分类；
   - `ADHD`：ADHD 二分类。
2. 用户拖拽或选择一个 EEG 文件。
3. 页面展示文件基本信息。
4. 用户点击 `Run Prediction`。
5. 页面显示运行状态和处理日志。
6. Mock REVE 后端返回：
   - 二分类标签 `0 / 1`；
   - 正类概率；
   - 判定阈值；
   - 当前是否为 Mock 模式。
7. 页面明确标记：
   - 当前结果为模拟结果；
   - 尚未连接真实模型；
   - 结果仅用于科研演示，不构成医学诊断。

---

## 3. 分类任务定义

每一种疾病单独进行二分类，不做三分类。

### 3.1 抑郁症分类

- `0`：未预测为抑郁症，即 Depression Negative；
- `1`：预测为抑郁症，即 Depression Positive。

### 3.2 ADHD 分类

- `0`：未预测为 ADHD，即 ADHD Negative；
- `1`：预测为 ADHD，即 ADHD Positive。

### 3.3 界面用词要求

不要把 `0` 写成“健康人”，因为未被某一疾病模型判为阳性，不代表受试者不存在其他疾病。

推荐显示：

```text
Task: Depression
Prediction: 1
Decision: Depression Positive
Positive-class probability: 82.4%
Threshold: 50.0%
```

或：

```text
Task: ADHD
Prediction: 0
Decision: ADHD Negative
Positive-class probability: 23.1%
Threshold: 50.0%
```

---

## 4. 技术方案

### 4.1 前端

使用 **Gradio**。

原因：

- Python 代码即可完成前端；
- 与后续 PyTorch / REVE 推理代码连接简单；
- 支持文件拖拽、按钮、进度、日志和结果展示；
- 适合当前科研 Demo 阶段。

### 4.2 后端

- Python 3.10 或更高版本；
- Gradio；
- NumPy；
- PyYAML；
- Pydantic 或 Python dataclass；
- pytest；
- 可选使用 MNE 和 SciPy 做 EEG 文件基础读取。

### 4.3 模型后端

当前默认：

```yaml
inference_backend: mock_reve
```

未来支持：

```yaml
inference_backend: real_reve
```

前端不得直接依赖具体模型类，而应通过统一的推理服务接口调用。

---

## 5. 默认支持的文件格式

第一版上传入口默认允许：

```text
.edf
.set
.fif
.mat
.npy
.npz
```

### 5.1 读取策略

- `.edf`：优先使用 MNE；
- `.set`：优先使用 MNE；
- `.fif`：使用 MNE；
- `.mat`：使用 SciPy；
- `.npy`：使用 NumPy；
- `.npz`：使用 NumPy。

### 5.2 第一阶段要求

第一阶段不要求兼容所有来源的 `.mat`、`.npy`、`.npz` 内部结构，因为这几类格式没有统一 schema。

实现原则：

1. 所有默认扩展名都可以上传；
2. 能读取时展示基础元数据；
3. 无法确定内部 EEG 数组、采样率或通道名时，给出清晰提示；
4. Mock 模式仍允许继续演示；
5. 不得伪造真实采样率、通道名或信号形状；
6. 不支持的扩展名必须在运行前拒绝。

后续可根据实际数据集格式增加专用 loader。

---

## 6. 页面布局

建议使用单页布局。

### 6.1 页面顶部

标题：

```text
EEG Disease Classification Demo
```

副标题：

```text
REVE-compatible research prototype
```

显著警告：

```text
Mock mode: no real model is connected.
For research demonstration only. Not a medical diagnosis.
```

### 6.2 输入区域

包含：

1. `Task` 下拉框：
   - Depression
   - ADHD
2. 文件拖拽上传框；
3. `Run Prediction` 按钮；
4. `Reset` 按钮。

### 6.3 文件信息区域

上传后展示：

- 文件名；
- 文件扩展名；
- 文件大小；
- 文件路径或临时路径不必完整暴露；
- 是否通过扩展名校验；
- 若能解析则展示：
  - 信号形状；
  - 通道数；
  - 时间点数；
  - 采样率；
  - 通道名数量。

未知字段显示 `Unknown`，不得凭空填写。

### 6.4 运行状态区域

显示阶段日志，例如：

```text
[1/4] Validating uploaded file...
[2/4] Reading EEG metadata...
[3/4] Running Mock REVE inference...
[4/4] Formatting prediction result...
Done.
```

### 6.5 输出区域

至少显示：

- 当前任务；
- 标签 `0 / 1`；
- Negative / Positive 文本；
- 正类概率；
- 阈值；
- Mock 模式标志；
- 简短免责声明。

可使用：

- Label；
- Markdown；
- JSON；
- 简单概率条形展示。

无需制作复杂图表。

---

## 7. Mock 推理行为

### 7.1 设计目标

Mock 推理只用于打通界面和接口，不代表真实模型能力。

### 7.2 结果稳定性

不要每次点击都完全随机。

建议根据以下内容生成稳定的伪概率：

```text
hash(file content or file name + task name)
```

同一个文件、同一个任务应尽量返回相同结果；切换任务后可以得到不同结果。

### 7.3 输出格式

```python
PredictionResult(
    task="depression",
    label=1,
    positive_probability=0.824,
    threshold=0.5,
    decision="Depression Positive",
    backend="mock_reve",
    is_mock=True,
    message="Mock result. No real model is connected.",
)
```

### 7.4 判定规则

```python
label = int(positive_probability >= threshold)
```

默认阈值：

```yaml
threshold: 0.5
```

阈值必须放在配置文件中，不要散落在 UI 代码里。

---

## 8. 面向 REVE 的接口设计

虽然当前不接入真实模型，但接口必须按后续 REVE 集成设计。

REVE 后续推理预计需要：

- EEG 信号；
- 采样率；
- 通道名；
- 电极三维坐标；
- 选择的疾病任务；
- 对应任务的分类头或 checkpoint。

### 8.1 EEG 数据对象

建议定义：

```python
@dataclass
class EEGRecord:
    file_path: Path
    file_type: str
    signal: np.ndarray | None
    sampling_rate: float | None
    channel_names: list[str]
    coordinates: np.ndarray | None
    metadata: dict[str, Any]
```

其中：

```text
signal shape: [channels, time]
coordinates shape: [channels, 3]
```

第一阶段允许 `signal`、`sampling_rate`、`coordinates` 为空，但必须如实标记。

### 8.2 任务枚举

```python
class DiseaseTask(str, Enum):
    DEPRESSION = "depression"
    ADHD = "adhd"
```

不要在代码中散落 `"Depression"`、`"depression"`、`"MDD"` 等不同字符串。

### 8.3 推理统一接口

```python
class BaseInferenceService(ABC):
    @abstractmethod
    def predict(
        self,
        record: EEGRecord,
        task: DiseaseTask,
    ) -> PredictionResult:
        ...
```

实现：

```python
class MockReveInferenceService(BaseInferenceService):
    ...
```

预留：

```python
class RealReveInferenceService(BaseInferenceService):
    ...
```

### 8.4 真实后端占位行为

`RealReveInferenceService` 当前不加载模型，不下载 checkpoint。

当配置为真实模式但模型不存在时，应抛出清晰错误：

```text
Real REVE backend is not available.
Please provide a model checkpoint and classifier configuration.
```

不得悄悄回退为 Mock 后仍显示真实结果。

---

## 9. REVE 预处理配置占位

当前阶段不要完整实现或强行调用真实 REVE 预处理，但要为后续预留配置。

建议 `configs/default.yaml`：

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
  allowed_extensions:
    - ".edf"
    - ".set"
    - ".fif"
    - ".mat"
    - ".npy"
    - ".npz"
  max_size_mb: 1024

reve:
  checkpoint_path: null
  model_size: null
  target_sampling_rate: 200
  bandpass_low_hz: 0.5
  bandpass_high_hz: 99.5
  zscore: true
  clip_std: 15.0
  window_seconds: 10
  window_overlap_seconds: 0
  patch_seconds: 1
```

这些参数只是后续默认配置，不代表最终下游分类必须原样使用。真实模型接入时应以模型负责人的训练配置为准。

---

## 10. 推荐目录结构

```text
EEG-UI/
├── app.py
├── plan.md
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── src/
│   └── eeg_ui/
│       ├── __init__.py
│       ├── config.py
│       ├── schemas.py
│       ├── ui/
│       │   ├── __init__.py
│       │   └── gradio_app.py
│       ├── io/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── registry.py
│       │   ├── mne_loader.py
│       │   ├── mat_loader.py
│       │   └── numpy_loader.py
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   └── reve_preprocessor.py
│       └── inference/
│           ├── __init__.py
│           ├── base.py
│           ├── factory.py
│           ├── mock_reve.py
│           └── real_reve.py
└── tests/
    ├── test_config.py
    ├── test_file_validation.py
    ├── test_mock_inference.py
    └── test_inference_factory.py
```

如果 Codex 判断第一版不需要拆出过多 loader 文件，可以适当合并，但必须保持以下三层解耦：

```text
UI
EEG I/O / preprocessing
Inference service
```

---

## 11. 各模块职责

### 11.1 `app.py`

只负责：

- 加载配置；
- 创建 Gradio 应用；
- 启动服务。

不得包含具体模型推理代码。

### 11.2 `schemas.py`

定义：

- `DiseaseTask`；
- `EEGRecord`；
- `PredictionResult`；
- 必要的文件元数据结构。

### 11.3 `io/`

负责：

- 文件扩展名校验；
- 文件大小校验；
- 按格式选择 loader；
- 尽力读取元数据；
- 返回统一的 `EEGRecord`。

### 11.4 `preprocessing/`

当前只提供接口和配置读取，可实现最小的 shape 检查。

不得在 Mock 阶段假装已经完成完整 REVE 预处理。

### 11.5 `inference/`

负责：

- 统一推理接口；
- 根据配置创建 Mock 或 Real 后端；
- 生成标准化结果。

### 11.6 `ui/`

负责：

- Gradio 组件；
- 用户事件；
- 进度和日志；
- 将输入转换为服务调用；
- 将结果格式化展示。

UI 不直接读取 checkpoint，不直接创建 PyTorch 模型。

---

## 12. 错误处理

需要覆盖：

1. 未上传文件时点击运行；
2. 未选择任务；
3. 不支持的文件扩展名；
4. 文件为空；
5. 文件超过限制；
6. EEG 文件损坏；
7. loader 无法确定内部数据结构；
8. 推理服务异常；
9. 配置为真实后端但 checkpoint 不存在；
10. 用户重复点击运行。

错误信息应使用普通用户可以理解的语言，不直接把完整 Python traceback 显示在页面上。

开发日志中可以保留 traceback。

---

## 13. 测试要求

至少实现以下测试。

### 13.1 文件校验

- 支持的扩展名通过；
- 不支持的扩展名被拒绝；
- 大小写扩展名正常处理；
- 空文件被拒绝。

### 13.2 Mock 推理

- 返回字段完整；
- 概率位于 `[0, 1]`；
- 标签只可能为 `0` 或 `1`；
- 标签符合阈值规则；
- 相同文件和任务的结果稳定；
- Depression 和 ADHD 可以产生不同结果；
- `is_mock` 必须为 `True`。

### 13.3 后端工厂

- `mock_reve` 返回 Mock 服务；
- `real_reve` 返回真实服务占位；
- 未知 backend 给出清晰错误。

### 13.4 配置

- 默认配置可以正常加载；
- 缺失可选字段时使用默认值；
- 非法阈值被拒绝。

### 13.5 基本启动检查

至少保证：

```bash
pytest
python app.py
```

可以执行。

若环境允许，可增加 Gradio 应用创建的 smoke test，但不要让测试长期占用端口。

---

## 14. README 要求

README 至少包括：

1. 项目用途；
2. 当前阶段说明；
3. Mock 模式警告；
4. 环境安装；
5. 启动命令；
6. 支持的文件格式；
7. 抑郁症和 ADHD 是两个独立二分类任务；
8. 目录结构；
9. 如何切换推理 backend；
10. 后续如何接入真实 REVE；
11. 医学免责声明。

建议启动方式：

```bash
git clone https://github.com/softstarssl/EEG-UI.git
cd EEG-UI

python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

macOS / Linux：

```bash
source .venv/bin/activate
```

安装并启动：

```bash
pip install -r requirements.txt
python app.py
```

---

## 15. `.gitignore` 要求

至少忽略：

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.env
.DS_Store

data/
checkpoints/
models/
outputs/
logs/

*.ckpt
*.pt
*.pth
*.safetensors
```

不要把真实 EEG 数据、模型权重或本地环境文件提交到仓库。

---

## 16. 第一阶段明确不做的内容

当前不要实现：

- 下载 REVE 模型；
- 下载任何大型 EEG 数据集；
- 训练模型；
- 微调模型；
- 真实疾病预测；
- 真实临床诊断；
- 自动生成或伪造电极坐标；
- 用户账户系统；
- 数据库；
- 云端部署；
- 多用户并发优化；
- 复杂权限系统；
- 复杂可解释性图表；
- 自动上传患者数据到外部服务。

本阶段所有处理应在本地运行。

---

## 17. 实施顺序

### Phase 1：初始化项目

- 创建 Python 项目目录；
- 添加 `requirements.txt`；
- 添加 `.gitignore`；
- 添加配置文件；
- 确保 `python app.py` 能显示空页面。

### Phase 2：定义数据结构和配置

- 实现任务枚举；
- 实现 EEGRecord；
- 实现 PredictionResult；
- 实现 YAML 配置加载与校验。

### Phase 3：文件上传与校验

- 实现扩展名检查；
- 实现文件大小检查；
- 实现 loader registry；
- 尽力提取基础元数据；
- 无法解析时给出清晰提示。

### Phase 4：Mock REVE 后端

- 定义统一推理接口；
- 实现稳定的 Mock 概率；
- 实现后端工厂；
- 实现 Real REVE 占位类。

### Phase 5：Gradio 页面

- 任务选择；
- 文件拖拽；
- 文件信息展示；
- 运行与重置按钮；
- 进度日志；
- 结果卡片；
- Mock 警告；
- 医学免责声明。

### Phase 6：测试与文档

- 编写 pytest；
- 编写 README；
- 完成手动运行测试；
- 检查 Mock 标志是否始终可见；
- 检查没有隐式下载行为。

---

## 18. 第一阶段验收标准

以下条件全部满足才算完成：

- [ ] 空仓库中生成完整项目结构；
- [ ] 安装依赖后可以运行 `python app.py`；
- [ ] 页面支持拖拽 EEG 文件；
- [ ] 默认允许 `.edf/.set/.fif/.mat/.npy/.npz`；
- [ ] 可选择 Depression 或 ADHD；
- [ ] 点击运行后出现进度和日志；
- [ ] 返回标签 `0/1`；
- [ ] 返回正类概率和阈值；
- [ ] 同一文件和同一任务的 Mock 结果稳定；
- [ ] 页面醒目标记 Mock 模式；
- [ ] 页面包含科研用途和非医学诊断声明；
- [ ] 没有真实模型也可运行；
- [ ] 不会自动下载模型或数据；
- [ ] UI、I/O、推理层解耦；
- [ ] 已预留真实 REVE 推理类；
- [ ] 测试通过；
- [ ] README 可指导新用户从零启动。

---

## 19. 后续真实 REVE 接入时需要模型负责人提供的信息

后续阶段再确认：

1. 使用 REVE Small、Base 还是 Large；
2. 官方 checkpoint 还是自行训练；
3. Depression 和 ADHD 是否分别使用独立 checkpoint；
4. 是共享 REVE encoder 加两个分类头，还是两个完整模型；
5. 模型输入片段长度；
6. 滑动窗口大小和重叠；
7. 训练时采样率；
8. 训练时滤波参数；
9. 归一化方式；
10. 支持的通道集合；
11. 电极坐标来源；
12. 缺失通道处理策略；
13. 多片段如何汇总到受试者级结果；
14. 分类阈值如何确定；
15. checkpoint 格式；
16. 推理设备要求；
17. 模型输出 logits、概率还是 embedding；
18. 是否需要置信区间或不确定性估计。

真实模型接口确定后，应尽量只修改：

```text
src/eeg_ui/preprocessing/reve_preprocessor.py
src/eeg_ui/inference/real_reve.py
configs/default.yaml
```

不要重写 Gradio 前端。

---

## 20. 给 Codex / Claude Code 的施工要求

执行本计划时：

1. 先阅读完整 `plan.md`；
2. 检查仓库是否为空；
3. 给出不超过十条的简短实施步骤；
4. 直接开始实现，不要等待真实模型或数据；
5. 每完成一个阶段就运行相关测试；
6. 不要自动下载大型依赖、数据或 checkpoint；
7. 不得把 Mock 输出描述为真实疾病预测；
8. 优先保证代码结构清楚、可替换和可测试；
9. 避免过度设计；
10. 最终运行全部测试并汇报：
    - 创建了哪些文件；
    - 如何启动；
    - 当前 Mock 限制；
    - 后续接入真实 REVE 应修改哪些文件。
