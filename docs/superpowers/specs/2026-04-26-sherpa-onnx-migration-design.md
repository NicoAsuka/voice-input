# sherpa-onnx 迁移与本地引擎重构设计

**Date:** 2026-04-26
**Status:** Draft
**Reference:** [xifan2333/fcitx5-vinput](https://github.com/xifan2333/fcitx5-vinput)

## 概述

把现有的 Whisper / SenseVoice 双本地引擎完全替换为 **sherpa-onnx**，并借鉴参考项目 fcitx5-vinput 的核心设计——异步预加载、配置签名 + reload worker、Session 抽象、VAD 集成、模型管理子系统、场景化后处理。

### 解决的核心问题

- "STT backend is still starting. Try again after it is ready" 错误：用户在模型还在加载时按热键会失败
- Whisper medium 模型 1.5 GB，加载耗时几十秒
- 切换引擎/模型时体验生硬
- LLM 后处理 prompt 单一，无法针对不同场景定制

### 不在范围内（YAGNI）

- ❌ 独立 daemon 进程 + D-Bus IPC（参考项目用这个，但 sherpa-onnx 加载快，对单进程足够）
- ❌ 指令模式（选中文本后语音执行修改）
- ❌ CLI 工具
- ❌ 流式（streaming）识别——先做 offline buffered，capabilities 字段预留扩展

### 核心约束

- 单进程架构，沿用现有 `systemctl --user` 部署方式
- 完全删除 Whisper / SenseVoice 相关代码与依赖
- 沿用 asyncio 编程模型
- 保持云 backend（VolcEngine、Google、OpenAI Whisper）兼容

---

## §1. 总体架构与模块划分

### 目录结构变更

```
src/voice_input/
├── backends/
│   ├── base.py            # 重写：TranscriptionBackend → Session 接口
│   ├── registry.py        # 新增：BackendRegistry（异步 reload + signature 比对）
│   ├── sherpa_backend.py  # 新增：替代 LocalBackend
│   ├── volcengine_speech.py  # 适配新接口
│   ├── google_speech.py      # 适配新接口
│   ├── openai_whisper.py     # 适配新接口
│   └── local/             # 删除整个目录
├── asr/                   # 新增子包
│   ├── model_manager.py
│   ├── vad.py
│   └── session.py
├── postprocess/           # 新增子包（升级现有 llm.py）
│   ├── llm.py             # 从根目录移入
│   ├── scene.py
│   └── pipeline.py
├── app.py                 # 适配 BackendRegistry，启动时触发异步预加载
├── config.py              # 新增字段：scenes、active_scene、sherpa 配置
└── ...
```

### 模块职责

| 模块 | 职责 | 主要依赖 |
|---|---|---|
| `BackendRegistry` | 管理所有 backend 实例；启动时异步 init；配置变化时后台 reload；提供 `get_session()` | `TranscriptionBackend`, `signature` |
| `SherpaBackend` | 实现 sherpa-onnx 本地识别；管理模型生命周期；可选 VAD 预处理 | `sherpa_onnx`, `ModelManager`, `VadTrimmer` |
| `ModelManager` | 模型从 HuggingFace 下载、SHA256 校验、缓存到 `~/.cache/voice-input/sherpa-models/<id>/` | `httpx` |
| `VadTrimmer` | 加载 Silero VAD ONNX；输入 PCM 输出去静音后的 PCM | `sherpa_onnx`（VAD） |
| `Session` | 一次识别会话：`push_audio()` → `finish()` → `final_text` | - |
| `ScenePipeline` | 根据 active_scene 选择 prompt + 应用 LLM 后处理 | `LLMRefiner` |

### 删除清单

- `backends/local/__init__.py`（LocalBackend）
- `backends/local/whisper_engine.py`
- `backends/local/sensevoice_engine.py`
- `backends/local/engine.py`
- `whisper_worker.py`
- `pyproject.toml` 的 `whisper`、`sensevoice` 可选依赖组
- `tests/test_local_backend.py`

---

## §2. 核心抽象接口（Session 模式）

参考项目的核心设计：**Backend 创建 Session，Session 驱动一次识别**。这个分离让"长生命周期的模型"和"短生命周期的识别"解耦。

```python
# backends/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class BackendCapabilities:
    supports_streaming: bool = False
    requires_network: bool = False
    supports_vad: bool = False

@dataclass(frozen=True)
class BackendDescriptor:
    backend_id: str          # "sherpa", "volcengine", "google", "openai-whisper"
    model_id: str            # 模型标识，用于显示
    capabilities: BackendCapabilities

class Session(ABC):
    """单次识别会话。生命周期：create → push_audio* → finish → final_text"""

    @abstractmethod
    def push_audio(self, pcm_int16: np.ndarray) -> None: ...

    @abstractmethod
    async def finish(self) -> str:
        """终止录音、等待识别完成、返回最终文本。可能抛 RecognitionError。"""

    @abstractmethod
    def cancel(self) -> None: ...

class TranscriptionBackend(ABC):
    """长生命周期，持有模型/连接。"""

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    def describe(self) -> BackendDescriptor: ...

    @abstractmethod
    def create_session(self, language: str) -> Session: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    def is_ready(self) -> bool:
        """同步快速检查，不阻塞。BackendRegistry 用它判断当前是否可用。"""
        return True
```

### 关键设计点

1. **`is_ready()` 同步方法**——解决 "STT backend is still starting" 问题。app 在调用 `create_session()` 之前先检查 `is_ready()`，未就绪则给用户友好提示。
2. **Session 异步 finish**——网络 backend 天然异步；本地 backend 的 finish 也包装为 async（即使内部是同步推理，让 UI 不卡）。
3. **`push_audio` 同步**——录音回调里调用，不能 await。本地 backend 直接缓冲；流式 backend 推到内部队列。
4. **`describe()` 同步**——用于 UI 显示当前 backend 信息、日志记录。

---

## §3. BackendRegistry —— 异步预加载与 reload worker

最核心的优化。借鉴参考项目的 `RecognitionSessionManager`，用 Python asyncio 实现。

### 状态模型

```python
@dataclass
class _Effective:
    backend: TranscriptionBackend
    descriptor: BackendDescriptor
    signature: str  # 配置指纹

class BackendRegistry:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._effective: _Effective | None = None     # 当前在用
        self._target_signature: str | None = None     # 用户期望
        self._reload_task: asyncio.Task | None = None # 正在准备的新 backend
        self._lock = asyncio.Lock()
        self._listeners: list[Callable[[RegistryState], None]] = []
        self._last_error: str | None = None
```

### 核心方法

| 方法 | 作用 | 调用方 |
|---|---|---|
| `start()` | 启动时调用一次：触发首次异步加载，立即返回 | `app.py` 启动时 |
| `synchronize()` | 配置变化时调用：算 signature，不变就 no-op；变了就调度 reload worker | settings_dialog 保存配置后 |
| `is_ready() -> bool` | 同步快速检查 effective 是否存在 | 录音热键回调 |
| `create_session(language)` | 同步返回 Session（必须 ready 才能调用） | 录音开始时 |
| `current_descriptor()` | 当前 effective backend 的描述 | 系统托盘菜单显示 |
| `add_state_listener(cb)` | 订阅 ready 状态变化 | UI 更新托盘图标 |

### 状态枚举

```python
class RegistryState(Enum):
    LOADING = "loading"      # 首次加载或 reload 中且无 effective
    READY = "ready"           # effective 可用
    RELOADING = "reloading"   # effective 可用，但有 reload 在后台进行
    ERROR = "error"           # 加载失败，无 effective
```

### 数据流（首次启动）

```
app.py 启动
  → registry.start()                       [立即返回，UI 可用]
  → 后台任务：_reload_worker()
      → backend = SherpaBackend(config)
      → await backend.initialize()         [模型加载 1-2 秒]
      → effective = backend                [原子赋值]
      → 通知 listeners(READY)              [托盘图标变绿]

用户按热键
  → registry.is_ready()
      ├─ False → 显示 toast "模型加载中..." 不录音
      └─ True  → registry.create_session() → 正常流程
```

### 数据流（用户切换引擎或模型）

```
settings_dialog 保存配置
  → registry.synchronize()
      → new_signature = compute(config)
      → 与 _target_signature 相同？是 → no-op
      → 否：
          → 取消进行中的 _reload_task（如有）
          → 启动新 _reload_task：
              → new_backend = build(config)
              → await new_backend.initialize()
              → 旧 effective.shutdown()
              → effective = new_backend     [原子切换]
              → 通知 listeners(READY)

【关键】reload 期间，用户按热键仍用旧 effective backend（不卡顿）
```

### Signature 计算

```python
def compute_signature(config: AppConfig) -> str:
    """配置指纹。只包含影响 backend 实例的字段。"""
    relevant = {
        "backend": config.get("stt", {}).get("backend"),
        "sherpa": config.get("stt", {}).get("sherpa"),
        "volcengine": config.get("stt", {}).get("volcengine"),
        "google": config.get("stt", {}).get("google"),
        "openai_whisper": config.get("stt", {}).get("openai_whisper"),
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()
```

### 关键设计点

1. **异步加载用 asyncio.Task**——不引入额外线程。app.py 已经是 asyncio 应用。
2. **原子切换**——`self._effective = new` 是原子赋值，旧 session 不会拿到一半状态。
3. **旧 backend 的 shutdown 在切换之后**——进行中的 session 持有旧 backend 引用（参考项目 `BackendKeepingSession` 的 Python 等价物），切换时不会销毁底层资源。
4. **取消进行中的 reload**——用户连续改两次配置时，取消旧任务避免浪费。

---

## §4. SherpaBackend + ModelManager + VAD

### 4.1 SherpaBackend 结构

```python
class SherpaBackend(TranscriptionBackend):
    def __init__(self, config: AppConfig) -> None:
        sherpa_cfg = config.get("stt", {}).get("sherpa", {})
        self._model_id = sherpa_cfg.get("model_id", "sherpa-onnx-paraformer-zh-2024-03-09")
        self._vad_enabled = sherpa_cfg.get("vad_enabled", True)
        self._num_threads = sherpa_cfg.get("num_threads", 2)
        self._provider = sherpa_cfg.get("provider", "cpu")
        self._recognizer = None
        self._vad = None
        self._model_info = None

    async def initialize(self) -> None:
        manager = ModelManager()
        self._model_info = await manager.ensure_model(self._model_id)

        loop = asyncio.get_running_loop()
        self._recognizer = await loop.run_in_executor(
            None, self._build_recognizer, self._model_info
        )

        if self._vad_enabled:
            self._vad = VadTrimmer()
            vad_path = await manager.ensure_vad_model()
            await loop.run_in_executor(None, self._vad.load, vad_path)

    def is_ready(self) -> bool:
        return self._recognizer is not None
```

### 4.2 SherpaSession（offline 模式）

```python
class SherpaSession(Session):
    def __init__(self, recognizer, vad, language: str) -> None:
        self._recognizer = recognizer
        self._vad = vad
        self._buffer: list[np.ndarray] = []

    def push_audio(self, pcm_int16: np.ndarray) -> None:
        self._buffer.append(pcm_int16)

    async def finish(self) -> str:
        if not self._buffer:
            return ""
        pcm = np.concatenate(self._buffer)
        audio_f32 = pcm.astype(np.float32) / 32768.0

        if self._vad and self._vad.available():
            audio_f32 = self._vad.trim(audio_f32, sample_rate=16000)
            if len(audio_f32) == 0:
                return ""

        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(None, self._recognize_sync, audio_f32)
        except Exception as e:
            raise RecognitionError(str(e), user_message="识别引擎错误，请检查模型") from e
        return text.strip()
```

### 4.3 ModelManager

```python
@dataclass
class ModelInfo:
    model_id: str
    family: str          # "paraformer" / "sense_voice" / "transducer"
    paths: dict[str, Path]  # {"model": ..., "tokens": ..., "vad": ...}
    language: str
    size_bytes: int

class ModelManager:
    BASE_DIR = xdg_cache_dir() / "sherpa-models"

    # SHA256 值在实施时从 HuggingFace 仓库实际下载后用 sha256sum 计算并填入。
    # 设计文档中先以常量名占位，确保字段结构清晰。
    REGISTRY = {
        "sherpa-onnx-paraformer-zh-2024-03-09": ModelMeta(
            family="paraformer",
            base_url="https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09/resolve/main/",
            files={"model": "model.int8.onnx", "tokens": "tokens.txt"},
            sha256={"model": PARAFORMER_MODEL_SHA256, "tokens": PARAFORMER_TOKENS_SHA256},
            language="zh-en",
            size_bytes=237_000_000,
        ),
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17": ModelMeta(
            family="sense_voice",
            base_url="https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/",
            files={"model": "model.int8.onnx", "tokens": "tokens.txt"},
            sha256={"model": SENSE_VOICE_MODEL_SHA256, "tokens": SENSE_VOICE_TOKENS_SHA256},
            language="zh-en-ja-ko-yue",
            size_bytes=234_000_000,
        ),
    }

    VAD_META = ModelMeta(
        family="vad",
        base_url="https://huggingface.co/csukuangfj/sherpa-onnx-silero-vad/resolve/main/",
        files={"model": "silero_vad.onnx"},
        sha256={"model": SILERO_VAD_SHA256},
        language="any",
        size_bytes=1_800_000,
    )

    async def ensure_model(self, model_id: str) -> ModelInfo: ...
    async def ensure_vad_model(self) -> Path: ...
    def list_installed(self) -> list[ModelSummary]: ...
    def remove(self, model_id: str) -> None: ...
```

**模型下载策略：**
- `httpx.AsyncClient` 异步下载
- 下载到临时文件 → SHA256 校验 → 原子 `rename` 到目标路径
- 进度回调（用于 UI 显示进度条）
- 重试 3 次（指数退避）

### 4.4 VadTrimmer

```python
class VadTrimmer:
    def __init__(self) -> None:
        self._vad = None  # sherpa_onnx.VoiceActivityDetector

    def load(self, model_path: Path) -> None:
        # 实际参数遵循 sherpa-onnx Python API：
        #   silero_vad.model = str(model_path)
        #   silero_vad.threshold = 0.5
        #   silero_vad.min_silence_duration = 0.25
        #   silero_vad.min_speech_duration = 0.25
        #   silero_vad.window_size = 512
        #   sample_rate = 16000
        config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(model_path),
                threshold=0.5,
                min_silence_duration=0.25,
                min_speech_duration=0.25,
                window_size=512,
            ),
            sample_rate=16000,
        )
        self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)

    def available(self) -> bool:
        return self._vad is not None

    def trim(self, samples_f32: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """喂入音频，提取所有 speech segment 拼接返回。"""
```

### 4.5 预置模型

| Model ID | 语言 | 大小 | 用途 |
|---|---|---|---|
| `sherpa-onnx-paraformer-zh-2024-03-09` | 中英混合 | ~230 MB | 默认，速度快、准确率高 |
| `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` | 中英日韩粤 | ~234 MB | 多语言场景 |

---

## §5. 场景化后处理（Scene Pipeline）

升级现有的 `llm.py`，让用户为不同场景配置不同的后处理 prompt。

### 5.1 配置格式

```toml
[postprocess]
enabled = true
active_scene = "default"

[[postprocess.scenes]]
id = "default"
name = "默认"
prompt = "修正 ASR 识别错误，保持原意，返回纯文本。"

[[postprocess.scenes]]
id = "code"
name = "代码场景"
prompt = "这是程序员说的话。修正中文同音字错误，把英文技术术语恢复（例如：'派森' → 'Python'，'瑞克特' → 'React'）。返回纯文本。"

[[postprocess.scenes]]
id = "translate-en"
name = "翻译为英文"
prompt = "把以下中文翻译为地道英文，只返回译文。"

[[postprocess.scenes]]
id = "polish"
name = "口语转书面"
prompt = "把口语化的中文改为书面表达，删除语气词和重复，保持原意。"
```

### 5.2 模块结构

```python
# postprocess/scene.py
@dataclass(frozen=True)
class Scene:
    id: str
    name: str
    prompt: str

class SceneRegistry:
    def get(self, scene_id: str) -> Scene | None: ...
    def list(self) -> list[Scene]: ...
    def active(self) -> Scene: ...  # 兜底返回 default
    def set_active(self, scene_id: str) -> None: ...

# postprocess/pipeline.py
class ScenePipeline:
    def __init__(self, scene_registry: SceneRegistry, llm_refiner: LLMRefiner) -> None:
        self._scenes = scene_registry
        self._llm = llm_refiner

    async def process(self, raw_text: str, scene_id: str | None = None) -> str:
        if not self._llm.is_configured():
            return raw_text
        scene = self._scenes.get(scene_id) if scene_id else self._scenes.active()
        try:
            return await self._llm.refine(raw_text, prompt=scene.prompt)
        except Exception:
            log.exception("scene postprocess failed, falling back to raw")
            return raw_text
```

### 5.3 与现有 `llm.py` 的关系

- `llm.py` → 移到 `postprocess/llm.py`
- `LLMRefiner.refine(text, prompt)` 接受外部传入的 prompt（不再硬编码）
- `ScenePipeline` 是 prompt 选择 + LLM 调用的组合器

### 5.4 UI 集成

托盘右键菜单新增"场景"子菜单，列出所有 scenes。切换场景立即生效（写 config + 更新 SceneRegistry 内存），无需 reload backend。

### 5.5 关键设计点

1. **降级**：LLM 未配置或调用失败 → 返回原始 ASR 文本，保证语音输入永远可用
2. **场景与 backend 解耦**：切场景不重启模型
3. **`default` 场景总是存在**：用户即使没配，系统也有兜底（代码层面 hardcode 一个 default）

---

## §6. 错误处理

### 6.1 错误分类与策略

| 错误类型 | 来源 | 处理策略 | 用户感知 |
|---|---|---|---|
| **模型未下载** | `ModelManager.ensure_model()` | 触发下载，进度条 toast | 进度条 + "下载完成"提示 |
| **模型下载失败** | 网络断、HF 503、磁盘满、SHA 校验失败 | 重试 3 次（指数退避）→ 仍失败则报错 | toast: "模型下载失败：{原因}" |
| **模型加载失败** | sherpa-onnx 抛异常 | reload worker 标记失败、保留旧 backend、记日志 | 托盘图标变红 + tooltip 提示 |
| **录音设备故障** | PipeWire/PulseAudio 错误（已存在） | 现有逻辑 | 现有 |
| **识别中异常** | sherpa 推理崩溃 | session.finish 抛 RecognitionError，UI 弹 toast | toast: "识别引擎错误，请检查模型" |
| **LLM 后处理失败** | API 超时、网络错、key 失效 | 降级返回原始 ASR 文本 + warning toast | toast: "LLM 后处理失败，已注入原始文本" |
| **VAD 失败** | VAD 加载失败 | 静默禁用 VAD，继续无 VAD 工作 | 日志 only |
| **配置不合法** | 用户填了错的 model_id | `synchronize()` 返回错误，保留旧 effective | toast: "配置无效：{原因}" |

### 6.2 RecognitionError 与上抛策略

```python
class RecognitionError(Exception):
    """识别期间的可恢复错误，上层应弹 toast 提示用户。"""
    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or "识别失败，请重试"

# Session 实现
async def finish(self) -> str:
    try:
        # ... 真正的识别
        return text
    except sherpa_onnx.SherpaError as e:
        log.exception("sherpa inference failed")
        raise RecognitionError(str(e), user_message="识别引擎错误，请检查模型")
    except Exception as e:
        log.exception("session.finish failed")
        raise RecognitionError(str(e), user_message="识别失败：" + str(e)[:80])

# app.py 录音流程
async def on_recording_finished(self, session):
    try:
        text = await session.finish()
        if not text:
            self._toast.show("未识别到内容", level="info")
            return
        # ... 后处理 + 注入
    except RecognitionError as e:
        self._toast.show(e.user_message, level="error")
        log.warning("recognition error: %s", e)
```

### 6.3 错误提示分级

| 场景 | toast 级别 | 提示文本 |
|---|---|---|
| 识别成功但文本为空 | info | "未识别到内容" |
| sherpa 推理异常 | error | "识别引擎错误，请检查模型" |
| 录音设备故障 | error | "录音失败：{原因}" |
| LLM 后处理失败 | warning | "LLM 后处理失败，已注入原始文本" |
| 模型加载中按热键 | info | "模型加载中，请稍候" |
| 模型下载中按热键 | info | "模型下载中（{进度}%），请稍候" |

### 6.4 toast 实现

复用现有的 `overlay.py` 显示在屏幕上。`overlay.py` 不可用时降级到 `notify-send`。

### 6.5 BackendRegistry 永远保留可用 effective

```python
async def _reload_worker(self, new_config):
    try:
        new_backend = build_backend(new_config)
        await new_backend.initialize()
        async with self._lock:
            old = self._effective
            self._effective = _Effective(new_backend, ...)
        if old:
            await old.backend.shutdown()
        self._notify(state=RegistryState.READY)
    except Exception as e:
        log.exception("reload failed")
        self._last_error = str(e)
        self._notify(state=RegistryState.ERROR, message=str(e))
        # 不 raise，旧 effective 继续工作
```

### 6.6 日志

- 所有错误进 `journalctl --user -u voice-input`
- DEBUG：reload signature 变化、session 时长、识别耗时
- INFO：backend 切换、模型下载完成
- ERROR：上面"红色"事件

### 6.7 不做的事

- ❌ 重试录音/识别（一次性操作，重试反而让用户困惑）
- ❌ 自动尝试备用 backend
- ❌ 错误上报到远程

---

## §7. 测试策略

### 7.1 测试分层

| 层 | 框架 | 特点 |
|---|---|---|
| 单元测试 | pytest + 现有 `tests/` | 快、纯函数、mock 重依赖 |
| 集成测试 | pytest + 真实小模型 | 慢、可选、CI 跳过 |
| 手动验证 | 系统服务实际运行 | 完成度门槛 |

### 7.2 各模块测试

**`backends/registry.py`（最重要）**

```python
class FakeBackend(TranscriptionBackend):
    def __init__(self, init_delay=0, fail_init=False, sig="x"):
        self.init_delay = init_delay
        self.fail_init = fail_init
        self.descriptor = BackendDescriptor("fake", sig, ...)
        self.shutdown_called = False
    async def initialize(self):
        await asyncio.sleep(self.init_delay)
        if self.fail_init:
            raise RuntimeError("boom")

# 必须覆盖：
- test_start_returns_immediately_while_loading
- test_is_ready_false_during_initialize
- test_is_ready_true_after_initialize
- test_synchronize_no_op_when_signature_unchanged
- test_synchronize_triggers_reload_when_signature_changes
- test_reload_keeps_old_effective_until_new_ready
- test_reload_failure_keeps_old_effective
- test_concurrent_reload_cancels_previous
- test_listeners_notified_on_state_change
- test_session_keeps_backend_alive_during_reload
```

**`asr/model_manager.py`**

```python
- test_ensure_model_skips_when_already_installed
- test_ensure_model_downloads_when_missing  # mock httpx
- test_ensure_model_validates_sha256
- test_ensure_model_atomic_rename
- test_ensure_model_retries_on_network_error
- test_remove_deletes_model_dir
- test_list_installed_returns_summaries
```

**`asr/vad.py`**

```python
- test_trim_returns_empty_for_pure_silence
- test_trim_keeps_speech_segments
- test_unavailable_when_not_loaded
```

**`backends/sherpa_backend.py`**

```python
- test_initialize_loads_model_via_manager
- test_create_session_returns_session
- test_session_finish_returns_text
- test_session_raises_recognition_error_on_failure
- test_is_ready_false_before_initialize
```

**`postprocess/scene.py` + `pipeline.py`**

```python
- test_scene_registry_loads_from_config
- test_scene_registry_active_returns_default_when_unset
- test_pipeline_uses_active_scene_prompt
- test_pipeline_returns_raw_when_llm_disabled
- test_pipeline_returns_raw_when_llm_fails
```

**`app.py` 适配（修改 `tests/test_app.py`）**

```python
- test_hotkey_with_not_ready_backend_shows_toast
- test_hotkey_with_ready_backend_starts_recording
- test_recording_failure_shows_error_toast
- test_recording_empty_shows_info_toast
```

### 7.3 删除的测试

- `tests/test_local_backend.py`
- `tests/test_app.py` 中针对 Whisper/SenseVoice 的部分

### 7.4 测试隔离原则

- **不在单测里加载真实模型**——用 `FakeBackend`、`FakeSherpaRecognizer`
- **不在单测里发网络请求**——`httpx.AsyncClient` 用 `respx` mock
- **不在单测里启 systemd**——app.py 测试用 `pytest-asyncio` event loop

### 7.5 手动验证清单（Definition of Done）

- [ ] `make run` 启动后 5 秒内托盘图标变绿
- [ ] 启动 1 秒内按热键 → 弹"模型加载中"toast，不录音不报错
- [ ] 模型 ready 后按热键 → 录音 → 识别 → 注入文本（中文 + 英文混合）
- [ ] 设置里改 model_id 保存 → 旧模型继续可用 → 新模型 ready 后无缝切换
- [ ] 故意配置错的 model_id → toast 报错 → 旧模型继续可用
- [ ] 切换场景 → 新文本走新 prompt
- [ ] LLM 配置错（key 失效）→ 注入未经 LLM 处理的 ASR 原文 + warning toast
- [ ] 删除 cache 目录 → 重启 → 模型自动重新下载

---

## 依赖变更

### 新增

```toml
[project.dependencies]
sherpa-onnx = "^1.10"
httpx = "^0.27"

[project.optional-dependencies]
dev = ["respx"]  # 用于 mock httpx
```

### 删除

```toml
# 删除以下可选依赖组：
[project.optional-dependencies]
whisper = ["faster-whisper"]
sensevoice = ["funasr", "modelscope"]
```

---

## 配置 schema 变更示例

```toml
[stt]
backend = "sherpa"  # sherpa | volcengine | google | openai-whisper
language = "zh"

[stt.sherpa]
model_id = "sherpa-onnx-paraformer-zh-2024-03-09"
vad_enabled = true
num_threads = 2
provider = "cpu"  # cpu | cuda | coreml

# stt.local 整体删除
# stt.whisper / stt.sensevoice 整体删除

[postprocess]
enabled = true
active_scene = "default"

[[postprocess.scenes]]
id = "default"
name = "默认"
prompt = "修正 ASR 识别错误，保持原意，返回纯文本。"

# 其他 scenes 同上
```

---

## 实施次序建议

1. **基础设施层**：`base.py`（接口）、`asr/session.py`、`postprocess/llm.py`（移动）
2. **模型管理层**：`asr/model_manager.py`、`asr/vad.py`
3. **Sherpa 实现**：`sherpa_backend.py`
4. **Registry 层**：`backends/registry.py` + 单元测试
5. **后处理升级**：`postprocess/scene.py`、`postprocess/pipeline.py`
6. **app.py / settings_dialog.py 适配**
7. **删除老代码** + 删除老测试
8. **手动验证清单**
