# SenseVoice Small 本地引擎设计

**Date:** 2026-04-22
**Status:** Draft

## 概述

为 voice-input 项目的本地 STT 后端添加阿里 SenseVoice Small 模型支持。用户可在 `local` 后端内部切换使用 faster-whisper 或 SenseVoice Small 引擎。SenseVoice Small 为非流式模型，录音结束后一次性转写。

### 核心约束

- **本地推理**：使用 `funasr` 库加载 `iic/SenseVoiceSmall` 模型，不依赖远程 API
- **引擎级切换**：`local` 后端内部通过 `engine` 字段选择 whisper 或 sensevoice
- **非流式转写**：SenseVoice 引擎录音期间只缓冲音频，录音结束后一次性转写
- **可选依赖**：whisper 和 sensevoice 的依赖分别作为可选安装组

---

## 第一部分：引擎抽象与文件结构

### LocalEngine Protocol

新增引擎协议，定义本地引擎的统一接口：

```python
# backends/local/engine.py
from pathlib import Path
from typing import Protocol

import numpy as np


class LocalEngine(Protocol):
    def load_model(self, model_name: str, device: str, cache_dir: Path) -> None:
        """加载模型到内存"""
        ...

    def transcribe(self, audio_f32: np.ndarray, language: str) -> str:
        """转写 float32 音频数组，返回文本"""
        ...

    def is_streaming(self) -> bool:
        """是否支持流式实时转写"""
        ...
```

### 文件结构变更

```
backends/
├── __init__.py              # create_backend 工厂函数
├── base.py                  # TranscriptionBackend ABC（不变）
├── local/
│   ├── __init__.py          # LocalBackend（实现 TranscriptionBackend，持有引擎）
│   ├── engine.py            # LocalEngine Protocol
│   ├── whisper_engine.py    # WhisperEngine（从现有 local_whisper.py 迁移）
│   └── sensevoice_engine.py # SenseVoiceEngine（新增）
├── local_whisper.py         # 删除（迁移到 local/whisper_engine.py）
```

### LocalBackend

`LocalBackend` 实现 `TranscriptionBackend`，内部持有一个 `LocalEngine` 实例：

- `initialize()`：根据 `engine` 配置实例化对应引擎，调用 `engine.load_model()`
- `transcribe()`：委托给 `engine.transcribe()`
- `is_streaming()`：委托给 `engine.is_streaming()`
- `cleanup()`：释放引擎持有的模型资源

---

## 第二部分：配置变更

### DEFAULT_CONFIG

原 `whisper` 配置段迁移到 `stt.local` 下，新增 `engine` 字段：

```python
"stt": {
    "backend": "local",
    "local": {
        "engine": "whisper",       # "whisper" / "sensevoice"
        "model": "medium",         # whisper: "tiny"/"base"/"small"/"medium"/"large-v3"
                                   # sensevoice: "iic/SenseVoiceSmall"
        "language": "zh",
        "device": "auto",
    },
    # openai / google / volcengine 配置保留原 spec 设计
},
```

### 变更影响

- 原 `config["whisper"]["model"]` 等引用全部更新为 `config["stt"]["local"]["model"]`
- `model` 字段含义随 engine 变化：
  - whisper 下：faster-whisper 模型名（tiny/base/small/medium/large-v3）
  - sensevoice 下：funasr 模型标识（默认 `iic/SenseVoiceSmall`）
- 当 engine 切换为 sensevoice 且用户未指定 model 时，引擎内部使用默认模型名

---

## 第三部分：SenseVoiceEngine 实现

```python
# backends/local/sensevoice_engine.py
class SenseVoiceEngine:
    DEFAULT_MODEL = "iic/SenseVoiceSmall"

    def is_streaming(self) -> bool:
        return False

    def load_model(self, model_name: str, device: str, cache_dir: Path) -> None:
        from funasr import AutoModel

        actual_device = self._resolve_device(device)  # "cuda:0" or "cpu"
        self._model = AutoModel(
            model=model_name or self.DEFAULT_MODEL,
            trust_remote_code=True,
            device=actual_device,
            cache_dir=str(cache_dir),
        )

    def transcribe(self, audio_f32: np.ndarray, language: str) -> str:
        result = self._model.generate(
            input=audio_f32,
            language=language,       # "zh"/"en"/"ja"/"ko"/"yue"
            use_itn=True,            # 逆文本正则化
        )
        if result and len(result) > 0:
            return result[0].get("text", "").strip()
        return ""
```

### 关键行为

- **非流式**：`is_streaming()` 返回 False
- **模型下载**：首次使用时从 ModelScope 下载到 `cache_dir`，后续加载本地缓存
- **语言支持**：zh/en/ja/ko/yue，language 参数直接透传给 `generate()`
- **ITN**：启用逆文本正则化，自动处理数字和标点

---

## 第四部分：WhisperWorker 与 AppController 适配

### WhisperWorker 变更

`WhisperWorker` 接收 `LocalBackend` 实例，感知引擎是否流式：

- **流式（whisper）**：行为不变，每 500ms 调用 `backend.transcribe()` 实时更新
- **非流式（sensevoice）**：录音期间只做 drain queue + accumulate，不调用转写
- 新增 `get_buffer()` 方法：供 `AppController` 在录音结束后提取完整音频 buffer

```python
class WhisperWorker(QThread):
    def __init__(self, ..., backend: LocalBackend):
        self._backend = backend
        self._streaming = backend.is_streaming()

    def run(self):
        asyncio.run(self._backend.initialize())
        self.model_ready.emit()
        self._running = True
        while self._running:
            self.msleep(self.POLL_INTERVAL_MS)
            if not self._active:
                continue
            self._drain_and_accumulate()
            if self._streaming:
                text = asyncio.run(self._backend.transcribe(
                    self.audio_buffer, self.language))
                if text:
                    self.transcription_updated.emit(text)

    def get_buffer(self) -> np.ndarray:
        return self.audio_buffer.copy()
```

### AppController 变更

`_on_stop_recording()` 根据流式/非流式分流处理：

- **流式**：使用已有的实时转写结果（现有逻辑不变）
- **非流式**：取完整 buffer，进入 `TRANSCRIBING` 状态，overlay 显示"识别中..."，异步调用 `backend.transcribe()`

```python
def _on_stop_recording(self):
    self._audio.stop()
    self._viz_timer.stop()
    self._whisper.set_active(False)

    if self._backend.is_streaming():
        text = self._last_transcription
        self._finish_transcription(text)
    else:
        buffer = self._whisper.get_buffer()
        if len(buffer) < 1600:
            self._overlay.animate_exit(on_finished=self._overlay.hide)
            self._set_state(AppState.IDLE)
            return
        self._set_state(AppState.TRANSCRIBING)
        self._overlay.update_text("识别中...")
        asyncio.ensure_future(self._transcribe_and_finish(buffer))

async def _transcribe_and_finish(self, buffer):
    text = await self._backend.transcribe(buffer, self._language)
    self._finish_transcription(text)
```

### AppState 变更

新增 `TRANSCRIBING` 状态（与原 STT API backends spec 一致）：

```python
class AppState(enum.Enum):
    IDLE = "Idle"
    RECORDING = "Recording"
    TRANSCRIBING = "Transcribing"
    REFINING = "Refining"

_VALID_TRANSITIONS = {
    AppState.IDLE: {AppState.RECORDING},
    AppState.RECORDING: {AppState.IDLE, AppState.TRANSCRIBING, AppState.REFINING},
    AppState.TRANSCRIBING: {AppState.IDLE, AppState.REFINING},
    AppState.REFINING: {AppState.IDLE},
}
```

---

## 第五部分：UI 与依赖管理

### 系统托盘菜单

Local 选项展开为子菜单显示引擎选择：

```
STT Backend >
  Local >
    * faster-whisper
      SenseVoice Small
  ---
    OpenAI Whisper API
    Google Speech-to-Text
    字节火山语音识别
```

- 使用 `QActionGroup`（exclusive）实现单选
- 切换引擎时：更新 `stt.local.engine` 配置，重新初始化 `LocalBackend`

### 依赖管理

`pyproject.toml` 中两个本地引擎作为可选依赖：

```toml
[project.optional-dependencies]
whisper = ["faster-whisper"]
sensevoice = ["funasr", "modelscope"]
local-all = ["faster-whisper", "funasr", "modelscope"]
```

运行时检测：`LocalBackend.initialize()` 时如果对应库未安装，抛出明确错误信息，例如：
- `RuntimeError("SenseVoice engine requires 'funasr'. Install with: pip install voice-input[sensevoice]")`

### 设置对话框

在 SettingsDialog 的 Local 引擎区域增加引擎下拉框（whisper / sensevoice），切换后 model 字段候选值联动更新。

---

## 测试策略

- **WhisperEngine 单元测试**：从现有 `LocalWhisperBackend` 测试迁移，mock `faster_whisper.WhisperModel`
- **SenseVoiceEngine 单元测试**：mock `funasr.AutoModel`，验证 `generate()` 调用参数和返回值解析
- **LocalBackend 单元测试**：验证引擎选择逻辑、is_streaming 委托、未安装依赖时的错误提示
- **AppController 集成测试**：验证非流式引擎下 RECORDING -> TRANSCRIBING -> IDLE/REFINING 状态流转
- **配置迁移测试**：验证旧 `whisper.*` 配置格式的兼容处理（如有需要）
