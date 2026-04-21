# STT API Backends Design

**Date:** 2026-04-21
**Status:** Approved

## 概述

为 voice-input 项目添加外部语音转文字（STT）API 后端支持。当前项目仅使用本地 faster-whisper 进行转写，本设计引入 OpenAI Whisper API、Google Cloud Speech-to-Text 和字节火山（Volcengine）语音识别 API 作为可选后端。

### 核心约束

- **互斥选择**：用户同一时间只能选择一个 STT 后端（本地或某个远程 API）
- **录音后发送**：远程 API 模式下，录音结束后一次性发送完整音频（非流式）
- **Overlay 行为**：远程模式下保留波形动画，隐藏实时转写文本，显示"识别中..."等待动画

---

## 第一部分：抽象接口架构

### TranscriptionBackend ABC

所有后端实现统一的抽象基类：

```python
# src/voice_input/backends/base.py
from abc import ABC, abstractmethod
import numpy as np

class TranscriptionBackend(ABC):
    @abstractmethod
    async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
        """接收 int16 音频数组，返回转写文本"""
        ...

    @abstractmethod
    def is_streaming(self) -> bool:
        """是否支持实时流式输出（本地 Whisper 返回 True，API 模式返回 False）"""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """加载模型/验证 API 连通性"""
        ...

    async def cleanup(self) -> None:
        """释放资源（默认空实现）"""
        pass
```

### 工厂函数

```python
# src/voice_input/backends/__init__.py
def create_backend(config: AppConfig) -> TranscriptionBackend:
    backend_name = config.get("stt", {}).get("backend", "local")
    if backend_name == "local":
        from .local_whisper import LocalWhisperBackend
        return LocalWhisperBackend(config)
    elif backend_name == "openai":
        from .openai_whisper import OpenAIWhisperBackend
        return OpenAIWhisperBackend(config)
    elif backend_name == "google":
        from .google_speech import GoogleSpeechBackend
        return GoogleSpeechBackend(config)
    elif backend_name == "volcengine":
        from .volcengine_speech import VolcengineSpeechBackend
        return VolcengineSpeechBackend(config)
    else:
        raise ValueError(f"Unknown STT backend: {backend_name}")
```

### 新增文件结构

```
src/voice_input/backends/
├── __init__.py          # create_backend 工厂函数
├── base.py              # TranscriptionBackend ABC
├── local_whisper.py     # 从 whisper_worker.py 迁移的本地转写逻辑
├── openai_whisper.py    # OpenAI Whisper API 客户端
├── google_speech.py     # Google Cloud Speech-to-Text 客户端
└── volcengine_speech.py # 字节火山语音识别客户端
```

### WhisperWorker 重构

当前 `WhisperWorker`（QThread）承担两个职责：音频缓冲和转写。重构后：

- **音频缓冲逻辑**保留在 `WhisperWorker` 中（drain queue、accumulate buffer、cap buffer）
- **转写逻辑**委托给 `TranscriptionBackend`
- 本地模式：`WhisperWorker` 持续轮询，每 500ms 调用 `backend.transcribe()`（保持实时转写）
- 远程模式：`WhisperWorker` 只做音频缓冲，录音结束后由 `AppController` 取出完整 buffer 调用 `backend.transcribe()`

### AppState 变更

新增 `TRANSCRIBING` 状态：

```python
class AppState(enum.Enum):
    IDLE = "Idle"
    RECORDING = "Recording"
    TRANSCRIBING = "Transcribing"  # 新增：远程 API 转写中
    REFINING = "Refining"

_VALID_TRANSITIONS = {
    AppState.IDLE: {AppState.RECORDING},
    AppState.RECORDING: {AppState.IDLE, AppState.TRANSCRIBING, AppState.REFINING},
    AppState.TRANSCRIBING: {AppState.IDLE, AppState.REFINING},
    AppState.REFINING: {AppState.IDLE},
}
```

---

## 第二部分：API 后端详细设计

### 1. LocalWhisperBackend

从现有 `whisper_worker.py` 的 `_load_model()` 和 `_transcribe()` 逻辑提取：

- `is_streaming() -> True`：支持实时转写
- `initialize()`：加载 faster-whisper 模型
- `transcribe()`：调用 `WhisperModel.transcribe()`
- 无额外依赖变化

### 2. OpenAIWhisperBackend

使用 OpenAI `/v1/audio/transcriptions` 端点：

- `is_streaming() -> False`
- `initialize()`：验证 API key 可用
- `transcribe()`：将 int16 音频编码为 WAV，POST multipart/form-data
- 使用 `httpx.AsyncClient`（与现有 `LLMRefiner` 保持一致）
- 支持自定义 `api_base`，兼容 OpenAI 兼容服务

```python
async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
    wav_bytes = self._encode_wav(audio_data)
    response = await self._client.post(
        "/audio/transcriptions",
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        data={"model": self.model, "language": language},
    )
    response.raise_for_status()
    return response.json()["text"]
```

### 3. GoogleSpeechBackend

使用 Google Cloud Speech-to-Text REST API（v1）：

- `is_streaming() -> False`
- `initialize()`：验证凭证路径或环境变量
- `transcribe()`：音频编码为 LINEAR16 base64，POST 到 `speech.googleapis.com/v1/speech:recognize`
- 认证方式：`GOOGLE_APPLICATION_CREDENTIALS` 环境变量或配置文件路径指定 service account JSON
- 使用 `google-auth` 库获取 access token

```python
async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
    audio_b64 = base64.b64encode(audio_data.tobytes()).decode()
    lang_code = self._map_language(language)  # "zh" -> "zh-CN"
    body = {
        "config": {
            "encoding": "LINEAR16",
            "sampleRateHertz": 16000,
            "languageCode": lang_code,
        },
        "audio": {"content": audio_b64},
    }
    response = await self._client.post(
        "https://speech.googleapis.com/v1/speech:recognize",
        json=body,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    return " ".join(r["alternatives"][0]["transcript"] for r in results)
```

### 4. VolcengineSpeechBackend

使用字节火山引擎语音识别 HTTP API：

- `is_streaming() -> False`
- `initialize()`：验证 access_key/secret_key
- `transcribe()`：音频编码后发送到火山引擎 ASR 端点
- 认证方式：HMAC-SHA256 签名（access_key + secret_key 存储在 keyring）
- 音频格式：raw PCM 或 WAV

```python
async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
    audio_bytes = audio_data.tobytes()
    headers = self._sign_request(audio_bytes)
    response = await self._client.post(
        self.api_endpoint,
        content=audio_bytes,
        headers=headers,
    )
    response.raise_for_status()
    data = response.json()
    return data["result"]["text"]
```

### 错误处理策略

所有远程后端统一错误处理：

- **网络错误**：显示通知 "STT API 连接失败"，回退到空文本
- **认证错误**（401/403）：显示通知 "API Key 无效或已过期"
- **超时**（默认 30s）：显示通知 "STT API 响应超时"
- **不重试**：单次失败直接通知用户，不自动重试

---

## 第三部分：配置与设置 UI

### config.toml 新增配置

```toml
[stt]
backend = "local"    # "local" / "openai" / "google" / "volcengine"

[stt.openai]
api_base = "https://api.openai.com/v1"
model = "whisper-1"
# api_key 存储在 keyring: service="voice-input", key="stt-openai-api-key"

[stt.google]
credentials_path = ""
# 或通过 GOOGLE_APPLICATION_CREDENTIALS 环境变量

[stt.volcengine]
app_id = ""
# access_key 存储在 keyring: service="voice-input", key="stt-volcengine-access-key"
# secret_key 存储在 keyring: service="voice-input", key="stt-volcengine-secret-key"
```

### DEFAULT_CONFIG 扩展

在 `config.py` 的 `DEFAULT_CONFIG` 中新增：

```python
"stt": {
    "backend": "local",
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "model": "whisper-1",
    },
    "google": {
        "credentials_path": "",
    },
    "volcengine": {
        "app_id": "",
    },
},
```

### 系统托盘菜单扩展

在 `TrayManager` 中新增 STT Backend 子菜单，与 Language 子菜单平级：

```
STT Backend ▸
  ● Local (faster-whisper)
    OpenAI Whisper API
    Google Speech-to-Text
    字节火山语音识别
  ─────────────
    Settings...
```

- 使用 `QActionGroup`（exclusive）实现单选
- 切换后端时更新 config 并重新初始化后端

### 设置对话框扩展

扩展现有 `SettingsDialog`，添加 STT 设置标签页或区域：

- **OpenAI 设置**：API Base URL、Model、API Key（keyring）、Test 按钮
- **Google 设置**：Credentials Path（文件选择器）、Test 按钮
- **火山设置**：App ID、Access Key（keyring）、Secret Key（keyring）、Test 按钮
- 复用现有 keyring 和 test connection 模式

---

## 第四部分：数据流与状态机

### 本地模式数据流（保持不变）

```
录音开始 → AudioRecorder → whisper_queue → WhisperWorker(轮询) →
LocalWhisperBackend.transcribe() → transcription_updated signal →
Overlay 实时显示文本 → 录音结束 → LLM refine → 注入
```

### 远程 API 模式数据流

```
录音开始 → AudioRecorder → whisper_queue → WhisperWorker(仅缓冲) →
Overlay 显示波形 + "Recording..." → 录音结束 →
AppController 取出完整 buffer → 状态切换到 TRANSCRIBING →
Overlay 显示 "识别中..." → RemoteBackend.transcribe(full_buffer) →
转写结果 → LLM refine(可选) → 注入
```

### Overlay 行为对比

| 模式 | 波形动画 | 文本区域 |
|------|----------|----------|
| 本地（streaming） | 显示 | 实时转写文本 |
| 远程（录音中） | 显示 | "Recording..." 或空 |
| 远程（转写中） | 静止/脉冲 | "识别中..." |

### 状态机完整流程

```
IDLE
  │── [按下热键] ──→ RECORDING
                        │── [本地模式: WhisperWorker 实时转写]
                        │── [远程模式: WhisperWorker 仅缓冲]
                        │
                        │── [释放热键/再按] ──→ (本地模式，有文本，LLM开启)  ──→ REFINING ──→ IDLE
                        │                   ──→ (本地模式，有文本，LLM关闭)  ──→ IDLE (注入)
                        │                   ──→ (本地模式，无文本)            ──→ IDLE
                        │                   ──→ (远程模式)                    ──→ TRANSCRIBING
                        │
TRANSCRIBING
  │── [API 返回文本，LLM开启]  ──→ REFINING ──→ IDLE
  │── [API 返回文本，LLM关闭]  ──→ IDLE (注入)
  │── [API 失败/无文本]        ──→ IDLE
```

### 依赖管理

新增可选依赖（不影响本地模式用户）：

```toml
# pyproject.toml
[project.optional-dependencies]
openai = ["httpx"]           # httpx 已是现有依赖
google = ["google-auth"]
volcengine = []              # 仅使用 httpx + hmac（标准库）
all-stt = ["google-auth"]
```

### 测试策略

- 每个后端独立单元测试，mock HTTP 响应
- `TranscriptionBackend` 接口合规测试（确保所有后端实现一致行为）
- 集成测试：录音 → 转写 → 注入的端到端流程（本地后端）
- 远程后端的 `test_connection()` 方法用于设置对话框的连通性验证
