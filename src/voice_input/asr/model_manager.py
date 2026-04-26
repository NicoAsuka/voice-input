from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from voice_input.config import xdg_cache_dir

log = logging.getLogger(__name__)


# SHA256 在实施时从 HuggingFace 实际下载文件用 sha256sum 计算并填入。
# 占位常量，运行时下载完成后会校验；首次实施需替换为真实值。
PARAFORMER_MODEL_SHA256 = ""
PARAFORMER_TOKENS_SHA256 = ""
SENSE_VOICE_MODEL_SHA256 = ""
SENSE_VOICE_TOKENS_SHA256 = ""
SILERO_VAD_SHA256 = ""


@dataclass(frozen=True)
class ModelMeta:
    family: str
    base_url: str
    files: dict[str, str]
    sha256: dict[str, str]
    language: str
    size_bytes: int


@dataclass
class ModelInfo:
    model_id: str
    family: str
    paths: dict[str, Path]
    language: str
    size_bytes: int


@dataclass
class ModelSummary:
    model_id: str
    family: str
    language: str
    installed: bool
    size_bytes: int


REGISTRY: dict[str, ModelMeta] = {
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


async def _download_to_path(
    client: httpx.AsyncClient, url: str, dest: Path, expected_sha256: str
) -> None:
    """下载到临时文件 -> SHA256 校验 -> 原子 rename。"""
    log.info("Downloading %s -> %s", url, dest)
    with tempfile.NamedTemporaryFile(delete=False, dir=dest.parent, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
    try:
        async with client.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            sha = hashlib.sha256()
            with open(tmp_path, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                    f.write(chunk)
                    sha.update(chunk)

        if expected_sha256:
            actual = sha.hexdigest()
            if actual != expected_sha256:
                raise RuntimeError(
                    f"SHA256 mismatch for {url}: expected {expected_sha256}, got {actual}"
                )
        tmp_path.replace(dest)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _paths_for_meta(meta: ModelMeta, target_dir: Path) -> dict[str, Path]:
    return {role: target_dir / filename for role, filename in meta.files.items()}


def _meta_files_installed(meta: ModelMeta, target_dir: Path) -> bool:
    return all(path.is_file() for path in _paths_for_meta(meta, target_dir).values())


async def _download_meta_to_dir(meta: ModelMeta, target_dir: Path) -> dict[str, Path]:
    """下载 meta 中所有文件到 staging dir，成功后提升为 target_dir。"""
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{target_dir.name}.",
            suffix=".tmp",
            dir=target_dir.parent,
        )
    )
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            for role, filename in meta.files.items():
                url = meta.base_url + filename
                dest = staging_dir / filename
                expected = meta.sha256.get(role, "")
                await _download_to_path(client, url, dest, expected)

        if _meta_files_installed(meta, target_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
            return _paths_for_meta(meta, target_dir)
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        staging_dir.replace(target_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return _paths_for_meta(meta, target_dir)


class ModelManager:
    """Sherpa-onnx 模型下载、校验、缓存管理。"""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or (xdg_cache_dir() / "sherpa-models")
        self._locks: dict[str, asyncio.Lock] = {}

    def _model_dir(self, model_id: str) -> Path:
        return self.base_dir / model_id

    def _lock_for(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def list_installed(self) -> list[ModelSummary]:
        """列出 REGISTRY 中所有模型，标记是否已安装。"""
        results: list[ModelSummary] = []
        for model_id, meta in REGISTRY.items():
            target_dir = self._model_dir(model_id)
            results.append(
                ModelSummary(
                    model_id=model_id,
                    family=meta.family,
                    language=meta.language,
                    installed=_meta_files_installed(meta, target_dir),
                    size_bytes=meta.size_bytes,
                )
            )
        return results

    def remove(self, model_id: str) -> None:
        """删除模型目录。不存在时静默；调用方需避免与下载并发执行。"""
        if model_id not in REGISTRY:
            log.debug("remove: unknown model_id %s", model_id)
            return

        base_dir = self.base_dir.resolve()
        target_dir = self._model_dir(model_id)
        target_resolved = target_dir.resolve(strict=False)
        try:
            target_resolved.relative_to(base_dir)
        except ValueError:
            log.warning(
                "remove: refusing to remove %s outside %s",
                target_resolved,
                base_dir,
            )
            return

        if not target_resolved.exists():
            log.debug("remove: %s not found at %s", model_id, target_resolved)
            return

        log.info("Removing model %s from %s", model_id, target_dir)
        try:
            shutil.rmtree(target_resolved)
        except FileNotFoundError:
            log.debug(
                "remove: %s disappeared before removal at %s",
                model_id,
                target_resolved,
            )

    async def ensure_model(self, model_id: str) -> ModelInfo:
        """已存在 -> 直接返回；不存在 -> 下载到 cache 目录。"""
        if model_id not in REGISTRY:
            raise KeyError(f"Unknown model_id: {model_id}")
        meta = REGISTRY[model_id]
        target_dir = self._model_dir(model_id)

        async with self._lock_for(model_id):
            existing_paths = _paths_for_meta(meta, target_dir)
            if _meta_files_installed(meta, target_dir):
                log.debug("Model %s already installed", model_id)
                return ModelInfo(
                    model_id=model_id,
                    family=meta.family,
                    paths=existing_paths,
                    language=meta.language,
                    size_bytes=meta.size_bytes,
                )

            return await self._install_model(model_id, meta, target_dir)

    async def _install_model(
        self, model_id: str, meta: ModelMeta, target_dir: Path
    ) -> ModelInfo:
        log.info("Installing model %s into %s", model_id, target_dir)
        paths = await _download_meta_to_dir(meta, target_dir)
        return ModelInfo(
            model_id=model_id,
            family=meta.family,
            paths=paths,
            language=meta.language,
            size_bytes=meta.size_bytes,
        )

    async def ensure_vad_model(self) -> Path:
        """确保 silero_vad.onnx 存在，返回路径。"""
        target_dir = self.base_dir / "silero-vad"
        existing = target_dir / VAD_META.files["model"]
        async with self._lock_for("silero-vad"):
            if existing.is_file():
                return existing
            paths = await _download_meta_to_dir(VAD_META, target_dir)
            return paths["model"]
