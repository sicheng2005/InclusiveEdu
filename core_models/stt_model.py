import os
import platform
import shutil
import tempfile
from functools import lru_cache

# 针对国内网络环境，默认使用 Hugging Face 镜像站加速模型下载
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


DEFAULT_TRANSCRIPT = "请同学们看当前课件的重点内容。"
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-small-mlx"
DEFAULT_WHISPER_LANGUAGE = "zh"

# 短别名 -> MLX Hugging Face 仓库
MODEL_ALIASES = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

# MLX 仓库 -> faster-whisper 模型名（Systran CTranslate2 权重）
MLX_REPO_TO_FASTER = {
    "mlx-community/whisper-tiny-mlx": "tiny",
    "mlx-community/whisper-small-mlx": "small",
    "mlx-community/whisper-large-v3-turbo": "large-v3-turbo",
}

FASTER_MODEL_BY_ALIAS = {
    "tiny": "tiny",
    "small": "small",
    "large-v3-turbo": "large-v3-turbo",
}


def _mock_transcript(audio_path: str | None) -> str:
    source_name = os.path.basename(audio_path) if audio_path else "unknown"
    source_name = source_name.lower()
    if "intro" in source_name:
        return "同学们好，今天我们开始上课。"
    if "question" in source_name or "ask" in source_name:
        return "请问光合作用是什么？"
    return DEFAULT_TRANSCRIPT


def _whisper_backend_mode() -> str:
    raw = os.environ.get("WHISPER_BACKEND", "auto").strip().lower().replace("-", "_")
    if raw == "mock":
        return "mock"
    if raw == "mlx":
        return "mlx"
    if raw == "faster_whisper":
        return "faster_whisper"
    return "auto"


def _mlx_available() -> bool:
    try:
        import mlx_whisper  # noqa: F401

        return True
    except ImportError:
        return False


def _faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        return False


def _auto_backend_order() -> list[str]:
    """Apple Silicon 上优先 MLX，其余平台优先 faster-whisper；仅加入当前环境可导入的后端。"""
    sys_name = platform.system()
    machine = platform.machine().lower()
    is_apple_silicon = sys_name == "Darwin" and machine in ("arm64", "aarch64")

    if is_apple_silicon and _mlx_available():
        order = ["mlx"]
        if _faster_whisper_available():
            order.append("faster_whisper")
        return order

    order = []
    if _faster_whisper_available():
        order.append("faster_whisper")
    if _mlx_available():
        order.append("mlx")
    return order


def _backends_to_try() -> list[str]:
    mode = _whisper_backend_mode()
    if mode == "mock":
        return []
    if mode == "mlx":
        return ["mlx"] + (["faster_whisper"] if _faster_whisper_available() else [])
    if mode == "faster_whisper":
        return ["faster_whisper"] + (["mlx"] if _mlx_available() else [])
    return _auto_backend_order()


def _resolve_mlx_model(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def _resolve_faster_whisper_model(model_name: str) -> str:
    mlx_id = MODEL_ALIASES.get(model_name, model_name)
    if mlx_id in MLX_REPO_TO_FASTER:
        return MLX_REPO_TO_FASTER[mlx_id]
    if model_name in FASTER_MODEL_BY_ALIAS:
        return FASTER_MODEL_BY_ALIAS[model_name]
    # 用户可传入 faster-whisper 支持的名称或 HF 模型 ID
    return model_name


@lru_cache(maxsize=1)
def _ensure_ffmpeg_on_path() -> str:
    """
    将 imageio-ffmpeg 自带的 ffmpeg 暴露到 PATH。

    Windows 上通常无法无管理员权限创建 symlink，且官方二进制名不是 ffmpeg.exe，
    会导致 Whisper 加载 webm 等格式时找不到 ffmpeg，识别失败并回退占位文本。
    """
    import imageio_ffmpeg

    ffmpeg_exe = os.path.normpath(imageio_ffmpeg.get_ffmpeg_exe())
    shim_dir = os.path.join(tempfile.gettempdir(), "inclusiveedu-bin")
    os.makedirs(shim_dir, exist_ok=True)
    shim_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    shim_path = os.path.join(shim_dir, shim_name)
    if not os.path.exists(shim_path):
        try:
            os.symlink(ffmpeg_exe, shim_path)
        except OSError:
            try:
                shutil.copy2(ffmpeg_exe, shim_path)
            except OSError as exc:
                print(f"[STT Model] 无法安装 ffmpeg 占位符（将尝试直接使用原路径）: {exc}")
                ffmpeg_dir = os.path.dirname(ffmpeg_exe)
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
                return ffmpeg_exe
    os.environ["PATH"] = shim_dir + os.pathsep + os.environ.get("PATH", "")
    return shim_path


@lru_cache(maxsize=1)
def _mlx_whisper_module():
    _ensure_ffmpeg_on_path()
    import mlx_whisper

    return mlx_whisper


def _faster_whisper_device() -> str:
    d = os.environ.get("WHISPER_DEVICE", "").strip().lower()
    if d in ("cpu", "cuda"):
        return d
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _faster_whisper_compute_type(device: str) -> str:
    env = os.environ.get("WHISPER_COMPUTE_TYPE", "").strip().lower()
    if env:
        return env
    return "float16" if device == "cuda" else "int8"


@lru_cache(maxsize=8)
def _faster_whisper_model(model_id: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_id, device=device, compute_type=compute_type)


def _transcribe_mlx(audio_path: str, mlx_model: str, language: str) -> str:
    mod = _mlx_whisper_module()
    result = mod.transcribe(
        audio_path,
        path_or_hf_repo=mlx_model,
        language=language,
        task="transcribe",
        fp16=True,
        verbose=False,
        condition_on_previous_text=False,
    )
    return (result.get("text") or "").strip()


def _faster_whisper_use_vad() -> bool:
    return os.environ.get("WHISPER_VAD_FILTER", "").strip().lower() in ("1", "true", "yes")


def _transcribe_faster_whisper(audio_path: str, fw_model: str, language: str) -> str:
    _ensure_ffmpeg_on_path()
    device = _faster_whisper_device()
    compute_type = _faster_whisper_compute_type(device)
    model = _faster_whisper_model(fw_model, device, compute_type)
    segments, _info = model.transcribe(
        audio_path,
        language=language or None,
        task="transcribe",
        vad_filter=_faster_whisper_use_vad(),
    )
    parts = [s.text for s in segments]
    return "".join(parts).strip()


def speech_to_text(audio_path: str | None):
    """
    本地 Whisper 语音转文字。

    后端由环境变量 WHISPER_BACKEND 控制：
    - auto（默认）：Apple Silicon 上优先 MLX，否则优先 faster-whisper；仅使用当前环境已安装且可导入的后端。
    - mlx / faster_whisper：强制优先使用该后端，失败时自动尝试另一可用后端。
    - mock：始终返回占位文本（便于无模型环境联调）。

    其它环境变量：
    - WHISPER_MODEL：默认与 MLX 社区仓库一致；会自动映射到 faster-whisper 对应尺寸。
    - WHISPER_LANGUAGE：如 zh。
    - WHISPER_DEVICE：cpu / cuda（仅 faster-whisper；默认自动检测 CUDA）。
    - WHISPER_COMPUTE_TYPE：覆盖 faster-whisper 的 compute_type（如 int8_float16）。
    - WHISPER_VAD_FILTER：设为 1 / true 时为 faster-whisper 开启 VAD；默认关闭，
      避免短录音、浏览器 webm 被误判为静音而得到空文本。

    模型或推理失败时回退占位文本，避免课堂链路中断。
    """
    source_name = os.path.basename(audio_path) if audio_path else "unknown"
    print(f"[STT Model] 正在识别音频文件: {source_name}")

    if not audio_path or not os.path.exists(audio_path):
        return _mock_transcript(audio_path)

    if _whisper_backend_mode() == "mock":
        print("[STT Model] WHISPER_BACKEND=mock，使用占位文本。")
        return _mock_transcript(audio_path)

    raw_model = os.environ.get("WHISPER_MODEL", DEFAULT_WHISPER_MODEL)
    mlx_model = _resolve_mlx_model(raw_model)
    fw_model = _resolve_faster_whisper_model(raw_model)
    language = os.environ.get("WHISPER_LANGUAGE", DEFAULT_WHISPER_LANGUAGE)
    backends = _backends_to_try()

    print(
        f"[STT Model] 后端顺序: {backends or '(无可用后端)'} | "
        f"MLX 模型: {mlx_model}, faster-whisper 模型: {fw_model}, language={language}"
    )

    if not backends:
        print("[STT Model] 未安装 mlx-whisper / faster-whisper，回退占位文本。")
        return _mock_transcript(audio_path)

    for name in backends:
        try:
            if name == "mlx":
                text = _transcribe_mlx(audio_path, mlx_model, language)
            else:
                text = _transcribe_faster_whisper(audio_path, fw_model, language)
            if text:
                print(f"[STT Model] Whisper 识别完成 ({name}): {text}")
                return text
            print(f"[STT Model] 后端 {name} 未返回文本，尝试下一后端。")
        except Exception as exc:
            print(f"[STT Model] 后端 {name} 失败: {exc}")

    print("[STT Model] 所有后端均失败，回退占位结果。")
    return _mock_transcript(audio_path)
