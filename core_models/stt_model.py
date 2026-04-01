import os
import tempfile
from functools import lru_cache


DEFAULT_TRANSCRIPT = "请同学们看当前课件的重点内容。"
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-small-mlx"
DEFAULT_WHISPER_LANGUAGE = "zh"

MODEL_ALIASES = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def _mock_transcript(audio_path: str | None) -> str:
    source_name = os.path.basename(audio_path) if audio_path else "unknown"
    source_name = source_name.lower()
    if "intro" in source_name:
        return "同学们好，今天我们开始上课。"
    if "question" in source_name or "ask" in source_name:
        return "请问光合作用是什么？"
    return DEFAULT_TRANSCRIPT


@lru_cache(maxsize=1)
def _ensure_ffmpeg_on_path() -> str:
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    shim_dir = os.path.join(tempfile.gettempdir(), "inclusiveedu-bin")
    os.makedirs(shim_dir, exist_ok=True)
    shim_path = os.path.join(shim_dir, "ffmpeg")
    if not os.path.exists(shim_path):
        os.symlink(ffmpeg_exe, shim_path)
    os.environ["PATH"] = shim_dir + os.pathsep + os.environ.get("PATH", "")
    return shim_path


@lru_cache(maxsize=1)
def _whisper_runtime():
    _ensure_ffmpeg_on_path()
    import mlx_whisper

    return mlx_whisper


def speech_to_text(audio_path: str | None):
    """
    使用本地 Whisper（MLX）进行语音转文字。
    若模型加载或推理失败，则回退到占位文本，避免课堂链路中断。
    """
    source_name = os.path.basename(audio_path) if audio_path else "unknown"
    print(f"[STT Model] 正在识别音频文件: {source_name}")

    if not audio_path or not os.path.exists(audio_path):
        return _mock_transcript(audio_path)

    model_name = os.environ.get("WHISPER_MODEL", DEFAULT_WHISPER_MODEL)
    model_name = MODEL_ALIASES.get(model_name, model_name)
    language = os.environ.get("WHISPER_LANGUAGE", DEFAULT_WHISPER_LANGUAGE)
    print(f"[STT Model] 当前 Whisper 模型: {model_name}, language={language}")

    try:
        mlx_whisper = _whisper_runtime()
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_name,
            language=language,
            task="transcribe",
            fp16=True,
            verbose=False,
            condition_on_previous_text=False,
        )
        text = (result.get("text") or "").strip()
        if text:
            print(f"[STT Model] Whisper 识别完成: {text}")
            return text
        print("[STT Model] Whisper 未返回文本，回退到占位结果。")
    except Exception as exc:
        print(f"[STT Model] Whisper 推理失败，已回退: {exc}")

    return _mock_transcript(audio_path)
