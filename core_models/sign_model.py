import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from functools import lru_cache


DEFAULT_SIGN_MODEL = "akahana/asl-vit"
DEFAULT_SIGN_CONFIDENCE = 0.45

SPECIAL_LABEL_ACTIONS = {
    "space": "space",
    "del": "delete",
    "nothing": "noop",
}
DEFAULT_SIGN_FRAME_COUNT = 8


@lru_cache(maxsize=1)
def _sign_runtime():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from PIL import Image
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    model_name = os.environ.get("SIGN_MODEL", DEFAULT_SIGN_MODEL)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    return {
        "Image": Image,
        "torch": torch,
        "processor": processor,
        "model": model,
        "model_name": model_name,
    }


@lru_cache(maxsize=1)
def _ffmpeg_executable() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _classify_image(image_path: str):
    runtime = _sign_runtime()
    image = runtime["Image"].open(image_path).convert("RGB")
    inputs = runtime["processor"](images=image, return_tensors="pt")
    with runtime["torch"].no_grad():
        logits = runtime["model"](**inputs).logits
        probs = runtime["torch"].softmax(logits, dim=-1)[0]
        idx = int(probs.argmax().item())
        confidence = float(probs[idx].item())
    label = runtime["model"].config.id2label[idx]
    return runtime, label, confidence


def _build_result(label: str, confidence: float, note: str):
    action = SPECIAL_LABEL_ACTIONS.get(label, "append")
    if action == "append":
        text = label
    elif action == "space":
        text = " "
    else:
        text = ""
    return {
        "text": text,
        "label": label,
        "confidence": confidence,
        "action": action,
        "note": note,
    }


def recognize_sign(image_path: str | None):
    """
    使用本地图像分类模型识别单帧 ASL 手语字母。
    当前模型能力边界：更适合字母级/静态手势，不适合整句中文手语。
    """
    source_name = os.path.basename(image_path) if image_path else "unknown"
    print(f"[手语模型] 正在处理图片输入: {source_name}")

    if not image_path or not os.path.exists(image_path):
        return {
            "text": "",
            "label": "nothing",
            "confidence": 0.0,
            "action": "noop",
            "note": "未检测到有效图片输入",
        }

    confidence_threshold = float(os.environ.get("SIGN_CONFIDENCE", DEFAULT_SIGN_CONFIDENCE))

    try:
        runtime, label, confidence = _classify_image(image_path)

        print(f"[手语模型] 模型={runtime['model_name']} label={label} confidence={confidence:.3f}")

        if confidence < confidence_threshold:
            return {
                "text": "",
                "label": label,
                "confidence": confidence,
                "action": "noop",
                "note": "置信度较低，请调整手势和光线后重试",
            }

        if label == "space":
            return _build_result(label, confidence, "识别到空格手势")
        if label == "del":
            return _build_result(label, confidence, "识别到删除手势")
        if label == "nothing":
            return _build_result(label, confidence, "当前帧未识别到可用手势")
        return _build_result(label, confidence, "当前为字母级手语识别，可连续识别拼出单词")
    except Exception as exc:
        print(f"[手语模型] 推理失败: {exc}")
        return {
            "text": "",
            "label": "error",
            "confidence": 0.0,
            "action": "noop",
            "note": f"手语模型暂时不可用: {exc}",
        }


def recognize_sign_video(video_path: str | None):
    """
    对短视频片段抽帧，多帧分类后做时间维度聚合。
    这比单帧更贴近课堂摄像头场景，但当前仍是字母级手势识别，不是整句 CSLR。
    """
    source_name = os.path.basename(video_path) if video_path else "unknown"
    print(f"[手语模型] 正在处理视频输入: {source_name}")

    if not video_path or not os.path.exists(video_path):
        return {
            "text": "",
            "label": "nothing",
            "confidence": 0.0,
            "action": "noop",
            "note": "未检测到有效视频输入",
            "samples": 0,
        }

    ffmpeg_exe = _ffmpeg_executable()
    frame_count = int(os.environ.get("SIGN_FRAME_COUNT", DEFAULT_SIGN_FRAME_COUNT))
    confidence_threshold = float(os.environ.get("SIGN_CONFIDENCE", DEFAULT_SIGN_CONFIDENCE))

    with tempfile.TemporaryDirectory(prefix="signframes_") as frame_dir:
        pattern = os.path.join(frame_dir, "frame_%03d.jpg")
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", video_path,
            "-vf", f"fps={frame_count}/2,scale=224:224:force_original_aspect_ratio=decrease,pad=224:224:(ow-iw)/2:(oh-ih)/2",
            "-frames:v", str(frame_count),
            pattern,
        ]
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if completed.returncode != 0:
            return {
                "text": "",
                "label": "error",
                "confidence": 0.0,
                "action": "noop",
                "note": f"视频抽帧失败: {completed.stderr.decode(errors='ignore')[:200]}",
                "samples": 0,
            }

        frame_paths = sorted(
            os.path.join(frame_dir, name)
            for name in os.listdir(frame_dir)
            if name.endswith(".jpg")
        )
        if not frame_paths:
            return {
                "text": "",
                "label": "nothing",
                "confidence": 0.0,
                "action": "noop",
                "note": "未能从视频中提取到有效帧",
                "samples": 0,
            }

        votes = Counter()
        confidence_sum = defaultdict(float)
        valid_frames = 0
        for frame_path in frame_paths:
            frame_result = recognize_sign(frame_path)
            label = frame_result.get("label", "error")
            confidence = float(frame_result.get("confidence", 0.0))
            if label == "error":
                continue
            if confidence < confidence_threshold:
                continue
            valid_frames += 1
            votes[label] += 1
            confidence_sum[label] += confidence

        if not votes:
            return {
                "text": "",
                "label": "nothing",
                "confidence": 0.0,
                "action": "noop",
                "note": "视频中未稳定识别到手势，请放慢动作并保持手在画面中央",
                "samples": valid_frames,
            }

        best_label, best_votes = votes.most_common(1)[0]
        avg_confidence = confidence_sum[best_label] / max(best_votes, 1)
        result = _build_result(best_label, avg_confidence, "基于短视频片段聚合的手语识别结果")
        result["samples"] = valid_frames
        return result
