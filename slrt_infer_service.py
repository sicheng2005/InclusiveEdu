import os
import json
import sys
import traceback
import torch
import cv2
import numpy as np

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# 将 slrt_vendor 的代码路径加入到环境变量中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(PROJECT_ROOT, "slrt_vendor")
SLT_DIR = os.path.join(VENDOR_DIR, "TwoStreamNetwork")  # 默认采用 TwoStreamNetwork；yaml 里 data/、pretrained_models/ 均相对此目录
if SLT_DIR not in sys.path:
    sys.path.insert(0, SLT_DIR)


def _resolve_project_path(p: str) -> str:
    """环境变量里常写相对项目根的路径；TwoStream 官方配置则假定 cwd 在 TwoStreamNetwork。"""
    p = (p or "").strip()
    if not p:
        return p
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p))


def _patch_mbart_tokenizer_use_slow():
    """
    新版 transformers 会把 mBart 目录下的 sentencepiece.bpe.model 误走 tiktoken 解析，触发
    ValueError: Error parsing line ... in sentencepiece.bpe.model
    TwoStream 官方权重需使用慢速（SentencePiece）分词器。
    """
    from transformers import MBartTokenizer

    if getattr(MBartTokenizer, "_slrt_mbart_slow_patched", False):
        return
    _fn = MBartTokenizer.from_pretrained.__func__

    @classmethod
    def _from_pretrained(cls, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("use_fast", False)
        return _fn(cls, *args, **kwargs)

    MBartTokenizer.from_pretrained = _from_pretrained
    MBartTokenizer._slrt_mbart_slow_patched = True


app = FastAPI(title="SLRT 推理服务")


model_cache = None
cfg_cache = None

def load_slrt_model():
    """
    加载 SLRT 手语翻译模型
    """
    global model_cache, cfg_cache
    if model_cache is not None:
        return model_cache, cfg_cache
        
    try:
        from utils.misc import load_config, make_logger
        from modelling.model import build_model
    except ImportError as e:
        raise ImportError(f"无法从 slrt_vendor 导入模型模块，请检查依赖或 PYTHONPATH: {e}")
        
    # 默认寻找 TwoStreamNetwork 的配置文件；SLRT_* 环境变量可为相对项目根的路径
    config_path = os.environ.get("SLRT_CONFIG", os.path.join(SLT_DIR, "experiments", "configs", "TwoStream", "phoenix-2014t_s2t_video.yaml"))
    ckpt_path = os.environ.get("SLRT_CKPT", "")
    config_path = _resolve_project_path(config_path)
    ckpt_path = _resolve_project_path(ckpt_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到模型配置文件: {config_path}。提示：在运行真实模型前，您需要按照 slrt_vendor 中的说明准备数据集、下载预训练权重及相关配置文件。")

    # 官方代码与 yaml 中路径（如 data/csl-daily/gloss2ids.pkl）均相对于 TwoStreamNetwork 目录；从项目根启动时必须切 cwd
    os.chdir(SLT_DIR)

    cfg = load_config(config_path)
    cfg['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 推理时不需要加载中间阶段的子模块权重，因为最终的 best.ckpt 已经包含了完整权重
    if 'model' in cfg and 'TranslationNetwork' in cfg['model']:
        cfg['model']['TranslationNetwork'].pop('load_ckpt', None)

    # TwoStream 训练脚本会先 make_logger() 写入全局 logger；独立推理进程未跑训练时 get_logger() 会 NameError
    infer_log_dir = os.path.join(PROJECT_ROOT, "slrt_infer_logs")
    os.makedirs(infer_log_dir, exist_ok=True)
    make_logger(model_dir=infer_log_dir, log_file="slrt_infer.log")

    _patch_mbart_tokenizer_use_slow()
    model = build_model(cfg)
    
    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=cfg['device'])
        if 'model_state' in state_dict:
            model.load_state_dict(state_dict['model_state'], strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
            
    model.eval()
    model_cache, cfg_cache = model, cfg
    return model, cfg

def preprocess_video(video_path, cfg):
    """
    按模型要求将视频抽取帧并转成 Tensor
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)) 
        frames.append(frame)
    cap.release()
    
    if not frames:
        raise ValueError("视频为空或读取失败")
        
    # 转换为 [C, T, H, W]
    video_tensor = torch.tensor(np.array(frames), dtype=torch.float32).permute(3, 0, 1, 2)
    # 转换为 [1, C, T, H, W]
    video_tensor = video_tensor.unsqueeze(0)
    return video_tensor

def do_slrt_inference(video_path):
    """
    执行真实的推理流程
    """
    os.chdir(SLT_DIR)
    model, cfg = load_slrt_model()
    device = cfg['device']
    video_tensor = preprocess_video(video_path, cfg).to(device)
    
    # 构造 inputs（根据模型 forward 方法所需）
    # 即使是推理阶段，TwoStreamNetwork 的 RecognitionNetwork.forward 也强制要求传入 gloss_labels 和 gls_lengths
    # 此外，TwoStream 模型需要 keypoint 输入，由于我们是在线推理没有预提取的 HRNet 关键点，这里传入全 0 的 dummy tensor (形状根据 in_channel=79)
    T = video_tensor.shape[2]
    recognition_inputs = {
        'sgn_videos': video_tensor,
        'sgn_lengths': torch.tensor([T], dtype=torch.long, device=device),
        'gloss_labels': torch.tensor([[0]], dtype=torch.long, device=device),
        'gls_lengths': torch.tensor([1], dtype=torch.long, device=device),
        'sgn_keypoints': torch.zeros((1, T, 79, 3), dtype=torch.float32, device=device),
    }
    
    with torch.no_grad():
        # S2T: Sign to Text
        outputs = model(is_train=False, recognition_inputs=recognition_inputs, translation_inputs={})
        if not isinstance(outputs, dict):
            raise RuntimeError(f"模型前向输出类型异常: {type(outputs)}")
        transformer_inputs = outputs.get('transformer_inputs')
        if transformer_inputs is None:
            raise RuntimeError("模型前向未返回 transformer_inputs（可能是识别分支输入不完整）")
        generate_cfg = cfg.get('testing', {}).get('cfg', {}).get('translation', {'beam_size': 1})
        gen_outputs = model.generate_txt(transformer_inputs=transformer_inputs, generate_cfg=generate_cfg)
        text = gen_outputs['decoded_sequences'][0]
        # 在真实使用场景中，这里返回的通常是带空格的词序列（如 phoenix 数据集格式），可按需合并或翻译
        return text


class InferRequest(BaseModel):
    video_path: str = Field(..., min_length=1, description="本机可访问的视频文件路径（由主服务保存后传入）")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/infer")
def infer(req: InferRequest):
    """
    推理服务骨架（先可跑通联调）。

    约定返回：
      - success: bool
      - text: 翻译后的中文
      - error: 失败原因（success=false 时）

    你接入 SLRT 时，把本函数里的占位逻辑替换为：
      1) 读取 req.video_path（必要时用 ffmpeg 解码成帧）
      2) 走 SLRT 的视频预处理
      3) 调用模型推理，输出中文文本
    """
    try:
        video_path = (req.video_path or "").strip()
        if not os.path.exists(video_path):
            return JSONResponse({"success": False, "error": "video_path 不存在或不可访问"}, status_code=400)

        # 尝试调用基于 slrt_vendor 的真实推理流程
        try:
            # 只有当用户配置了真实环境变量，或者准备好了配置后才会运行
            if os.environ.get("SLRT_ENABLE_REAL_INFER", "0") == "1":
                inferred_text = do_slrt_inference(video_path)
                return {"success": True, "text": inferred_text, "mode": "real_inference"}
            else:
                # 联调占位：可通过环境变量调整
                mock_text = os.environ.get("SLRT_MOCK_TEXT", "（SLRT 推理服务已连接：如需启用真实开源模型推理，请配置 SLRT_ENABLE_REAL_INFER=1 以及相关权重路径）").strip()
                return {"success": True, "text": mock_text, "mode": "mock_service"}
        except FileNotFoundError as e:
            # 常见于没有下载配置或权重
            return JSONResponse({"success": False, "error": f"配置或权重文件缺失: {e}", "mode": "error"}, status_code=500)
        except Exception as e:
            # 其他模型执行错误
            trace = traceback.format_exc()
            print(trace)
            return JSONResponse(
                {"success": False, "error": f"模型推理异常: {e}", "trace": trace, "mode": "error"},
                status_code=500,
            )
    except Exception as e:
        return JSONResponse(
            {
                "success": False,
                "error": str(e),
                "trace": traceback.format_exc(),
            },
            status_code=500,
        )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("SLRT_HOST", "127.0.0.1")
    port = int(os.environ.get("SLRT_PORT", "9001"))
    uvicorn.run(app, host=host, port=port)

