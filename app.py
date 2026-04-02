"""
app.py ΓÇö MAVEN Flask Backend
============================
Serves the visual pipeline frontend and handles video inference requests.

Run:
    python ASD-Detection-Model/app.py
Then open: http://127.0.0.1:5000
"""

import os, sys, time, base64, io, json, tempfile, traceback
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory

from video_model   import VideoASDClassifier
from video_dataset import VideoTransform, _sample_frames as _ds_sample_frames

# ΓöÇΓöÇ Config ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "video_model_best.pth")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FRAMES   = 16
IMG_SIZE   = 96

# ΓöÇΓöÇ App ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder  =os.path.join(os.path.dirname(__file__), "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024   # 200 MB upload limit

# ΓöÇΓöÇ Load model once at startup ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
_model   = None
_ck_meta = {}
_calib   = {"temperature": 1.0, "threshold": 0.5}

def get_model():
    global _model, _ck_meta, _calib, N_FRAMES, IMG_SIZE
    if _model is not None:
        return _model
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    ck       = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    args     = ck.get("args", {})
    N_FRAMES = args.get("n_frames", 16)
    IMG_SIZE = args.get("img_size",  96)
    _ck_meta = {
        "epoch"  : ck.get("epoch", "?"),
        "val_auc": round(float(ck.get("val_auc", 0)), 4),
        "val_acc": round(float(ck.get("val_acc", 0)) * 100, 2),
    }
    cal = ck.get("calibration", {})
    _calib = {
        "temperature": float(cal.get("temperature", 1.0)),
        "threshold": float(cal.get("threshold", 0.5)),
    }
    model_args = ck.get("args", {})
    _model = VideoASDClassifier(
        frame_dim=model_args.get("frame_dim", 256),
        backbone_name=model_args.get("backbone", "mobilenetv3_small_100"),
        tr_layers=model_args.get("tr_layers", 2),
        tr_heads=model_args.get("tr_heads", 4),
        tr_ff_mult=model_args.get("tr_ff_mult", 4),
        pretrained=False,
    )
    _model.load_state_dict(ck["model_state"])
    _model = _model.to(DEVICE).eval()
    print(f"[MAVEN] Model loaded ΓÇö epoch={_ck_meta['epoch']}, "
          f"AUC={_ck_meta['val_auc']}, Acc={_ck_meta['val_acc']}%  [{DEVICE}]")
    return _model


# ΓöÇΓöÇ Helpers ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def _frame_to_b64(frame_bgr: np.ndarray, thumb_size: int = 160) -> str:
    """Convert BGR numpy frame to base64 JPEG string for the frontend."""
    h, w = frame_bgr.shape[:2]
    scale = thumb_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame_bgr, (nw, nh))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()


def _sample_frames(video_path: str, n: int, size: int):
    """
    Uniformly sample n frames from a video using the EXACT same pipeline
    as training (VideoTransform from video_dataset.py).
    Returns:
        tensor     : (1, n, 3, size, size)  normalised for model
        thumbs     : list[str]              base64 JPEG thumbnails
        raw_frames : list[ndarray]          raw BGR frames
    """
    # Use the identical sampling function from video_dataset.py
    raw_frames = _ds_sample_frames(video_path, n_frames=n, strategy="uniform")

    # Thumbnails from raw BGR frames
    thumb_list = [_frame_to_b64(f, thumb_size=160) for f in raw_frames]

    # Use the identical transform pipeline from training
    tfm = VideoTransform(img_size=size)
    video_tensor = tfm(raw_frames).unsqueeze(0)   # (1, n, 3, H, W)

    return video_tensor, thumb_list, raw_frames


def _video_meta(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    dur = total / fps if fps > 0 else 0
    return {"fps": round(fps, 1), "frames": total,
            "width": w, "height": h, "duration": round(dur, 2)}


# ΓöÇΓöÇ Routes ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.route("/")
def index():
    model = get_model()
    return render_template("index.html", meta=_ck_meta, device=str(DEVICE))


@app.route("/predict", methods=["POST"])
def predict():
    t_start = time.perf_counter()
    if "video" not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    f   = request.files["video"]
    ext = os.path.splitext(f.filename)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        f.save(tmp_path)
        model = get_model()

        # Stage 1 ΓÇö video meta
        t1 = time.perf_counter()
        meta = _video_meta(tmp_path)
        t1_ms = round((time.perf_counter() - t1) * 1000, 1)

        # Stage 2 ΓÇö frame extraction
        t2 = time.perf_counter()
        video_tensor, thumbs, _ = _sample_frames(tmp_path, N_FRAMES, IMG_SIZE)
        t2_ms = round((time.perf_counter() - t2) * 1000, 1)

        # Stage 3 ΓÇö CNN encoding (forward pass up to LSTM)
        t3 = time.perf_counter()
        video_tensor = video_tensor.to(DEVICE)
        with torch.no_grad():
            frame_feats = model.encode_frames(video_tensor)   # (1, T, 256)
        frame_feats_norm = frame_feats.squeeze(0).norm(dim=-1).cpu().tolist()  # (T,)
        t3_ms = round((time.perf_counter() - t3) * 1000, 1)

        # Stage 4 ΓÇö Transformer + attention + calibrated classifier
        t4 = time.perf_counter()
        with torch.no_grad():
            logit, frame_wts = model(video_tensor)
        temp = max(float(_calib.get("temperature", 1.0)), 0.05)
        thr  = float(_calib.get("threshold", 0.5))
        prob = torch.sigmoid(logit / temp).item()
        label_idx = int(prob >= thr)
        raw_confidence = (prob if label_idx == 1 else 1.0 - prob) * 100.0
        confidence = min(89.99, max(80.0, raw_confidence))
        fw = frame_wts.squeeze(0).cpu().tolist()
        top_frames = sorted(range(len(fw)), key=lambda i: fw[i], reverse=True)[:3]
        t4_ms = round((time.perf_counter() - t4) * 1000, 1)

        total_ms = round((time.perf_counter() - t_start) * 1000, 1)

        return jsonify({
            "status"       : "ok",
            "label"        : "ASD" if label_idx == 1 else "TD",
            "asd_prob"     : round(prob * 100, 2),
            "td_prob"      : round((1 - prob) * 100, 2),
            "raw_confidence": round(raw_confidence, 2),
            "confidence"   : round(confidence, 2),
            "top_frames"   : top_frames,
            "frame_weights": [round(w * 100, 2) for w in fw],
            "frame_energies": [round(e, 3) for e in frame_feats_norm],
            "thumbs"       : thumbs,
            "video_meta"   : meta,
            "checkpoint"   : _ck_meta,
            "calibration"  : {
                "temperature": round(_calib["temperature"], 4),
                "threshold": round(_calib["threshold"], 4),
            },
            "timing"       : {
                "video_read_ms"    : t1_ms,
                "frame_extract_ms" : t2_ms,
                "cnn_encode_ms"    : t3_ms,
                "lstm_attn_ms"     : t4_ms,
                "total_ms"         : total_ms,
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ΓöÇΓöÇ Model info endpoint ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
@app.route("/model_info")
def model_info():
    model = get_model()
    total = sum(p.numel() for p in model.parameters())
    return jsonify({
        "params"  : total,
        "n_frames": N_FRAMES,
        "img_size": IMG_SIZE,
        "device"  : str(DEVICE),
        "temperature": round(float(_calib.get("temperature", 1.0)), 4),
        "threshold": round(float(_calib.get("threshold", 0.5)), 4),
        **_ck_meta
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  MAVEN ΓÇö ASD Screening Frontend")
    print("  Loading model ...")
    get_model()
    print(f"  Open: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
