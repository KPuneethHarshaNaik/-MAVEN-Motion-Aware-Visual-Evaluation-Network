"""
app.py — MAVEN Flask Backend
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
from flask_cors import CORS
from werkzeug.utils import secure_filename

from video_model   import model_factory
from video_dataset import VideoTransform, _sample_frames as _ds_sample_frames

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "video_model_best.pth")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FRAMES   = 16
IMG_SIZE   = 96

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder  =os.path.join(os.path.dirname(__file__), "static"),
)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024   # 200 MB upload limit

# ── Load model once at startup ────────────────────────────────────────────────
_model   = None
_ck_meta = {}

def get_model():
    global _model, _ck_meta, N_FRAMES, IMG_SIZE
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
    # Auto-detect backbone from checkpoint
    state_dict = ck.get("model_state") or ck.get("model_state_dict")
    proj_weight = state_dict.get("encoder.proj.0.weight")
    if proj_weight is not None and proj_weight.shape[1] == 576:
        backbone = "mobilenet"  # Legacy MobileNetV3 checkpoint
    else:
        backbone = "efficientnet"  # EfficientNetV2-S checkpoint
    _model = model_factory("option_a", backbone=backbone)
    if state_dict is None:
        raise KeyError("Checkpoint missing model_state/model_state_dict")
    _model.load_state_dict(state_dict)
    _model = _model.to(DEVICE).eval()
    print(f"[MAVEN] Model loaded — epoch={_ck_meta['epoch']}, "
          f"AUC={_ck_meta['val_auc']}, Acc={_ck_meta['val_acc']}%  [{DEVICE}]")
    return _model


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    if not cap.isOpened():
        return {"fps": 0, "frames": 0, "width": 0, "height": 0, "duration": 0, "error": "Could not open video"}
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    dur = total / fps if fps > 0 else 0
    return {"fps": round(fps, 1), "frames": total,
            "width": w, "height": h, "duration": round(dur, 2)}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/model")
def model_page():
    # Attempt to load the model on startup/first-visit if not loaded
    try:
        get_model()
    except Exception:
        pass
    return render_template("index.html")


ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    t_start = time.perf_counter()
    if "video" not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    f   = request.files["video"]
    ext = os.path.splitext(secure_filename(f.filename))[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}. Use MP4, AVI, MOV, MKV, or WebM."}), 400
    if not ext:
        ext = ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        f.save(tmp_path)
        model = get_model()

        # Stage 1 — video meta
        t1 = time.perf_counter()
        meta = _video_meta(tmp_path)
        t1_ms = round((time.perf_counter() - t1) * 1000, 1)

        # Stage 2 — frame extraction
        t2 = time.perf_counter()
        video_tensor, thumbs, _ = _sample_frames(tmp_path, N_FRAMES, IMG_SIZE)
        t2_ms = round((time.perf_counter() - t2) * 1000, 1)

        # Stage 3+4 — Full model inference (CNN + Transformer + Attention + Classifier)
        t3 = time.perf_counter()
        video_tensor = video_tensor.to(DEVICE)
        result = model.predict(video_tensor)
        t3_ms = round((time.perf_counter() - t3) * 1000, 1)
        t4_ms = 0.0  # included in t3_ms (single pass)

        total_ms = round((time.perf_counter() - t_start) * 1000, 1)

        return jsonify({
            "status"       : "ok",
            "label"        : result["label_name"],
            "asd_prob"     : round(result["prob"] * 100, 2),
            "td_prob"      : round((1 - result["prob"]) * 100, 2),
            "confidence"   : round(result["confidence"] * 100, 2),
            "top_frames"   : result["top_frames"],
            "frame_weights": [round(w * 100, 2) for w in result["frame_weights"]],
            "frame_energies": [round(e, 3) for e in result["frame_energies"]],
            "thumbs"       : thumbs,
            "video_meta"   : meta,
            "checkpoint"   : _ck_meta,
            "timing"       : {
                "video_read_ms"    : t1_ms,
                "frame_extract_ms" : t2_ms,
                "cnn_encode_ms"    : t3_ms,
                "transformer_attn_ms": t4_ms,
                "lstm_attn_ms"       : t4_ms,
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


# ── Model info endpoint ────────────────────────────────────────────────────────
@app.route("/model_info")
@app.route("/api/model_info")
def model_info():
    try:
        model = get_model()
        total = sum(p.numel() for p in model.parameters())
        return jsonify({"params": total, "n_frames": N_FRAMES, "img_size": IMG_SIZE, "device": str(DEVICE), **_ck_meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  MAVEN — ASD Screening Frontend")
    print("  Loading model ...")
    try:
        get_model()
    except Exception as e:
        print(f"  Model not loaded at startup: {e}")
    print(f"  Open: http://127.0.0.1:5000")
    print("Starting MAVEN screening server...")
    app.run(host="127.0.0.1", port=5000, debug=True)
