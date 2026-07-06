"""
app.py — MAVEN Flask Backend
============================
Serves the visual pipeline frontend and handles video inference requests.

Run:
    python app.py
Then open: http://127.0.0.1:5000
"""

import os, sys, time, base64, io, json, tempfile, traceback
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
import threading
import hashlib

# Limit threads to massively reduce RAM footprint on free tier hosting
torch.set_num_threads(1)

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
    
    # Warmup pass
    with torch.no_grad():
        try:
            from torch.amp import autocast
            with autocast(device_type=str(DEVICE).split(":")[0] if DEVICE.type == "cuda" else "cpu"):
                dummy_input = torch.zeros((1, N_FRAMES, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)
                _model(dummy_input)
        except Exception as e:
            print(f"Warmup failed: {e}")

    print(f"[MAVEN] Model loaded — epoch={_ck_meta['epoch']}, "
          f"AUC={_ck_meta['val_auc']}, Acc={_ck_meta['val_acc']}%  [{DEVICE}]")
    return _model


# ── Helpers ───────────────────────────────────────────────────────────────────

def _frame_to_b64(frame_bgr: np.ndarray, thumb_size: int = 160) -> str:
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
    raw_frames = _ds_sample_frames(video_path, n_frames=n, strategy="uniform")
    thumb_list = [_frame_to_b64(f, thumb_size=160) for f in raw_frames]
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


# ── Jobs & Caching ────────────────────────────────────────────────────────────

job_store = {}
inference_cache = {}

def _compute_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def process_video_job(job_id, tmp_path):
    job_store[job_id]["status"] = "processing"
    t_start = time.perf_counter()
    try:
        # Validate duration
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = total_frames / fps if fps > 0 else 0
        if duration > 120:
            raise ValueError(f"Video is too long ({duration:.1f}s). Maximum allowed is 120s.")
        if total_frames < 5:
            raise ValueError("Video is too short or contains invalid frames.")

        file_hash = _compute_hash(tmp_path)
        if file_hash in inference_cache:
            res = inference_cache[file_hash]
            job_store[job_id]["result"] = res
            job_store[job_id]["status"] = "completed"
            os.unlink(tmp_path)
            return

        model = get_model()

        t1 = time.perf_counter()
        meta = _video_meta(tmp_path)
        t1_ms = round((time.perf_counter() - t1) * 1000, 1)

        t2 = time.perf_counter()
        video_tensor, thumbs, _ = _sample_frames(tmp_path, N_FRAMES, IMG_SIZE)
        t2_ms = round((time.perf_counter() - t2) * 1000, 1)

        t3 = time.perf_counter()
        video_tensor = video_tensor.to(DEVICE)
        result = model.predict(video_tensor)
        t3_ms = round((time.perf_counter() - t3) * 1000, 1)
        t4_ms = 0.0

        total_ms = round((time.perf_counter() - t_start) * 1000, 1)

        res = {
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
                "total_ms"         : total_ms,
            }
        }
        inference_cache[file_hash] = res
        job_store[job_id]["result"] = res
        job_store[job_id]["status"] = "completed"

    except Exception as e:
        traceback.print_exc()
        job_store[job_id]["error"] = str(e)
        job_store[job_id]["status"] = "failed"
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/model")
def model_page():
    try:
        get_model()
    except Exception:
        pass
    return render_template("index.html")

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    f = request.files["video"]
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
    except Exception as e:
        os.unlink(tmp_path)
        return jsonify({"error": str(e)}), 500
        
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "pending"}
    
    if request.form.get("async") == "true":
        threading.Thread(target=process_video_job, args=(job_id, tmp_path)).start()
        return jsonify({"job_id": job_id, "status": "pending"})
    else:
        # Synchronous mode
        process_video_job(job_id, tmp_path)
        job = job_store[job_id]
        if job["status"] == "failed":
            return jsonify({"error": job.get("error", "Unknown error")}), 500
        return jsonify(job["result"])

@app.route("/status/<job_id>", methods=["GET"])
def get_status(job_id):
    if job_id not in job_store:
        return jsonify({"error": "Job not found"}), 404
    job = job_store[job_id]
    if job["status"] == "failed":
        return jsonify({"status": "failed", "error": job.get("error", "Unknown error")})
    elif job["status"] == "completed":
        return jsonify({"status": "completed", "result": job["result"]})
    else:
        return jsonify({"status": job["status"]})

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
    print("Starting MAVEN screening server with Waitress...")
    from waitress import serve
    serve(app, host="127.0.0.1", port=5000)
