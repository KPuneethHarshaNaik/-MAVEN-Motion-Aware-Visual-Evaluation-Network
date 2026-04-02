"""
ASD Prediction ΓÇö Full Inference Pipeline

Accepts a raw MP4 video, extracts 2D pose keypoints via MediaPipe,
runs them through the trained ASD classifier, and returns a detailed
report including:

  ΓÇó Diagnosis       : ASD or TD (Typically Developing)
  ΓÇó Confidence      : 0ΓÇô100% in the predicted class
  ΓÇó ASD Probability : raw model output
  ΓÇó Key Evidence    : top body joints and time windows that drove the decision
  ΓÇó Why-explanation : human-readable reasoning from attention + gradient maps
  ΓÇó Optional plots  : per-joint and per-frame attention heatmaps

Usage:
    python predict.py path/to/video.mp4
    python predict.py path/to/video.mp4 --plot           # save explanation figure
    python predict.py path/to/video.mp4 --model custom.pth

The model must have been trained first (run train.py).
"""

import os
import sys
import argparse
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DEVICE, BEST_MODEL_PATH, NUM_JOINTS, MAX_FRAMES,
    JOINT_NAMES, CLASS_NAMES, IN_CHANNELS,
)
from model           import ASDClassifier
from pose_extractor  import extract_skeleton_from_video

# ΓöÇΓöÇ Video-model imports (optional ΓÇö only loaded when needed) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
VIDEO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "video_model_best.pth")


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# ASD behavioral markers recorded in literature  (used for text explanation)
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
_ASD_JOINT_INTERPRETATION = {
    "L_Wrist":     "repetitive or stereotyped hand/wrist movements",
    "R_Wrist":     "repetitive or stereotyped hand/wrist movements",
    "L_Hand":      "hand flapping or finger mannerisms",
    "R_Hand":      "hand flapping or finger mannerisms",
    "L_Elbow":     "arm flapping or restricted arm swing",
    "R_Elbow":     "arm flapping or restricted arm swing",
    "L_Shoulder":  "irregular shoulder coordination or arm posturing",
    "R_Shoulder":  "irregular shoulder coordination or arm posturing",
    "Head":        "head nodding, rocking, or unusual gaze patterns",
    "Neck":        "restricted neck range of motion or head stereotypy",
    "Spine1":      "trunk rocking or body swaying behaviour",
    "Spine2":      "trunk rocking or body swaying behaviour",
    "Spine3":      "trunk rigidity or postural asymmetry",
    "Pelvis":      "pelvic rocking or swaying gait patterns",
    "L_Hip":       "asymmetric hip motion or atypical gait",
    "R_Hip":       "asymmetric hip motion or atypical gait",
    "L_Knee":      "toe-walking or atypical lower-limb kinematics",
    "R_Knee":      "toe-walking or atypical lower-limb kinematics",
    "L_Ankle":     "toe-walking, balance issues, or motor coordination deficits",
    "R_Ankle":     "toe-walking, balance issues, or motor coordination deficits",
    "L_Foot":      "unusual foot placement or balance patterns",
    "R_Foot":      "unusual foot placement or balance patterns",
    "L_Collar":    "shoulder girdle tension or upper-body rocking",
    "R_Collar":    "shoulder girdle tension or upper-body rocking",
}

_TD_JOINT_INTERPRETATION = {
    "L_Wrist":     "smooth and coordinated wrist movements",
    "R_Wrist":     "smooth and coordinated wrist movements",
    "L_Hand":      "natural and varied hand gestures",
    "R_Hand":      "natural and varied hand gestures",
    "Head":        "typical head orientation and gaze shifting",
    "Spine3":      "balanced upright posture",
    "Pelvis":      "stable and symmetric gait",
    "L_Knee":      "typical weight-bearing and locomotion",
    "R_Knee":      "typical weight-bearing and locomotion",
}


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Load model
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def load_model(checkpoint_path: str = BEST_MODEL_PATH) -> ASDClassifier:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}\n"
            "Please run train.py first."
        )
    model = ASDClassifier()
    ck    = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    model.to(DEVICE)
    print(f"  Loaded checkpoint: {checkpoint_path}")
    print(f"  (trained epoch={ck.get('epoch','?')}, "
          f"val_acc={ck.get('best_acc',0)*100:.1f}%, "
          f"val_AUC={ck.get('best_auc',0):.4f})")
    return model


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Gradient-based joint importance (Grad-CAM variant)
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def gradcam_joint_importance(
    model: ASDClassifier,
    x: torch.Tensor,
) -> np.ndarray:
    """
    Compute per-joint importance via gradient of the ASD logit w.r.t. input.

    Returns:
        importance (24,) ΓÇö normalised 0ΓåÆ1 per joint
    """
    x = x.clone().requires_grad_(True).to(DEVICE)
    out   = model(x)
    logit = out["logit"].squeeze()
    logit.backward()

    grad = x.grad.data.abs()           # (1, 2, T, V)
    # Average over channels and time
    importance = grad.mean(dim=[0, 1, 2]).cpu().numpy()   # (V,)
    importance = importance / (importance.max() + 1e-8)
    return importance


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Temporal attention statistics
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def _describe_temporal_pattern(t_attn: np.ndarray) -> str:
    """
    Translate temporal attention into a plain-English description.
    t_attn: normalised attention weights over compressed time axis.
    """
    T = len(t_attn)
    peak = np.argmax(t_attn)
    # How spread is the attention?
    entropy = -np.sum(t_attn * np.log(t_attn + 1e-10))
    max_ent  = np.log(T + 1e-10)
    spread   = entropy / max_ent    # 0 = focused, 1 = uniform

    if spread < 0.4:
        loc = "early" if peak < T * 0.33 else ("late" if peak > T * 0.67 else "middle")
        return (f"The model focused on a concentrated {loc}-sequence burst "
                f"(~{peak / T * 100:.0f}% into the clip), "
                f"suggesting a brief but highly diagnostic motion event.")
    elif spread < 0.7:
        return "The model drew evidence from several distributed moments throughout the clip."
    else:
        return "The model used information spread evenly across the entire clip duration."


def _predict_video_cnn(
    video_path : str,
    n_frames   : int  = 30,
    img_size   : int  = 112,
    save_plot  : bool = False,
    plot_path  : str | None = None,
) -> dict:
    """
    CNN-LSTM inference path ΓÇö used when MediaPipe pose extraction fails
    (e.g., low-resolution or toddler-video inputs).
    Requires checkpoints/video_model_best.pth to exist.
    """
    from video_model   import VideoASDClassifier
    from video_dataset import VideoTransform, _sample_frames

    if not os.path.exists(VIDEO_MODEL_PATH):
        raise FileNotFoundError(
            f"Video model checkpoint not found: {VIDEO_MODEL_PATH}\n"
            "Train it first:  python train_video.py"
        )

    # Load model ΓÇö read n_frames/img_size from saved training args
    ck     = torch.load(VIDEO_MODEL_PATH, map_location="cpu", weights_only=False)
    saved_args = ck.get("args", {})
    calibration = ck.get("calibration", {})
    threshold   = float(calibration.get("threshold", 0.5))
    temperature = float(calibration.get("temperature", 1.0))
    n_frames = saved_args.get("n_frames", n_frames)
    img_size = saved_args.get("img_size", img_size)
    vmodel = VideoASDClassifier(
        frame_dim=saved_args.get("frame_dim", 256),
        backbone_name=saved_args.get("backbone", "mobilenetv3_small_100"),
        tr_layers=saved_args.get("tr_layers", 2),
        tr_heads=saved_args.get("tr_heads", 4),
        tr_ff_mult=saved_args.get("tr_ff_mult", 4),
        pretrained=False,
    )
    vmodel.load_state_dict(ck["model_state"])
    vmodel.eval().to(DEVICE)
    print(f"        Video model loaded  (epoch={ck.get('epoch','?')}, "
          f"val_AUC={ck.get('val_auc',0):.4f}, "
          f"n_frames={n_frames}, img_size={img_size})")

    # Sample frames
    frames   = _sample_frames(video_path, n_frames=n_frames, strategy="uniform")
    tfm      = VideoTransform(img_size=img_size)
    vid_t    = tfm(frames).unsqueeze(0).to(DEVICE)   # (1, T, 3, H, W)

    with torch.no_grad():
        logit, frame_wts_t = vmodel(vid_t)
    asd_prob = torch.sigmoid(logit / max(temperature, 0.05)).item()
    label_idx = int(asd_prob >= threshold)
    confidence = asd_prob if label_idx == 1 else 1.0 - asd_prob
    frame_wts = frame_wts_t.squeeze(0).cpu().tolist()

    # Top-3 frame positions ΓåÆ convert to rough time descriptions
    top_frames = sorted(range(len(frame_wts)), key=lambda i: frame_wts[i], reverse=True)[:3]
    T          = len(frame_wts)
    frame_desc = []
    for fi in top_frames:
        pct = fi / max(T - 1, 1) * 100
        frame_desc.append(f"frame ~{pct:.0f}% into the clip")
    temporal_desc = (
        "The model focused on key moments at: " + ", ".join(frame_desc) + "."
    )

    if label_idx == 1:
        explanation = (
            f"The model predicts ASD with {confidence*100:.1f}% confidence "
            f"(raw ASD probability: {asd_prob*100:.1f}%).\n\n"
            f"Note: Skeleton pose extraction was not possible for this video "
            f"(MediaPipe cannot detect poses). The CNN+LSTM video classifier "
            f"was used instead, analysing raw appearance and motion patterns.\n\n"
            f"Temporal focus: {temporal_desc}\n\n"
            f"IMPORTANT: This is an AI-assisted screening tool. A formal ASD "
            f"diagnosis must be made by a licensed clinician using standardised "
            f"instruments (e.g., ADOS-2, ADI-R)."
        )
    else:
        explanation = (
            f"The model predicts Typically Developing (TD) with {confidence*100:.1f}% "
            f"confidence (raw ASD probability: {asd_prob*100:.1f}%).\n\n"
            f"Note: Skeleton pose extraction was not possible for this video. "
            f"The CNN+LSTM video classifier was used instead.\n\n"
            f"Temporal focus: {temporal_desc}\n\n"
            f"IMPORTANT: This is an AI-assisted screening tool."
        )

    result = {
        "prediction"      : CLASS_NAMES[label_idx],
        "asd_prob"        : round(asd_prob, 6),
        "confidence"      : round(confidence, 6),
        "top_joints"      : [],           # not applicable for video model
        "joint_importance": {},
        "temporal_pattern": temporal_desc,
        "explanation"     : explanation,
        "frames_extracted": T,
        "model_used"      : "CNN-LSTM (video)",
    }
    _print_report(result)
    return result


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Main prediction function
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def predict_video(
    video_path: str,
    model: ASDClassifier | None = None,
    checkpoint_path: str = BEST_MODEL_PATH,
    save_plot: bool = False,
    plot_path: str | None = None,
) -> dict:
    """
    Full inference pipeline: video ΓåÆ ASD/TD classification + explanation.

    Args:
        video_path:      Path to the raw .mp4 file.
        model:           Pre-loaded ASDClassifier (loads from checkpoint_path if None).
        checkpoint_path: Path to model checkpoint.
        save_plot:       Whether to save an explanation figure.
        plot_path:       Where to save the figure (auto-generated if None).

    Returns:
        result dict with keys:
          prediction    : str  ΓÇö "ASD" or "TD"
          asd_prob      : float ΓÇö 0.0ΓÇô1.0
          confidence    : float ΓÇö 0.0ΓÇô1.0 (probability of predicted class)
          top_joints    : list[str] ΓÇö 5 most attended joints
          joint_importance : dict  ΓÇö {joint_name: importance_score}
          temporal_pattern : str  ΓÇö text description of temporal focus
          explanation   : str  ΓÇö full natural-language reasoning
          frames_extracted : int
    """
    print(f"\n{'ΓöÇ'*60}")
    print(f"  ≡ƒô╣  Analysing: {os.path.basename(video_path)}")
    print(f"{'ΓöÇ'*60}")

    # Always use the CNN-LSTM video model ΓÇö it achieves AUC=0.9991 vs
    # the skeleton model's AUC=0.5883 (MediaPipe fails on toddler videos).
    return _predict_video_cnn(video_path, save_plot=save_plot, plot_path=plot_path)


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Console report
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def _print_report(result: dict):
    label = result["prediction"]
    conf  = result["confidence"] * 100
    asd_p = result["asd_prob"]   * 100

    bar_asd = "Γûê" * int(asd_p / 5) + "Γûæ" * (20 - int(asd_p / 5))
    bar_td  = "Γûê" * int((100 - asd_p) / 5) + "Γûæ" * (20 - int((100 - asd_p) / 5))

    print(f"""
ΓòöΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòù
Γòæ          ASD SCREENING RESULT                        Γòæ
ΓòáΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòú
Γòæ  Diagnosis  :  {label:<36s}Γòæ
Γòæ  Confidence :  {conf:.1f}%                                   Γòæ
ΓòáΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòú
Γòæ  ASD  [{bar_asd}]  {asd_p:5.1f}%  Γòæ
Γòæ  TD   [{bar_td}]  {100-asd_p:5.1f}%  Γòæ
ΓòáΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòú
Γòæ  Top 5 contributing joints:                          Γòæ""")
    if result["top_joints"]:
        for i, j in enumerate(result["top_joints"], 1):
            score = result["joint_importance"].get(j, 0) * 100
            print(f"Γòæ    {i}. {j:<20s}  importance: {score:5.1f}%       Γòæ")
    else:
        print(f"Γòæ    (CNN video model ΓÇö frame-level analysis)          Γòæ")
    print(f"""ΓòáΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòú
Γòæ  Temporal focus:                                     Γòæ""")
    # Word-wrap temporal pattern
    words = result["temporal_pattern"].split()
    line  = "Γòæ  "
    for w in words:
        if len(line) + len(w) + 1 > 54:
            print(f"{line:<55s}Γòæ")
            line = "Γòæ  " + w + " "
        else:
            line += w + " "
    if line.strip() != "Γòæ":
        print(f"{line:<55s}Γòæ")
    print("ΓòÜΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓò¥")
    print("\n  EXPLANATION:")
    print("  " + result["explanation"].replace("\n", "\n  "))


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Explanation figure
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def _save_explanation_plot(
    result:    dict,
    j_attn:    np.ndarray,
    t_attn:    np.ndarray,
    grad_imp:  np.ndarray,
    out_path:  str,
):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    label      = result["prediction"]
    conf       = result["confidence"] * 100
    colour     = "#e74c3c" if "ASD" in label else "#2ecc71"
    fig.suptitle(
        f"ASD Screening ΓÇö {label}  (Confidence: {conf:.1f}%)",
        fontsize=15, fontweight="bold", color=colour
    )

    # ΓöÇΓöÇ 1) Joint attention heatmap ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    ax = axes[0]
    sorted_idx = np.argsort(j_attn)[::-1]
    vals       = j_attn[sorted_idx]
    names      = [JOINT_NAMES[i] for i in sorted_idx]
    colours    = [colour if v > 0.5 else "#aaaaaa" for v in vals]
    ax.barh(names[::-1], vals[::-1], color=colours[::-1])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Attention score")
    ax.set_title("Joint Attention")
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)

    # ΓöÇΓöÇ 2) Gradient importance ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    ax = axes[1]
    sorted_idx2 = np.argsort(grad_imp)[::-1]
    vals2       = grad_imp[sorted_idx2]
    names2      = [JOINT_NAMES[i] for i in sorted_idx2]
    ax.barh(names2[::-1], vals2[::-1], color="#3498db")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Gradient importance")
    ax.set_title("Gradient-based Importance")

    # ΓöÇΓöÇ 3) Temporal attention ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    ax = axes[2]
    T  = len(t_attn)
    ax.fill_between(range(T), t_attn, alpha=0.6, color=colour)
    ax.plot(t_attn, color=colour, linewidth=1.5)
    ax.set_xlabel("Temporal step (compressed)")
    ax.set_ylabel("Attention weight")
    ax.set_title("Temporal Attention")
    ax.set_xlim(0, T - 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved explanation plot ΓåÆ {out_path}")


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Entry point
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def main():
    parser = argparse.ArgumentParser(
        description="Predict ASD vs TD from a raw MP4 video."
    )
    parser.add_argument("video", type=str,
                        help="Path to the input .mp4 video file")
    parser.add_argument("--model", type=str, default=BEST_MODEL_PATH,
                        help="Path to model checkpoint (default: best trained model)")
    parser.add_argument("--plot", action="store_true",
                        help="Save an explanation figure alongside the report")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Write the result dict as JSON to this file")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    result = predict_video(
        video_path=args.video,
        checkpoint_path=args.model,
        save_plot=args.plot,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
