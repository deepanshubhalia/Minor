import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, url_for
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms
import timm
from werkzeug.utils import secure_filename

# Ensure we can import training modules
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from train_ae import AutoEncoder  # noqa: E402

# --------------------------------------------------------------------------- #
# Flask + Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "mov", "avi", "mkv"}

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB uploads


class NoFaceFoundError(Exception):
    """Raised when no faces are detected in an asset."""


# --------------------------------------------------------------------------- #
# Model Loading
# --------------------------------------------------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
cnn.load_state_dict(
    torch.load(ROOT_DIR / "models" / "cnn_efficientnet_b0.pt", map_location=device)
)
cnn = cnn.to(device).eval()

ae = AutoEncoder().to(device)
ae.load_state_dict(torch.load(ROOT_DIR / "models" / "ae.pt", map_location=device))
ae.eval()

mtcnn = MTCNN(
    image_size=160,
    margin=10,
    keep_all=True,
    post_process=True,
    device=device,
)

cnn_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

ae_transform = transforms.Compose(
    [
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
    ]
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def allowed_file(filename: str, allowed_exts) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_exts


def combine_scores(prob_fake: float, ae_loss: float) -> float:
    # Normalize AE error — based on actual observed range
    # Real images typically have lower reconstruction error
    # Fake images have higher reconstruction error
    # Map to 0-1 scale where higher = more fake-like
    # Using a more conservative normalization
    ae_norm = min(max((ae_loss - 0.10) / 0.15, 0), 1)
    # Rely more on CNN (0.90) - it's the better classifier
    # AE helps as secondary signal (0.10)
    return (prob_fake * 0.90) + (ae_norm * 0.10)


def analyze_faces(pil_image: Image.Image, slug: str, raise_on_empty: bool = True):
    faces_tensor, probs = mtcnn(pil_image, return_prob=True)
    boxes, _ = mtcnn.detect(pil_image)

    if faces_tensor is None or len(faces_tensor) == 0:
        if raise_on_empty:
            raise NoFaceFoundError("No face detected.")
        return []

    faces = []
    with torch.no_grad():
        for idx, face_tensor in enumerate(faces_tensor):
            face_pil = transforms.ToPILImage()(face_tensor)

            cnn_input = cnn_transform(face_pil).unsqueeze(0).to(device)
            cnn_out = cnn(cnn_input)
            prob_fake = torch.softmax(cnn_out, dim=1)[0][1].item()

            ae_input = ae_transform(face_pil).unsqueeze(0).to(device)
            ae_recon = ae(ae_input)
            ae_loss = torch.mean(torch.abs(ae_input - ae_recon)).item()

            final_score = combine_scores(prob_fake, ae_loss)
            # Lower threshold to catch more fakes - 0.15 instead of 0.25
            # If CNN says >15% fake probability, mark as FAKE
            verdict = "FAKE" if final_score > 0.15 else "REAL"

            bbox = None
            if boxes is not None and len(boxes) > idx and boxes[idx] is not None:
                bbox = [int(x) for x in boxes[idx]]

            faces.append(
                {
                    "id": f"{slug}-face-{idx}",
                    "bbox": bbox,
                    "prob_fake": round(float(prob_fake), 4),
                    "ae_loss": round(float(ae_loss), 5),
                    "final_score": round(float(final_score), 4),
                    "verdict": verdict,
                    "confidence": round(float(probs[idx]) if probs is not None else 0, 4),
                }
            )

    return faces


def summarize_faces(faces):
    if not faces:
        return {"final_verdict": "NO_FACE", "aggregate_score": None, "face_count": 0}

    scores = [face["final_score"] for face in faces]
    avg_score = mean(scores)
    verdict_counts = {"FAKE": 0, "REAL": 0}
    for face in faces:
        verdict_counts[face["verdict"]] += 1

    # Use same threshold as individual face detection
    final_verdict = "FAKE" if avg_score > 0.15 else "REAL"
    if verdict_counts["FAKE"] and verdict_counts["REAL"]:
        final_verdict = "MIXED"

    return {
        "final_verdict": final_verdict,
        "aggregate_score": round(avg_score, 4),
        "face_count": len(faces),
        "fake_faces": verdict_counts["FAKE"],
        "real_faces": verdict_counts["REAL"],
    }


def run_video_pipeline(video_path: Path, slug: str):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Unable to read the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    frame_interval = max(int(fps // 2), 1)  # sample twice per second
    timeline = []
    frame_id = 0
    processed_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frame_interval != 0:
                frame_id += 1
                continue

            processed_frames += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            timestamp = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2)

            faces = analyze_faces(
                pil_frame, f"{slug}_frame{frame_id}", raise_on_empty=False
            )

            frame_score = (
                round(mean(face["final_score"] for face in faces), 4) if faces else None
            )
            frame_verdict = "NO_FACE"
            if frame_score is not None:
                frame_verdict = "FAKE" if frame_score > 0.15 else "REAL"

            timeline.append(
                {
                    "timestamp": timestamp,
                    "score": frame_score,
                    "verdict": frame_verdict,
                    "face_count": len(faces),
                }
            )

            frame_id += 1
    finally:
        cap.release()

    scored_frames = [entry["score"] for entry in timeline if entry["score"] is not None]
    final_score = round(mean(scored_frames), 4) if scored_frames else None
    final_verdict = "NO_FACE"
    if final_score is not None:
        final_verdict = "FAKE" if final_score > 0.15 else "REAL"

    return {
        "timeline": timeline,
        "frames_analyzed": processed_frames,
        "final_score": final_score,
        "final_verdict": final_verdict,
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html", edition_date=datetime.utcnow())


@app.route("/detect-image", methods=["POST"])
def detect_image():
    image_file = request.files.get("image")
    mode = request.form.get("mode", "single")

    if image_file is None or image_file.filename == "":
        return jsonify({"status": "error", "message": "Please upload an image."}), 400

    if not allowed_file(image_file.filename, ALLOWED_IMAGE_EXT):
        return (
            jsonify({"status": "error", "message": "Unsupported image format."}),
            400,
        )

    filename = secure_filename(image_file.filename)
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
    image_file.save(save_path)

    try:
        pil_image = Image.open(save_path).convert("RGB")
        faces = analyze_faces(pil_image, Path(filename).stem)
        summary = summarize_faces(faces)

        # For single mode, only keep top face
        if mode == "single" and faces:
            faces = [faces[0]]

        response = {
            "status": "success",
            "summary": summary,
            "faces": faces,
        }
        return jsonify(response)
    except NoFaceFoundError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 404
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"status": "error", "message": str(exc)}), 500
    finally:
        if save_path.exists():
            save_path.unlink()


@app.route("/detect-video", methods=["POST"])
def detect_video():
    video_file = request.files.get("video")
    if video_file is None or video_file.filename == "":
        return jsonify({"status": "error", "message": "Please upload a video."}), 400

    if not allowed_file(video_file.filename, ALLOWED_VIDEO_EXT):
        return (
            jsonify({"status": "error", "message": "Unsupported video format."}),
            400,
        )

    filename = secure_filename(video_file.filename)
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
    video_file.save(save_path)

    try:
        payload = run_video_pipeline(save_path, Path(filename).stem)
        return jsonify({"status": "success", **payload})
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"status": "error", "message": str(exc)}), 500
    finally:
        if save_path.exists():
            save_path.unlink()


# --------------------------------------------------------------------------- #
# Entry Point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    app.run(host=host, port=port, debug=False)


