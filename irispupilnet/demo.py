"""
IrisPupilNet Demo - iris and pupil segmentation inference.

Supports webcam, video files, image files, and image directories.
Requires an ONNX model exported with export_onnx.py.

Usage:
    # Webcam
    python -m irispupilnet.demo --model model.onnx --source 0 --color false

    # Video file
    python -m irispupilnet.demo --model model.onnx --source video.mp4 --save-video out.mp4

    # Image directory
    python -m irispupilnet.demo --model model.onnx --source images/ --save-frames out/

Install:
    pip install opencv-python mediapipe numpy onnxruntime
    # For GPU: pip install onnxruntime-gpu
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

import mediapipe as mp

# MediaPipe FaceMesh eye landmark indices
LEFT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

# Colors (BGR)
C_EYE  = (255, 200, 0)
C_HEAD = (0, 200, 255)
C_TXT  = (255, 255, 255)

DEFAULT_IMG_SIZE = 160
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Segmentation overlay colors (BGR): iris=green, pupil=red
IRIS_COLOR  = (0, 255, 0)
PUPIL_COLOR = (0, 0, 255)


# ---------------------------------------------------------------------------
# ONNX helpers
# ---------------------------------------------------------------------------

def load_onnx_session(model_path):
    """Load ONNX model and auto-detect input size and channels.

    Returns:
        (session, img_size, input_channels)
    """
    if ort is None:
        raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
    except Exception:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    # Expected shape: [batch, H, W, C] (NHWC, as exported by export_onnx.py)
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) == 4:
        _, h, w, c = input_shape
        if isinstance(h, str): h = DEFAULT_IMG_SIZE
        if isinstance(w, str): w = DEFAULT_IMG_SIZE
        if isinstance(c, str): c = 1
        print(f"  Detected ONNX input: [batch, {h}, {w}, {c}] (NHWC)")
        return session, int(h), int(c)
    else:
        print(f"  Warning: unexpected input shape {input_shape}, using defaults")
        return session, DEFAULT_IMG_SIZE, 1


def run_segmentation(sess, crop_bgr, input_channels):
    """Run ONNX inference on a BGR crop.

    Args:
        crop_bgr: BGR image array [H, W, 3]
        input_channels: 1 for grayscale, 3 for RGB

    Returns:
        mask [H, W] with values 0=bg, 1=iris, 2=pupil
    """
    if crop_bgr is None:
        return None
    if input_channels == 1:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        inp_img = gray[..., None]  # [H, W, 1]
    else:
        inp_img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)  # [H, W, 3]
    inp = (inp_img.astype(np.float32) / 255.0)[None, ...]    # [1, H, W, C]
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    out = sess.run([output_name], {input_name: inp})[0]      # [1, H, W, num_classes]
    return np.argmax(out[0], axis=-1).astype(np.uint8)       # [H, W]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def overlay_mask(img_bgr, mask, alpha=0.5):
    """Blend iris (green) and pupil (red) overlay onto a BGR image, with contours."""
    h, w = img_bgr.shape[:2]
    mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    result = img_bgr.copy()
    color_layer = np.zeros_like(img_bgr)
    color_layer[mask_r == 1] = IRIS_COLOR
    color_layer[mask_r == 2] = PUPIL_COLOR
    has_mask = mask_r > 0
    blended = cv2.addWeighted(img_bgr, 1 - alpha, color_layer, alpha, 0)
    result[has_mask] = blended[has_mask]
    # Draw contours
    for cls, color in [(1, (0, 200, 0)), (2, (0, 0, 200))]:
        cls_mask = (mask_r == cls).astype(np.uint8)
        contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 1)
    return result


def _label_tile(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TXT, 1, cv2.LINE_AA)
    return out


def make_eye_grid(left_raw, right_raw, left_seg, right_seg, tile=240):
    """Build a 2x2 grid: [raw_left, raw_right; seg_left, seg_right]."""
    def fit(img):
        return cv2.resize(img, (tile, tile), interpolation=cv2.INTER_LINEAR)
    tl = _label_tile(left_raw,  "Left eye")
    tr = _label_tile(right_raw, "Right eye")
    bl = _label_tile(left_seg,  "Left seg")
    br = _label_tile(right_seg, "Right seg")
    return np.vstack([np.hstack([fit(tl), fit(tr)]), np.hstack([fit(bl), fit(br)])])


# ---------------------------------------------------------------------------
# MediaPipe helpers
# ---------------------------------------------------------------------------

def bbox_from_landmarks(lm, idxs, W, H, pad=0.25):
    xs, ys = [], []
    for i in idxs:
        if i < len(lm):
            xs.append(lm[i].x)
            ys.append(lm[i].y)
    if not xs:
        return None
    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
    x0, y0, x1, y1 = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    px, py = int(bw * pad), int(bh * pad)
    x = max(0, x0 - px)
    y = max(0, y0 - py)
    return (x, y, min(W - x, bw + 2 * px), min(H - y, bh + 2 * py))


def head_bbox(lm, W, H, pad=0.15):
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
    x0, y0, x1, y1 = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    px, py = int(bw * pad), int(bh * pad)
    x = max(0, x0 - px)
    y = max(0, y0 - py)
    return (x, y, min(W - x, bw + 2 * px), min(H - y, bh + 2 * py))


def crop_to_size(frame, rect, size):
    x, y, w, h = rect
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Core frame processor
# ---------------------------------------------------------------------------

def process_frame(frame, mesh, session, img_size, input_channels, mirror=False):
    """Detect eyes, run segmentation, return annotated frame and eye grid.

    Returns:
        (annotated_frame, eye_grid)
    """
    if mirror:
        frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    annotated = frame.copy()

    left_rect = right_rect = None
    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        hbox = head_bbox(lm, W, H)
        cv2.rectangle(annotated, (hbox[0], hbox[1]),
                      (hbox[0] + hbox[2], hbox[1] + hbox[3]), C_HEAD, 2)
        left_rect  = bbox_from_landmarks(lm, LEFT_EYE,  W, H, pad=0.25)
        right_rect = bbox_from_landmarks(lm, RIGHT_EYE, W, H, pad=0.25)

    # Fallback eye boxes when detection fails
    if left_rect is None or right_rect is None:
        boxW, boxH = int(W * 0.18), int(H * 0.18)
        cx, cy = W // 2, int(H * 0.42)
        gap = int(boxW * 0.6)
        left_rect  = left_rect  or (cx - gap - boxW // 2, cy - boxH // 2, boxW, boxH)
        right_rect = right_rect or (cx + gap - boxW // 2, cy - boxH // 2, boxW, boxH)

    for (x, y, bw, bh) in (left_rect, right_rect):
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), C_EYE, 2)

    blank = np.zeros((img_size, img_size, 3), np.uint8)
    left_crop  = crop_to_size(annotated, left_rect,  img_size) or blank
    right_crop = crop_to_size(annotated, right_rect, img_size) or blank

    if session is not None:
        lmask = run_segmentation(session, left_crop,  input_channels)
        rmask = run_segmentation(session, right_crop, input_channels)
        left_seg  = overlay_mask(left_crop,  lmask) if lmask  is not None else left_crop
        right_seg = overlay_mask(right_crop, rmask) if rmask is not None else right_crop
    else:
        left_seg  = blank.copy()
        right_seg = blank.copy()

    grid = make_eye_grid(left_crop, right_crop, left_seg, right_seg)
    return annotated, grid


# ---------------------------------------------------------------------------
# Source runners
# ---------------------------------------------------------------------------

def _make_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )


def run_webcam(args, session, img_size, input_channels):
    mesh = _make_mesh()
    cap = cv2.VideoCapture(int(args.source))
    if not cap.isOpened():
        print(f"ERROR: could not open camera {args.source}", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Press Q or ESC to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, grid = process_frame(frame, mesh, session, img_size, input_channels, mirror=True)
        cv2.imshow("IrisPupilNet - Camera", annotated)
        cv2.imshow("IrisPupilNet - Eyes",   grid)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


def run_video(args, session, img_size, input_channels):
    mesh = _make_mesh()
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"ERROR: could not open video {args.source}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_video:
        Path(args.save_video).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (fw, fh))
        print(f"Saving video to: {args.save_video}")

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, grid = process_frame(frame, mesh, session, img_size, input_channels)
        n += 1
        if writer:
            writer.write(annotated)
        if not args.no_display:
            cv2.imshow("IrisPupilNet - Video", annotated)
            cv2.imshow("IrisPupilNet - Eyes",  grid)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    print(f"Processed {n} frames.")
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def run_images(args, session, img_size, input_channels):
    mesh = _make_mesh()
    src = Path(args.source)
    if src.is_file():
        image_paths = [src]
    else:
        image_paths = sorted(p for p in src.iterdir()
                             if p.suffix.lower() in IMAGE_EXTENSIONS)

    if not image_paths:
        print(f"No images found in {args.source}", file=sys.stderr)
        sys.exit(1)

    save_dir = None
    if args.save_frames:
        save_dir = Path(args.save_frames)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving frames to: {save_dir}")

    print(f"Processing {len(image_paths)} image(s)."
          + ("" if args.no_display else " Press any key to advance, Q/ESC to quit."))

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  Skipping unreadable: {img_path}")
            continue
        annotated, grid = process_frame(frame, mesh, session, img_size, input_channels)
        if save_dir:
            out_path = save_dir / img_path.name
            cv2.imwrite(str(out_path), annotated)
            print(f"  Saved: {out_path}")
        if not args.no_display:
            cv2.imshow("IrisPupilNet - Image", annotated)
            cv2.imshow("IrisPupilNet - Eyes",  grid)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_bool(v):
    return v.lower() not in ("false", "0", "no")


def _is_webcam(source):
    try:
        int(source)
        return True
    except ValueError:
        return False


def _is_video(source):
    return Path(source).is_file() and Path(source).suffix.lower() not in IMAGE_EXTENSIONS


def main():
    parser = argparse.ArgumentParser(
        description="IrisPupilNet iris/pupil segmentation demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",       type=str, required=True,
                        help="Path to ONNX model")
    parser.add_argument("--source",      type=str, default="0",
                        help="Camera index, video file, image file, or image directory")
    parser.add_argument("--img-size",    type=int, default=None,
                        help="Override model input size (auto-detected from ONNX)")
    parser.add_argument("--color",       type=_parse_bool, default=None,
                        help="true=RGB input, false=grayscale (auto-detected from ONNX if omitted)")
    parser.add_argument("--save-video",  type=str, default=None,
                        help="Save annotated video to this path (video source only)")
    parser.add_argument("--save-frames", type=str, default=None,
                        help="Save annotated frames to this directory (image source only)")
    parser.add_argument("--no-display",  action="store_true",
                        help="Disable GUI windows (useful with --save-video/--save-frames)")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading ONNX model: {args.model}")
    session, detected_size, detected_channels = load_onnx_session(args.model)

    img_size = args.img_size if args.img_size is not None else detected_size
    if args.color is None:
        input_channels = detected_channels
    else:
        input_channels = 3 if args.color else 1

    print(f"  Image size    : {img_size}x{img_size}")
    print(f"  Input channels: {input_channels} ({'RGB' if input_channels == 3 else 'grayscale'})")

    if _is_webcam(args.source):
        run_webcam(args, session, img_size, input_channels)
    elif _is_video(args.source):
        run_video(args, session, img_size, input_channels)
    else:
        run_images(args, session, img_size, input_channels)


if __name__ == "__main__":
    main()
