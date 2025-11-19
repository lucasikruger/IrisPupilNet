"""
Eye & Iris/Pupil Demo (OpenCV + MediaPipe FaceMesh + optional ONNX)

Features
- Default MIRROR preview (use --no-mirror to disable).
- Draws a head bounding box + both eye boxes on the main camera window.
- If --model is provided, runs ONNX segmentation with auto-detected config.
- Opens a second "Eyes Grid" window:
    Top-left  : Left eye (no seg)
    Top-right : Right eye (no seg)
    Bottom-left  : Left eye (seg overlay if model)
    Bottom-right : Right eye (seg overlay if model)

Controls
- Q / ESC to quit.

Install
    pip install opencv-python mediapipe==0.10.14 numpy
    # optional for segmentation:
    pip install onnxruntime    # or onnxruntime-gpu if you have CUDA

Simple Usage (auto-detects config from model)
    python demo/webcam_demo.py
    python demo/webcam_demo.py --model model.onnx
    python demo/webcam_demo.py --model runs/experiment/model.onnx --no-mirror

Advanced Usage (with manual overrides)
    python demo/webcam_demo.py --model model.onnx --rgb
    python demo/webcam_demo.py --model model.onnx --bw --img-size 224
"""

import argparse, sys, cv2, numpy as np

# ---- optional segmentation (only if --model is passed)
try:
    import onnxruntime as ort
except Exception:
    ort = None

import mediapipe as mp

# Landmark indices (same as the web version)
LEFT_EYE  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE = [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]

# Colors (BGR)
C_EYE  = (255,200,0)   # eye boxes
C_HEAD = (0,200,255)   # head box
C_TXT  = (255,255,255)

DEFAULT_IMG_SIZE = 160  # default model input size (HxW)

def bbox_from_landmarks(lm, idxs, W, H, pad=0.25):
    xs, ys = [], []
    for i in idxs:
        if i < len(lm):
            xs.append(lm[i].x); ys.append(lm[i].y)
    if not xs: return None
    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
    x0, y0 = int(x0*W), int(y0*H); x1, y1 = int(x1*W), int(y1*H)
    w, h = max(1, x1-x0), max(1, y1-y0)
    px, py = int(w*pad), int(h*pad)
    x = max(0, x0-px); y = max(0, y0-py)
    w = min(W-x, w+2*px); h = min(H-y, h+2*py)
    return (x,y,w,h)

def head_bbox(lm, W, H, pad=0.15):
    xs = [p.x for p in lm]; ys = [p.y for p in lm]
    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
    x0, y0 = int(x0*W), int(y0*H); x1, y1 = int(x1*W), int(y1*H)
    w, h = max(1, x1-x0), max(1, y1-y0)
    px, py = int(w*pad), int(h*pad)
    x = max(0, x0-px); y = max(0, y0-py)
    w = min(W-x, w+2*px); h = min(H-y, h+2*py)
    return (x,y,w,h)

def crop_resize_bgr(frame, rect, size):
    x, y, w, h = rect
    crop = frame[y:y+h, x:x+w, :]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

def load_onnx_session(model_path):
    """
    Load ONNX session and automatically detect input shape and channels.

    Returns:
        tuple: (session, img_size, input_channels)
    """
    if ort is None:
        raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime")
    # Try GPU, fall back to CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
    except Exception:
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])

    # Auto-detect input shape: expected [batch, height, width, channels] (NHWC)
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) == 4:
        _, height, width, channels = input_shape
        # Handle dynamic batch dimension
        if isinstance(height, str):
            height = DEFAULT_IMG_SIZE
        if isinstance(width, str):
            width = DEFAULT_IMG_SIZE
        if isinstance(channels, str):
            channels = 1
        img_size = height  # Assuming square images
        input_channels = channels
        print(f"  Detected input shape: [batch, {height}, {width}, {channels}]")
    else:
        print(f"  Warning: Unexpected input shape {input_shape}, using defaults")
        img_size = DEFAULT_IMG_SIZE
        input_channels = 1

    return session, img_size, input_channels

def run_segmentation(sess, crop_bgr, input_channels=1):
    """crop_bgr: 160x160x3 BGR -> [160,160] mask 0:bg,1:iris,2:pupil"""
    if crop_bgr is None:
        return None
    if input_channels == 1:
        crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)[..., None]
    else:
        crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    inp = (crop.astype(np.float32) / 255.0)[None, ...]
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    out = sess.run([output_name], {input_name: inp})[0]  # [1,H,W,3]
    logits = out[0]                                      # [H,W,3]
    return np.argmax(logits, axis=-1).astype(np.uint8)   # [H,W]

def overlay_mask_on_image(img_bgr, mask, alpha=0.45):
    """Return an overlayed BGR image; mask values: 0 bg, 1 iris (green), 2 pupil (blue-ish)."""
    h, w = img_bgr.shape[:2]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = np.zeros_like(img_bgr)
    iris  = mask == 1
    pupil = mask == 2
    overlay[iris]  = (0, 180, 0)    # green-ish
    overlay[pupil] = (220, 0, 0)    # blue-ish (BGR; tweak if you want pure blue)
    mixed = img_bgr.copy()
    cv2.addWeighted(overlay, alpha, mixed, 1 - alpha, 0, dst=mixed)
    return mixed

def label_tile(img, text):
    out = img.copy()
    cv2.rectangle(out, (0,0), (out.shape[1], 28), (0,0,0), -1)
    cv2.putText(out, text, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TXT, 1, cv2.LINE_AA)
    return out

def make_grid(tl, tr, bl, br, tile_wh=(240,240)):
    tw, th = tile_wh
    def fit(img):
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
    top = np.hstack([fit(tl), fit(tr)])
    bot = np.hstack([fit(bl), fit(br)])
    return np.vstack([top, bot])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--no-mirror", action="store_true", help="disable selfie mirror view")
    ap.add_argument("--model", type=str, default="", help="path to ONNX model (optional)")
    ap.add_argument("--tile", type=int, default=240, help="per-tile size for grid window")

    # Optional overrides (auto-detected from model if not provided)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--rgb", dest="input_channels", action="store_const", const=3,
                       help="Override: Run segmentation with RGB input (3 channels)")
    group.add_argument("--bw", dest="input_channels", action="store_const", const=1,
                       help="Override: Run segmentation with grayscale input (1 channel)")
    ap.set_defaults(input_channels=None)  # Auto-detect if None
    ap.add_argument("--img-size", type=int, default=None,
                    help="Override: Model input size (HxW), auto-detected if not provided")
    args = ap.parse_args()

    # MediaPipe FaceMesh
    mp_face = mp.solutions.face_mesh
    mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Camera
    # On Windows, if you get black preview, try cv2.CAP_DSHOW as second arg.
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("ERROR: could not open camera", file=sys.stderr); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Optional model - auto-detect config
    session = None
    img_size = args.img_size or DEFAULT_IMG_SIZE
    input_channels = args.input_channels or 1

    if args.model:
        try:
            session, detected_size, detected_channels = load_onnx_session(args.model)
            print(f"Loaded ONNX model: {args.model}")

            # Use detected values if not overridden by CLI
            if args.img_size is None:
                img_size = detected_size
                print(f"  Using detected image size: {img_size}")
            else:
                print(f"  Using CLI override image size: {img_size}")

            if args.input_channels is None:
                input_channels = detected_channels
                channel_type = "RGB" if input_channels == 3 else "grayscale"
                print(f"  Using detected input channels: {input_channels} ({channel_type})")
            else:
                channel_type = "RGB" if input_channels == 3 else "grayscale"
                print(f"  Using CLI override input channels: {input_channels} ({channel_type})")
        except Exception as e:
            print(f"Could not load model: {e}")

    print("Press Q or ESC to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed."); break
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)

        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)

        left_rect = right_rect = None
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            head = head_bbox(lm, W, H)
            if head: cv2.rectangle(frame, (head[0], head[1]), (head[0]+head[2], head[1]+head[3]), C_HEAD, 2)

            left_rect  = bbox_from_landmarks(lm, LEFT_EYE,  W, H, pad=0.25)
            right_rect = bbox_from_landmarks(lm, RIGHT_EYE, W, H, pad=0.25)

        # Fallback if eyes not detected
        if left_rect is None or right_rect is None:
            boxW = int(W * 0.18); boxH = int(H * 0.18)
            cx, cy = W // 2, int(H * 0.42)
            gap = int(boxW * 0.6)
            left_rect  = left_rect  or (cx - gap - boxW//2, cy - boxH//2, boxW, boxH)
            right_rect = right_rect or (cx + gap - boxW//2, cy - boxH//2, boxW, boxH)

        # draw eye boxes on main frame
        for (x,y,bw,bh) in (left_rect, right_rect):
            cv2.rectangle(frame, (x,y), (x+bw, y+bh), C_EYE, 2)

        # crops
        left_crop  = crop_resize_bgr(frame, left_rect,  img_size)
        right_crop = crop_resize_bgr(frame, right_rect, img_size)

        # build grid tiles
        tl = label_tile(left_crop  if left_crop  is not None else np.zeros((img_size,img_size,3), np.uint8), "Left eye")
        tr = label_tile(right_crop if right_crop is not None else np.zeros((img_size,img_size,3), np.uint8), "Right eye")

        if session is not None and left_crop is not None:
            l_mask = run_segmentation(session, left_crop, input_channels=input_channels)
            bl_img = overlay_mask_on_image(left_crop, l_mask) if l_mask is not None else left_crop
            bl = label_tile(bl_img, "Left seg")
        else:
            bl = label_tile(np.zeros((img_size,img_size,3), np.uint8), "Left seg (no model)")

        if session is not None and right_crop is not None:
            r_mask = run_segmentation(session, right_crop, input_channels=input_channels)
            br_img = overlay_mask_on_image(right_crop, r_mask) if r_mask is not None else right_crop
            br = label_tile(br_img, "Right seg")
        else:
            br = label_tile(np.zeros((img_size,img_size,3), np.uint8), "Right seg (no model)")

        grid = make_grid(tl, tr, bl, br, tile_wh=(args.tile, args.tile))

        # show windows
        cv2.imshow("Camera (head + eye boxes)", frame)
        cv2.imshow("Eyes Grid (2x2)", grid)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
