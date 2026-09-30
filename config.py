from pathlib import Path
import torch


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "yolo26x.pt"
)

FACE_DETECTOR_MODEL = (
    BASE_DIR
    / "models"
    / "face_detection_yunet_2026may.onnx"
)

FACE_RECOGNITION_MODEL = (
    BASE_DIR
    / "models"
    / "face_recognition_sface_2021dec.onnx"
)


# ============================================================
# DATA
# ============================================================

STUDENT_DB = (
    BASE_DIR
    / "data"
    / "students"
    / "students.json"
)

STUDENT_IMAGES = (
    BASE_DIR
    / "data"
    / "students"
    / "images"
)

UPLOAD_DIR = (
    BASE_DIR
    / "uploads"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
)


# ============================================================
# GPU
# ============================================================

CUDA_AVAILABLE = torch.cuda.is_available()

DEVICE = (
    0
    if CUDA_AVAILABLE
    else "cpu"
)

# NVIDIA GPU -> FP16
HALF = CUDA_AVAILABLE


# ============================================================
# YOLO PERFORMANCE
# ============================================================

# AI xử lý mỗi N frame.
#
# 30 FPS camera:
#
# 10 -> ~3 AI FPS
# 5  -> ~6 AI FPS
# 3  -> ~10 AI FPS
# 1  -> ~30 AI FPS
#
# Dùng 5 để head pose LEFT/RIGHT phản hồi nhanh hơn.

PROCESS_EVERY_N_FRAMES = 5


# ============================================================
# YOLO IMAGE SIZE
# ============================================================

# 416 nhanh hơn 640 đáng kể.
IMG_SIZE = 416


# ============================================================
# YOLO DETECTION
# ============================================================

CONFIDENCE = 0.35

IOU = 0.45


# ============================================================
# YOLO CLASSES
# ============================================================

# COCO:
#
# 0  = person
# 63 = laptop
# 67 = cell phone
# 73 = book

DETECTION_CLASSES = [
    0,      # person
    63,     # laptop
    67,     # phone
    73      # book
]


# ============================================================
# FACE RECOGNITION
# ============================================================

# Cosine similarity.
#
# >= 0.40 -> nhận diện là sinh viên tương ứng.

FACE_MATCH_THRESHOLD = 0.40


# Nhận diện khuôn mặt lại tối đa 1 lần / 2 giây.
#
# Không cần nhận diện SFace ở mọi frame vì
# ByteTrack đã giữ track_id.

FACE_RECOGNITION_INTERVAL = 2.0


# ============================================================
# HEAD POSE
# ============================================================

# Thời gian nhìn LEFT/RIGHT để tạo cảnh báo.

LOOKING_AWAY_SECONDS = 2.0


# ------------------------------------------------------------
# YAW THRESHOLD
# ------------------------------------------------------------

# Khi yaw đạt ngưỡng này:
#
# yaw <= -20 -> LEFT
# yaw >= +20 -> RIGHT

HEAD_YAW_THRESHOLD = 20.0


# ------------------------------------------------------------
# HYSTERESIS
# ------------------------------------------------------------

# Sau khi đã LEFT:
#
# yaw vẫn < -12 -> giữ LEFT
# yaw >= -12 -> CENTER
#
# Sau khi đã RIGHT:
#
# yaw vẫn > +12 -> giữ RIGHT
# yaw <= +12 -> CENTER

HEAD_YAW_CENTER_THRESHOLD = 12.0


# ------------------------------------------------------------
# PITCH
# ------------------------------------------------------------

# Góc cúi/ngẩng đầu.
#
# Hiện tại hệ thống vẫn trả pitch,
# chưa dùng pitch để thay thế LEFT/RIGHT.

HEAD_PITCH_THRESHOLD = 15.0


# ------------------------------------------------------------
# YAW SMOOTHING
# ------------------------------------------------------------

# EMA smoothing.
#
# 0.35:
# - giảm rung
# - vẫn phản hồi tương đối nhanh
#
# Tăng lên 0.5 nếu muốn phản ứng nhanh hơn.
# Giảm xuống 0.2 nếu muốn ổn định hơn.

HEAD_YAW_SMOOTH_ALPHA = 0.35


# ------------------------------------------------------------
# DIRECTION STABILITY
# ------------------------------------------------------------

# LEFT/RIGHT phải xuất hiện liên tiếp N lần AI
# mới được xác nhận.

HEAD_DIRECTION_STABLE_FRAMES = 3


# ============================================================
# OBJECT CLASSES
# ============================================================

PHONE_CLASS_IDS = {
    67
}

BOOK_CLASS_IDS = {
    73
}

LAPTOP_CLASS_IDS = {
    63
}


# ============================================================
# DIRECTORIES
# ============================================================

DIRECTORIES = [
    STUDENT_DB.parent,
    STUDENT_IMAGES,
    UPLOAD_DIR,
    OUTPUT_DIR
]


for path in DIRECTORIES:

    path.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CREATE STUDENT DATABASE
# ============================================================

if not STUDENT_DB.exists():

    STUDENT_DB.write_text(
        "[]",
        encoding="utf-8"
    )


# ============================================================
# DEBUG / CONFIG INFO
# ============================================================

print("=" * 60)
print("CHEATING DETECTOR CONFIG")
print("=" * 60)

print(
    f"CUDA available : "
    f"{CUDA_AVAILABLE}"
)

print(
    f"Device         : "
    f"{DEVICE}"
)

print(
    f"FP16           : "
    f"{HALF}"
)

if CUDA_AVAILABLE:

    print(
        f"GPU            : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA version   : "
        f"{torch.version.cuda}"
    )

print(
    f"Model          : "
    f"{MODEL_PATH.name}"
)

print(
    f"Image size     : "
    f"{IMG_SIZE}"
)

print(
    f"Confidence     : "
    f"{CONFIDENCE}"
)

print(
    f"IOU            : "
    f"{IOU}"
)

print(
    f"Frame skip     : "
    f"{PROCESS_EVERY_N_FRAMES}"
)

print(
    f"Face interval  : "
    f"{FACE_RECOGNITION_INTERVAL}s"
)

print(
    f"Yaw threshold  : "
    f"{HEAD_YAW_THRESHOLD}°"
)

print(
    f"Yaw center     : "
    f"{HEAD_YAW_CENTER_THRESHOLD}°"
)

print(
    f"Yaw smoothing  : "
    f"{HEAD_YAW_SMOOTH_ALPHA}"
)

print(
    f"Stable frames  : "
    f"{HEAD_DIRECTION_STABLE_FRAMES}"
)

print(
    f"Long looking   : "
    f"{LOOKING_AWAY_SECONDS}s"
)

print(
    f"Classes        : "
    f"{DETECTION_CLASSES}"
)

print("=" * 60)