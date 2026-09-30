# CheatingDetector Full

Hệ thống Flask + YOLO26X + ByteTrack + YuNet + SFace + Head Pose + Temporal Event + Dashboard.

## Models
Đặt vào `models/`:
- yolo26x.pt
- face_detection_yunet_2026may.onnx
- face_recognition_sface_2021dec.onnx

## Chạy
```powershell
cd D:\BTL_HK1_2026-2027\PTHTTM\CheatingDetector_Full
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```
Mở `http://127.0.0.1:5000`.


