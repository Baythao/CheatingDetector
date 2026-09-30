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

## Pipeline
Camera/video -> YOLO26X -> ByteTrack -> YuNet/SFace -> MSSV/tên/ghế -> Head Pose -> Direction -> Temporal 2s -> Events; đồng thời YOLO phát hiện phone/book/laptop.

## API Phase 3 -> Phase 4
Phase 3D tối thiểu truyền:
```python
{
  'track_id': 1,
  'student_id': '1234567',
  'name': 'Thào A Bảy',
  'seat': 'A01',
  'bbox': (x1,y1,x2,y2),
  'face': face
}
```
Bản `core.py` hiện thực toàn pipeline để chạy độc lập; khi ghép Phase 4 tách riêng, giữ đúng contract trên để Phase 4 không chạy lại YOLO/ByteTrack/SFace.

## Lưu ý
COCO nhận phone/book/laptop. `paper/scratch paper` cần model custom; không nên giả mạo class book thành giấy nháp.
Head pose là hướng đầu, không phải eye/iris gaze.
Các event là tín hiệu cần xem xét, không tự động kết luận gian lận.
