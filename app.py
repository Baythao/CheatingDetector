import threading
import time
import cv2

from pathlib import Path
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from config import *
from core import DB, Monitor


app = Flask(
    __name__,
    template_folder="app/templates"
)

db = DB()
monitor = Monitor(db)


S = {
    "mode": "camera",
    "cap": None,
    "video_name": "",
    "finished": False,
    "gen": 0,
    "jpg": None,
    "tracks": [],
    "lock": threading.Lock(),
}


def release():
    if S["cap"] is not None:
        S["cap"].release()
        S["cap"] = None


def set_camera():
    release()

    c = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not c.isOpened():
        c = cv2.VideoCapture(0)

    if not c.isOpened():
        raise RuntimeError("Không mở được webcam")

    S.update(
        cap=c,
        mode="camera",
        video_name="",
        finished=False,
        gen=S["gen"] + 1,
    )


def set_video(path, name):
    release()

    c = cv2.VideoCapture(str(path))

    if not c.isOpened():
        raise RuntimeError("Không mở được video")

    S.update(
        cap=c,
        mode="video",
        video_name=name,
        finished=False,
        gen=S["gen"] + 1,
    )


def loop():
    while True:
        c = S["cap"]

        if c is None:
            time.sleep(0.05)
            continue

        ok, frame = c.read()

        if not ok:
            if S["mode"] == "video":
                S["finished"] = True
                S["cap"] = None
                c.release()

            time.sleep(0.05)
            continue

        tracks = monitor.process(frame)
        frame = monitor.draw(frame, tracks)

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 82]
        )

        if ok:
            with S["lock"]:
                S["jpg"] = encoded.tobytes()
                S["tracks"] = tracks

        if S["mode"] == "video":
            fps = c.get(cv2.CAP_PROP_FPS) or 25
            time.sleep(1 / fps)


threading.Thread(
    target=loop,
    daemon=True
).start()


@app.get("/")
def home():
    return render_template("dashboard.html")


@app.get("/video_feed")
def feed():

    def generate():
        while True:
            with S["lock"]:
                jpg = S["jpg"]

            if jpg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpg
                    + b"\r\n"
                )

            time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/status")
def status():

    with S["lock"]:
        tracks = list(S["tracks"])

    students = db.public()

    return jsonify(
        success=True,
        source_generation=S["gen"],
        monitor_mode=S["mode"],
        camera_ready=S["cap"] is not None,
        video_name=S["video_name"],
        video_finished=S["finished"],
        persons=monitor.persons,
        recognized=monitor.recognized,
        students=students,
        tracks=tracks,
        phones_data=monitor.objects,
        events=monitor.events,
        fps=monitor.fps,
    )


@app.post("/api/monitor/camera")
def camera():

    try:
        set_camera()

        return jsonify(
            success=True,
            message="Đã chuyển sang camera"
        )

    except Exception as e:
        return jsonify(
            success=False,
            message=str(e)
        ), 400


@app.post("/api/video/upload")
def upload():

    f = request.files.get("video")

    if not f:
        return jsonify(
            success=False,
            message="Chưa chọn video"
        ), 400

    ext = Path(f.filename).suffix.lower().lstrip(".")

    if ext not in {"mp4", "avi", "mov", "mkv"}:
        return jsonify(
            success=False,
            message="Định dạng video không hỗ trợ"
        ), 400

    name = secure_filename(f.filename)
    path = UPLOAD_DIR / name

    f.save(path)

    try:
        set_video(path, name)

        return jsonify(
            success=True,
            message=f"Đã tải {name} và bắt đầu xử lý"
        )

    except Exception as e:
        return jsonify(
            success=False,
            message=str(e)
        ), 400


@app.post("/api/students")
def add_student():

    try:
        student_id = request.form["student_id"].strip()
        name = request.form["name"].strip()
        seat = request.form["seat"].strip()

        image_file = request.files["image"]

        temp_path = UPLOAD_DIR / f"reg_{student_id}.jpg"
        image_file.save(temp_path)

        image = cv2.imread(str(temp_path))

        if image is None:
            raise ValueError("Không đọc được ảnh")

        faces = monitor.faces.faces(image)

        if len(faces) != 1:
            raise ValueError(
                f"Ảnh phải có đúng 1 khuôn mặt, hiện có {len(faces)}"
            )

        embedding = monitor.faces.emb(
            image,
            faces[0]
        )

        destination = STUDENT_IMAGES / f"{student_id}.jpg"

        cv2.imwrite(
            str(destination),
            image
        )

        db.add(
            {
                "student_id": student_id,
                "name": name,
                "seat": seat,
                "image": str(
                    destination.relative_to(BASE_DIR)
                ).replace("\\", "/"),
                "embedding": embedding.reshape(-1).astype(float).tolist(),
            }
        )

        temp_path.unlink(missing_ok=True)

        return jsonify(
            success=True,
            message=f"Đã thêm {name}"
        )

    except Exception as e:
        return jsonify(
            success=False,
            message=str(e)
        ), 400


@app.get("/api/students")
def students():
    return jsonify(
        success=True,
        students=db.public()
    )


@app.delete("/api/students/<sid>")
def delete_student(sid):

    db.delete(sid)

    return jsonify(
        success=True,
        message="Đã xóa thí sinh"
    )


@app.post("/api/warnings/clear")
def clear_warnings():

    monitor.events.clear()

    return jsonify(
        success=True,
        message="Đã xóa cảnh báo"
    )


if __name__ == "__main__":
    set_camera()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )