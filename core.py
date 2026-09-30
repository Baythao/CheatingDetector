import json
import time

import cv2
import numpy as np

from ultralytics import YOLO

from config import *


# ============================================================
# DATABASE
# ============================================================

class DB:

    def __init__(self):
        self.path = STUDENT_DB

    def all(self):

        if not self.path.exists():
            return []

        try:
            return json.loads(
                self.path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception as e:

            print(f"[DB ERROR] {e}")

            return []

    def public(self):

        return [
            {
                "student_id": student.get("student_id"),
                "name": student.get("name"),
                "seat": student.get("seat")
            }
            for student in self.all()
        ]

    def add(self, student):

        data = self.all()

        student_id = student.get("student_id")
        seat = student.get("seat")

        if any(
            item.get("student_id") == student_id
            for item in data
        ):

            raise ValueError(
                "MSSV đã tồn tại"
            )

        if any(
            item.get("seat") == seat
            for item in data
        ):

            raise ValueError(
                "Ghế đã được đăng ký"
            )

        data.append(student)

        self.path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

    def delete(self, student_id):

        data = [
            item
            for item in self.all()
            if item.get("student_id") != student_id
        ]

        self.path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )


# ============================================================
# FACE ENGINE
# ============================================================

class Faces:

    def __init__(self):

        print("Loading YuNet...")

        self.det = cv2.FaceDetectorYN.create(
            str(FACE_DETECTOR_MODEL),
            "",
            (320, 320),
            0.6,
            0.3,
            5000
        )

        print("Loading SFace...")

        self.rec = cv2.FaceRecognizerSF.create(
            str(FACE_RECOGNITION_MODEL),
            ""
        )

        print("Face models loaded.")

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    def faces(self, image):

        if (
            image is None
            or image.size == 0
        ):
            return []

        height, width = image.shape[:2]

        if height <= 0 or width <= 0:
            return []

        try:

            self.det.setInputSize(
                (width, height)
            )

            _, detected = self.det.detect(
                image
            )

        except Exception as e:

            print(
                f"[FACE DETECT ERROR] {e}"
            )

            return []

        if detected is None:
            return []

        return detected

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    def emb(
        self,
        image,
        face
    ):

        try:

            aligned = self.rec.alignCrop(
                image,
                face
            )

            return self.rec.feature(
                aligned
            )

        except Exception as e:

            print(
                f"[FACE EMBEDDING ERROR] {e}"
            )

            return None

    # --------------------------------------------------------
    # RECOGNITION
    # --------------------------------------------------------

    def recognize(
        self,
        image,
        students
    ):

        detected_faces = self.faces(
            image
        )

        if (
            detected_faces is None
            or len(detected_faces) == 0
        ):
            return None

        # ----------------------------------------------------
        # LẤY KHUÔN MẶT LỚN NHẤT
        # ----------------------------------------------------

        try:

            face = max(
                detected_faces,
                key=lambda item:
                float(item[2]) * float(item[3])
            )

        except Exception:

            return None

        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

        embedding = self.emb(
            image,
            face
        )

        if embedding is None:

            return {
                "face": face,
                "student": None,
                "similarity": 0.0
            }

        # ----------------------------------------------------
        # KHÔNG CÓ DATABASE
        # ----------------------------------------------------

        if not students:

            return {
                "face": face,
                "student": None,
                "similarity": 0.0
            }

        # ----------------------------------------------------
        # MATCH
        # ----------------------------------------------------

        best_student = None
        best_score = -1.0

        for student in students:

            try:

                reference = np.asarray(
                    student.get(
                        "embedding",
                        []
                    ),
                    dtype=np.float32
                ).reshape(
                    1,
                    -1
                )

                if reference.size == 0:
                    continue

                score = float(
                    self.rec.match(
                        embedding,
                        reference,
                        cv2.FaceRecognizerSF_FR_COSINE
                    )
                )

                if score > best_score:

                    best_score = score
                    best_student = student

            except Exception:

                continue

        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        matched_student = None

        if (
            best_student is not None
            and best_score >= FACE_MATCH_THRESHOLD
        ):

            matched_student = best_student

        return {
            "face": face,
            "student": matched_student,
            "similarity": float(best_score)
        }


# ============================================================
# HEAD POSE
# ============================================================

class Pose:

    def __init__(self):

        self.temporal = {}

        self.smooth = {}

        self.direction_state = {}

        self.pending_direction = {}

        self.pending_count = {}

    # ========================================================
    # ESTIMATE HEAD POSE
    # ========================================================

    def estimate(
        self,
        face,
        width,
        height,
        track_id=None
    ):

        if (
            face is None
            or len(face) < 14
            or width <= 0
            or height <= 0
        ):
            return None

        # ====================================================
        # YUNET LANDMARKS
        #
        # face[4:6]   RIGHT EYE
        # face[6:8]   LEFT EYE
        # face[8:10]  NOSE
        # face[10:12] RIGHT MOUTH
        # face[12:14] LEFT MOUTH
        # ====================================================

        right_eye = np.array(
            [
                face[4],
                face[5]
            ],
            dtype=np.float64
        )

        left_eye = np.array(
            [
                face[6],
                face[7]
            ],
            dtype=np.float64
        )

        nose = np.array(
            [
                face[8],
                face[9]
            ],
            dtype=np.float64
        )

        right_mouth = np.array(
            [
                face[10],
                face[11]
            ],
            dtype=np.float64
        )

        left_mouth = np.array(
            [
                face[12],
                face[13]
            ],
            dtype=np.float64
        )

        # ====================================================
        # 5 LANDMARKS
        # ====================================================

        points_2d = np.array(
            [
                right_eye,
                left_eye,
                nose,
                right_mouth,
                left_mouth
            ],
            dtype=np.float64
        )

        # ====================================================
        # VALIDATE
        # ====================================================

        if not np.isfinite(
            points_2d
        ).all():

            return None

        eye_distance = np.linalg.norm(
            right_eye - left_eye
        )

        if eye_distance < 8:

            return None

        # ====================================================
        # 3D FACE MODEL
        # ====================================================

        model_points = np.array(
            [
                # RIGHT EYE
                [
                    30.0,
                    -35.0,
                    -30.0
                ],

                # LEFT EYE
                [
                    -30.0,
                    -35.0,
                    -30.0
                ],

                # NOSE
                [
                    0.0,
                    0.0,
                    0.0
                ],

                # RIGHT MOUTH
                [
                    35.0,
                    35.0,
                    -20.0
                ],

                # LEFT MOUTH
                [
                    -35.0,
                    35.0,
                    -20.0
                ]
            ],
            dtype=np.float64
        )

        # ====================================================
        # CAMERA MATRIX
        # ====================================================

        focal_length = float(
            max(
                width,
                height
            )
        )

        camera_matrix = np.array(
            [
                [
                    focal_length,
                    0,
                    width / 2.0
                ],

                [
                    0,
                    focal_length,
                    height / 2.0
                ],

                [
                    0,
                    0,
                    1
                ]
            ],
            dtype=np.float64
        )

        distortion = np.zeros(
            (4, 1),
            dtype=np.float64
        )

        # ====================================================
        # SOLVE PNP
        #
        # SQPNP hỗ trợ 5 điểm.
        #
        # Không dùng ITERATIVE vì OpenCV 5 trên máy bạn
        # có thể đi qua DLT và yêu cầu >= 6 điểm.
        # ====================================================

        try:

            success, rotation_vector, translation_vector = (
                cv2.solvePnP(
                    model_points,
                    points_2d,
                    camera_matrix,
                    distortion,
                    flags=cv2.SOLVEPNP_SQPNP
                )
            )

        except Exception as e:

            print(
                f"[POSE ERROR] {e}"
            )

            return None

        if not success:

            return None

        # ====================================================
        # ROTATION MATRIX
        # ====================================================

        try:

            rotation_matrix, _ = cv2.Rodrigues(
                rotation_vector
            )

        except Exception as e:

            print(
                f"[RODRIGUES ERROR] {e}"
            )

            return None

        # ====================================================
        # EULER ANGLES
        # ====================================================

        sy = np.sqrt(
            rotation_matrix[0, 0] ** 2
            +
            rotation_matrix[1, 0] ** 2
        )

        singular = (
            sy < 1e-6
        )

        if not singular:

            yaw = np.degrees(
                np.arctan2(
                    -rotation_matrix[2, 0],
                    sy
                )
            )

            pitch = np.degrees(
                np.arctan2(
                    rotation_matrix[2, 1],
                    rotation_matrix[2, 2]
                )
            )

            roll = np.degrees(
                np.arctan2(
                    rotation_matrix[1, 0],
                    rotation_matrix[0, 0]
                )
            )

        else:

            yaw = np.degrees(
                np.arctan2(
                    -rotation_matrix[2, 0],
                    sy
                )
            )

            pitch = np.degrees(
                np.arctan2(
                    -rotation_matrix[1, 2],
                    rotation_matrix[1, 1]
                )
            )

            roll = 0.0

        yaw = float(yaw)
        pitch = float(pitch)
        roll = float(roll)

        # ====================================================
        # SMOOTHING
        # ====================================================

        if track_id is not None:

            previous = self.smooth.get(
                track_id
            )

            alpha = float(
                HEAD_YAW_SMOOTH_ALPHA
            )

            if previous is None:

                smooth_yaw = yaw
                smooth_pitch = pitch
                smooth_roll = roll

            else:

                smooth_yaw = (
                    alpha * yaw
                    +
                    (1.0 - alpha)
                    * previous["yaw"]
                )

                smooth_pitch = (
                    alpha * pitch
                    +
                    (1.0 - alpha)
                    * previous["pitch"]
                )

                smooth_roll = (
                    alpha * roll
                    +
                    (1.0 - alpha)
                    * previous["roll"]
                )

            self.smooth[
                track_id
            ] = {
                "yaw": smooth_yaw,
                "pitch": smooth_pitch,
                "roll": smooth_roll
            }

            yaw = float(smooth_yaw)
            pitch = float(smooth_pitch)
            roll = float(smooth_roll)

        # ====================================================
        # DEBUG
        # ====================================================

        print(
            f"[POSE] "
            f"ID={track_id} "
            f"Yaw={yaw:.1f} "
            f"Pitch={pitch:.1f} "
            f"Roll={roll:.1f}"
        )

        # ====================================================
        # IMPORTANT
        #
        # CHỈ trả về số Python.
        #
        # Không trả:
        # rotation_vector
        # translation_vector
        # camera_matrix
        # distortion
        #
        # Vì các biến đó là numpy.ndarray và Flask
        # không thể jsonify trực tiếp.
        # ====================================================

        return {
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll)
        }

    # ========================================================
    # DIRECTION
    # ========================================================

    def direction(
        self,
        yaw,
        pitch,
        track_id=None
    ):

        if yaw is None:

            return "CENTER"

        yaw = float(yaw)

        # ====================================================
        # PREVIOUS
        # ====================================================

        previous = "CENTER"

        if track_id is not None:

            previous = (
                self.direction_state.get(
                    track_id,
                    "CENTER"
                )
            )

        # ====================================================
        # RAW DIRECTION
        # ====================================================

        if yaw <= -HEAD_YAW_THRESHOLD:

            candidate = "LEFT"

        elif yaw >= HEAD_YAW_THRESHOLD:

            candidate = "RIGHT"

        else:

            candidate = "CENTER"

        # ====================================================
        # HYSTERESIS
        # ====================================================

        if previous == "LEFT":

            if yaw < -HEAD_YAW_CENTER_THRESHOLD:

                candidate = "LEFT"

            else:

                candidate = "CENTER"

        elif previous == "RIGHT":

            if yaw > HEAD_YAW_CENTER_THRESHOLD:

                candidate = "RIGHT"

            else:

                candidate = "CENTER"

        # ====================================================
        # KHÔNG CÓ TRACK
        # ====================================================

        if track_id is None:

            return candidate

        # ====================================================
        # GIỮ NGUYÊN TRẠNG THÁI
        # ====================================================

        if candidate == previous:

            self.pending_direction[
                track_id
            ] = candidate

            self.pending_count[
                track_id
            ] = 0

            return previous

        # ====================================================
        # CENTER
        # ====================================================

        if candidate == "CENTER":

            self.pending_direction[
                track_id
            ] = "CENTER"

            self.pending_count[
                track_id
            ] = 0

            self.direction_state[
                track_id
            ] = "CENTER"

            return "CENTER"

        # ====================================================
        # LEFT / RIGHT
        # ====================================================

        pending = (
            self.pending_direction.get(
                track_id
            )
        )

        if pending != candidate:

            self.pending_direction[
                track_id
            ] = candidate

            self.pending_count[
                track_id
            ] = 1

        else:

            self.pending_count[
                track_id
            ] = (
                self.pending_count.get(
                    track_id,
                    0
                )
                + 1
            )

        # ====================================================
        # CONFIRM
        # ====================================================

        if (
            self.pending_count[
                track_id
            ]
            >= HEAD_DIRECTION_STABLE_FRAMES
        ):

            self.direction_state[
                track_id
            ] = candidate

            self.pending_count[
                track_id
            ] = 0

            return candidate

        return previous

    # ========================================================
    # TEMPORAL
    # ========================================================

    def temporal_update(
        self,
        track_id,
        direction
    ):

        now = time.time()

        state = (
            self.temporal.setdefault(
                track_id,
                {
                    "direction": "CENTER",
                    "start": now
                }
            )
        )

        if direction != state["direction"]:

            state["direction"] = direction
            state["start"] = now

        if direction == "CENTER":

            duration = 0.0

        else:

            duration = (
                now
                -
                state["start"]
            )

        long_event = (
            duration >= LOOKING_AWAY_SECONDS
        )

        return (
            float(duration),
            bool(long_event)
        )

    # ========================================================
    # REMOVE LOST
    # ========================================================

    def remove_lost(
        self,
        active_ids
    ):

        active_ids = set(
            active_ids
        )

        stores = [
            self.temporal,
            self.smooth,
            self.direction_state,
            self.pending_direction,
            self.pending_count
        ]

        for store in stores:

            for track_id in list(
                store.keys()
            ):

                if track_id not in active_ids:

                    del store[
                        track_id
                    ]


# ============================================================
# MONITOR
# ============================================================

class Monitor:

    def __init__(
        self,
        db
    ):

        self.db = db

        # ====================================================
        # YOLO
        # ====================================================

        print("Loading YOLO model...")

        self.model = YOLO(
            str(MODEL_PATH)
        )

        print("YOLO model loaded.")

        # ====================================================
        # FACE
        # ====================================================

        self.faces = Faces()

        self.pose = Pose()

        # ====================================================
        # CACHE
        # ====================================================

        self.cache = {}

        self.lastrec = {}

        # ====================================================
        # DATA
        # ====================================================

        self.events = []

        self.objects = []

        self.last_tracks = []

        # ====================================================
        # STATS
        # ====================================================

        self.persons = 0

        self.recognized = 0

        self.fps = 0.0

        self.ai_fps = 0.0

        self.prev = time.perf_counter()

        # ====================================================
        # EVENT COOLDOWN
        # ====================================================

        self._event_times = {}

        # ====================================================
        # FRAME COUNT
        # ====================================================

        self.frame_count = 0

        # ====================================================
        # STUDENT CACHE
        # ====================================================

        self.students_cache = []

        self.students_cache_time = 0.0

    # ========================================================
    # STUDENTS
    # ========================================================

    def get_students(self):

        now = time.time()

        if (
            now
            -
            self.students_cache_time
            >
            2.0
        ):

            self.students_cache = (
                self.db.all()
            )

            self.students_cache_time = now

        return self.students_cache

    # ========================================================
    # PROCESS
    # ========================================================

    def process(
        self,
        frame
    ):

        if frame is None:

            return self.last_tracks

        # ====================================================
        # CAMERA FPS
        # ====================================================

        now = time.perf_counter()

        dt = max(
            now - self.prev,
            1e-6
        )

        camera_fps = 1.0 / dt

        if self.fps == 0:

            self.fps = camera_fps

        else:

            self.fps = (
                self.fps * 0.9
                +
                camera_fps * 0.1
            )

        self.prev = now

        # ====================================================
        # FRAME SKIP
        # ====================================================

        self.frame_count += 1

        if (
            self.frame_count
            %
            PROCESS_EVERY_N_FRAMES
            != 1
        ):

            return self.last_tracks

        # ====================================================
        # AI TIMER
        # ====================================================

        ai_start = time.perf_counter()

        # ====================================================
        # STUDENTS
        # ====================================================

        students = self.get_students()

        # ====================================================
        # YOLO
        # ====================================================

        try:

            results = self.model.track(

                frame,

                persist=True,

                tracker="bytetrack.yaml",

                classes=DETECTION_CLASSES,

                conf=CONFIDENCE,

                iou=IOU,

                imgsz=IMG_SIZE,

                device=DEVICE,

                half=HALF,

                verbose=False
            )

        except Exception as e:

            print(
                f"[YOLO ERROR] {e}"
            )

            return self.last_tracks

        if not results:

            return self.last_tracks

        result = results[0]

        # ====================================================
        # NO BOX
        # ====================================================

        if result.boxes is None:

            self.last_tracks = []

            self.objects = []

            self.persons = 0

            self.recognized = 0

            return self.last_tracks

        boxes = result.boxes

        # ====================================================
        # EXTRACT
        # ====================================================

        xyxy = (
            boxes.xyxy
            .cpu()
            .numpy()
        )

        confs = (
            boxes.conf
            .cpu()
            .numpy()
        )

        classes = (
            boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        # ====================================================
        # TRACK IDS
        # ====================================================

        if boxes.id is not None:

            track_ids = (
                boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

        else:

            track_ids = (
                [-1] * len(xyxy)
            )

        # ====================================================
        # INITIALIZE
        # ====================================================

        tracks = []

        objects = []

        active_ids = []

        self.persons = 0

        self.recognized = 0

        height, width = frame.shape[:2]

        # ====================================================
        # PERSON
        # ====================================================

        for (
            box,
            confidence,
            class_id,
            track_id
        ) in zip(
            xyxy,
            confs,
            classes,
            track_ids
        ):

            if class_id != 0:
                continue

            self.persons += 1

            track_id = int(track_id)

            if track_id >= 0:

                active_ids.append(
                    track_id
                )

            # ------------------------------------------------
            # BBOX
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box
            )

            x1 = max(
                0,
                min(
                    x1,
                    width - 1
                )
            )

            x2 = max(
                0,
                min(
                    x2,
                    width
                )
            )

            y1 = max(
                0,
                min(
                    y1,
                    height - 1
                )
            )

            y2 = max(
                0,
                min(
                    y2,
                    height
                )
            )

            if (
                x2 <= x1
                or
                y2 <= y1
            ):
                continue

            bbox = [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ]

            crop = frame[
                y1:y2,
                x1:x2
            ]

            # ------------------------------------------------
            # DEFAULT
            # ------------------------------------------------

            item = {

                "track_id":
                    int(track_id),

                "bbox":
                    bbox,

                "student_id":
                    None,

                "name":
                    "Chưa nhận diện",

                "seat":
                    None,

                "similarity":
                    0.0,

                "yaw":
                    None,

                "pitch":
                    None,

                "roll":
                    None,

                "direction":
                    "CENTER",

                "duration":
                    0.0,

                "long_event":
                    False
            }

            # =================================================
            # FACE RECOGNITION CACHE
            # =================================================

            timestamp = time.time()

            need_recognition = (

                track_id not in self.cache

                or

                timestamp
                -
                self.lastrec.get(
                    track_id,
                    0.0
                )
                >=
                FACE_RECOGNITION_INTERVAL
            )

            if need_recognition:

                recognition = (
                    self.faces.recognize(
                        crop,
                        students
                    )
                )

                self.cache[
                    track_id
                ] = recognition

                self.lastrec[
                    track_id
                ] = timestamp

            else:

                recognition = (
                    self.cache.get(
                        track_id
                    )
                )

            # =================================================
            # RECOGNITION
            # =================================================

            if recognition is not None:

                item[
                    "similarity"
                ] = float(
                    recognition.get(
                        "similarity",
                        0.0
                    )
                )

                student = (
                    recognition.get(
                        "student"
                    )
                )

                if student is not None:

                    item.update({

                        "student_id":
                            student.get(
                                "student_id"
                            ),

                        "name":
                            student.get(
                                "name",
                                "Unknown"
                            ),

                        "seat":
                            student.get(
                                "seat"
                            )
                    })

                    self.recognized += 1

                # =================================================
                # HEAD POSE
                # =================================================

                face = (
                    recognition.get(
                        "face"
                    )
                )

                if face is not None:

                    pose = (
                        self.pose.estimate(
                            face,
                            crop.shape[1],
                            crop.shape[0],
                            track_id
                        )
                    )

                    if pose is not None:

                        # pose chỉ chứa float Python
                        item.update(
                            pose
                        )

                        direction = (
                            self.pose.direction(
                                pose["yaw"],
                                pose["pitch"],
                                track_id
                            )
                        )

                        (
                            duration,
                            long_event
                        ) = (
                            self.pose.temporal_update(
                                track_id,
                                direction
                            )
                        )

                        item.update({

                            "direction":
                                str(direction),

                            "duration":
                                float(duration),

                            "long_event":
                                bool(long_event)
                        })

                        # =================================================
                        # LONG LOOKING EVENT
                        # =================================================

                        if (
                            long_event
                            and
                            direction != "CENTER"
                        ):

                            self._event(

                                f"LOOKING_"
                                f"{direction}_LONG",

                                item,

                                duration=round(
                                    duration,
                                    2
                                )
                            )

            tracks.append(
                item
            )

        # ====================================================
        # REMOVE LOST TRACKS
        # ====================================================

        self.pose.remove_lost(
            active_ids
        )

        # ====================================================
        # OBJECTS
        # ====================================================

        for (
            box,
            confidence,
            class_id
        ) in zip(
            xyxy,
            confs,
            classes
        ):

            class_id = int(class_id)

            if class_id not in (
                63,
                67,
                73
            ):
                continue

            x1, y1, x2, y2 = map(
                int,
                box
            )

            x1 = max(
                0,
                min(
                    x1,
                    width - 1
                )
            )

            x2 = max(
                0,
                min(
                    x2,
                    width
                )
            )

            y1 = max(
                0,
                min(
                    y1,
                    height - 1
                )
            )

            y2 = max(
                0,
                min(
                    y2,
                    height
                )
            )

            if (
                x2 <= x1
                or
                y2 <= y1
            ):
                continue

            bbox = [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ]

            # ------------------------------------------------
            # TÌM PERSON GẦN NHẤT
            # ------------------------------------------------

            nearby_person = next(
                (
                    person
                    for person in tracks
                    if self.near(
                        person["bbox"],
                        bbox
                    )
                ),
                None
            )

            class_name = (
                result.names.get(
                    class_id,
                    str(class_id)
                )
            )

            obj = {

                "class_id":
                    int(class_id),

                "class_name":
                    class_name,

                "confidence":
                    float(confidence),

                "bbox":
                    bbox,


                "student_id":
                    (
                        nearby_person["student_id"]
                        if nearby_person
                        else None
                    ),

                "name":
                    (
                        nearby_person["name"]
                        if nearby_person
                        else "Chưa nhận diện"
                    ),

                "seat":
                    (
                        nearby_person["seat"]
                        if nearby_person
                        else None
                    )
            }

            objects.append(
                obj
            )

            # =================================================
            # EVENTS
            # =================================================

            if class_id == 67:

                self._event(
                    "PHONE_DETECTED",
                    obj
                )

            elif class_id == 73:

                self._event(
                    "BOOK_DETECTED",
                    obj
                )

            elif class_id == 63:

                self._event(
                    "LAPTOP_DETECTED",
                    obj
                )

        # ====================================================
        # SAVE OBJECTS
        # ====================================================

        self.objects = objects

        # ====================================================
        # SAVE TRACKS
        # ====================================================

        self.last_tracks = tracks

        # ====================================================
        # AI FPS
        # ====================================================

        ai_time = (
            time.perf_counter()
            -
            ai_start
        )

        if ai_time > 0:

            current_ai_fps = (
                1.0 / ai_time
            )

            if self.ai_fps == 0:

                self.ai_fps = (
                    current_ai_fps
                )

            else:

                self.ai_fps = (
                    self.ai_fps * 0.8
                    +
                    current_ai_fps * 0.2
                )

        return self.last_tracks

    # ========================================================
    # EVENT
    # ========================================================

    def _event(
        self,
        event_type,
        obj,
        **extra
    ):

        now = time.time()

        key = (
            event_type,
            obj.get("track_id")
        )

        last = (
            self._event_times.get(
                key,
                0.0
            )
        )

        if (
            now - last
            <
            3.0
        ):

            return

        self._event_times[
            key
        ] = now

        event = {

            "type":
                str(event_type),

            "time":
                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "student_id":
                obj.get(
                    "student_id"
                ),

            "name":
                obj.get(
                    "name"
                ),

            "seat":
                obj.get(
                    "seat"
                ),

            **extra
        }

        self.events.insert(
            0,
            event
        )

        self.events = (
            self.events[:200]
        )

    # ========================================================
    # PERSON / OBJECT RELATION
    # ========================================================

    def near(
        self,
        person_bbox,
        object_bbox
    ):

        person_cx = (
            person_bbox[0]
            +
            person_bbox[2]
        ) / 2.0

        person_cy = (
            person_bbox[1]
            +
            person_bbox[3]
        ) / 2.0

        object_cx = (
            object_bbox[0]
            +
            object_bbox[2]
        ) / 2.0

        object_cy = (
            object_bbox[1]
            +
            object_bbox[3]
        ) / 2.0

        person_width = (
            person_bbox[2]
            -
            person_bbox[0]
        )

        person_height = (
            person_bbox[3]
            -
            person_bbox[1]
        )

        return (

            abs(
                person_cx
                -
                object_cx
            )
            <
            person_width * 0.9

            and

            abs(
                person_cy
                -
                object_cy
            )
            <
            person_height * 1.2
        )

    # ========================================================
    # DRAW
    # ========================================================

    def draw(
        self,
        frame,
        tracks
    ):

        # ====================================================
        # PERSON
        # ====================================================

        for person in tracks:

            x1, y1, x2, y2 = (
                person["bbox"]
            )

            # ------------------------------------------------
            # PERSON BOX
            # ------------------------------------------------

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 220, 0),

                2
            )

            # ------------------------------------------------
            # NAME
            # ------------------------------------------------

            cv2.putText(
                frame,
                (
                    f"{person['student_id'] or '-'} | "
                    f"{person['name']}"
                ),
                (
                    x1,
                    max(
                        20,
                        y1 - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 0),
                2
            )


        # ====================================================
        # OBJECTS
        # ====================================================

        for obj in self.objects:

            x1, y1, x2, y2 = (
                obj["bbox"]
            )

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 80, 255),

                2
            )

            cv2.putText(

                frame,

                (
                    f"{obj['class_name']} "
                    f"{obj['confidence']:.2f}"
                ),

                (
                    x1,
                    max(
                        20,
                        y1 - 4
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (0, 80, 255),

                2
            )

        # ====================================================
        # FPS
        # ====================================================

        cv2.putText(

            frame,

            f"Camera FPS: {self.fps:.1f}",

            (10, 25),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (0, 255, 255),

            2
        )

        cv2.putText(

            frame,

            f"AI FPS: {self.ai_fps:.1f}",

            (10, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (0, 255, 255),

            2
        )

        cv2.putText(

            frame,

            (
                f"AI: 1/"
                f"{PROCESS_EVERY_N_FRAMES}"
                f" frames"
            ),

            (10, 75),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            1
        )

        return frame