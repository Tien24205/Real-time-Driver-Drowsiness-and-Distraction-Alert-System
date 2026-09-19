import os
import cv2
import torch
import sqlite3
import winsound
import threading
import datetime
import collections
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Import các hàm trích xuất từ module của em
from head_pose_extractor import calculate_ear, calculate_mar, get_head_pose
from train_lstm import MultiTaskLSTM

# --- CẤU HÌNH ĐƯỜNG DẪN (RELATIVE PATHS) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LSTM_WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "best_multitask_lstm.pth")
# Mặc định dùng yolov8n.pt (pretrained COCO) để nhận diện được bottle (chai nước) và cell phone (điện thoại)
YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
# Nếu muốn dùng mô hình custom chuyên dụng đã huấn luyện (có các lớp: Open Eye, Closed Eye, Cigarette, Phone, Seatbelt), hãy bỏ comment dòng dưới đây:
# YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, "runs", "detect", "runs", "train", "yolo_distraction-2", "weights", "best.pt")
DB_PATH = os.path.join(os.path.dirname(BASE_DIR), "driver_safety.db")
DB_BACKUP_DIR = os.path.join(os.path.dirname(DB_PATH), "driver_safety_backups")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- KHỞI TẠO MEDIAPIPE FACE MESH ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH     = [78, 81, 13, 308, 14, 312]

# =========================================================
# MODULE HỖ TRỢ: SQLITE LOGGING
# =========================================================
def init_db(db_path: str):
    """Tạo bảng safety_events nếu chưa tồn tại."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS safety_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT    NOT NULL,
            event_type TEXT    NOT NULL,
            severity   TEXT    NOT NULL,
            detail     TEXT
        )
    """)
    conn.commit()
    conn.close()


def backup_db(db_path: str, backup_dir: str):
    """Sao lưu database cũ vào thư mục backup với timestamp."""
    if not os.path.exists(db_path):
        return

    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"driver_safety_{timestamp}.db")
    try:
        with sqlite3.connect(db_path) as src_conn, sqlite3.connect(backup_path) as dst_conn:
            src_conn.backup(dst_conn)
        print(f"✅ Đã sao lưu database cũ sang: {backup_path}")
    except Exception as e:
        print(f"[DB] Lỗi backup: {e}")


def reset_db(db_path: str):
    """Xóa database hiện tại để bắt đầu lại từ một DB trống."""
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"✅ Đã xóa database cũ: {db_path}")
        except Exception as e:
            print(f"[DB] Lỗi xóa database: {e}")


def log_event(db_path: str, event_type: str, severity: str, detail: str = ""):
    """Ghi một sự kiện vi phạm vào database (chạy trên thread riêng để không block camera)."""
    def _write():
        try:
            conn = sqlite3.connect(db_path)
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                "INSERT INTO safety_events (timestamp, event_type, severity, detail) VALUES (?,?,?,?)",
                (ts, event_type, severity, detail)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DB] Lỗi ghi log: {e}")
    threading.Thread(target=_write, daemon=True).start()

# =========================================================
# MODULE HỖ TRỢ: AUDIO ALERT
# =========================================================
_beep_lock = threading.Lock()

def play_alert(alert_type: str = "warning"):
    """
    Phát âm thanh cảnh báo trên thread riêng (không block vòng lặp camera).
    alert_type:
        'warning'  → 1 tiếng bíp ngắn  (Phân tâm)
        'danger'   → 3 tiếng bíp liên tiếp (Buồn ngủ)
    """
    def _beep():
        if not _beep_lock.acquire(blocking=False):
            return  # Đang có âm thanh → bỏ qua, tránh chồng chéo
        try:
            if alert_type == "danger":
                for _ in range(3):
                    winsound.Beep(1000, 300)
                    winsound.PlaySound(None, winsound.SND_PURGE)
            else:
                winsound.Beep(880, 400)
        finally:
            _beep_lock.release()
    threading.Thread(target=_beep, daemon=True).start()

# =========================================================
# MODULE HỖ TRỢ: MÔI TRƯỜNG ÁNH SÁNG
# =========================================================
def is_dark_environment(frame, brightness_threshold=55):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < brightness_threshold, avg_brightness

# =========================================================
# MAIN REALTIME SYSTEM
# =========================================================
def main_realtime_system():
    # Kiểm tra file trọng số
    if not os.path.exists(LSTM_WEIGHTS_PATH) or not os.path.exists(YOLO_WEIGHTS_PATH):
        print("❌ Lỗi: Không tìm thấy file trọng số!")
        print(f"   LSTM: {LSTM_WEIGHTS_PATH}")
        print(f"   YOLO: {YOLO_WEIGHTS_PATH}")
        return

    # Backup và reset database mỗi lần khởi động
    backup_db(DB_PATH, DB_BACKUP_DIR)
    reset_db(DB_PATH)
    init_db(DB_PATH)
    print(f"✅ Database hiện tại đã được reset và khởi tạo lại: {DB_PATH}")
    print(f"✅ Backup cũ lưu tại: {DB_BACKUP_DIR}")

    # Nạp mô hình
    lstm_model = MultiTaskLSTM().to(DEVICE)
    lstm_model.load_state_dict(torch.load(LSTM_WEIGHTS_PATH, map_location=DEVICE, weights_only=False))
    lstm_model.eval()
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)

    sequence_buffer = collections.deque(maxlen=30)

    # --- PERCLOS: cửa sổ trượt 1 phút (giả sử ~20 FPS → 1200 frame) ---
    PERCLOS_WINDOW = 1200
    perclos_buffer = collections.deque(maxlen=PERCLOS_WINDOW)

    # Mảng phục vụ bộ lọc hiệu chuẩn thích ứng
    calibration_ear_values  = []
    calibration_pitch_values = []
    calibration_yaw_values  = []

    ear_baseline  = 0.28
    pitch_baseline = 0.0
    yaw_baseline  = 0.0

    pitch_std_threshold = 15.0
    yaw_std_threshold   = 18.0
    dynamic_threshold   = 0.20
    is_calibrated       = False

    # Bộ đếm cảnh báo song song độc lập
    drowsy_counter      = 0
    distraction_counter = 0
    DEBOUNCE_LIMIT      = 15
    YAWN_THRESHOLD      = 0.58

    # Bộ đếm chống spam log & audio (ghi log tối đa 1 lần / 5 giây)
    last_log_time = {"drowsy": 0.0, "distraction": 0.0}
    LOG_COOLDOWN  = 5.0   # giây

    cap = cv2.VideoCapture(0)
    window_name = "Driver Monitoring System HUD v2"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n🚀 HỆ THỐNG GIÁM SÁT TÀI XẾ V2 ĐÃ SẴN SÀNG!")
    print("   Nhấn 'q' để thoát.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now        = datetime.datetime.now().timestamp()
        h, w, _   = frame.shape
        display_frame = frame.copy()
        is_dark, brightness = is_dark_environment(frame)

        # =============================================================
        # MODULE 1: LUỒNG NHẬN DIỆN HÀNH VI CHỦ ĐỘNG (YOLOv8)
        # =============================================================
        yolo_results = yolo_model(frame, verbose=False)[0]
        yolo_detected = False
        yolo_label    = "Normal"

        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = yolo_model.names[cls_id]

            # Kiểm tra nếu nhãn thuộc nhóm thiết bị (phone/cell phone) hoặc đồ uống (drink/bottle)
            if conf > 0.45 and label in ['phone', 'drink', 'bottle', 'cell phone']:
                yolo_detected = True
                
                # Chuẩn hóa nhãn hiển thị sang PHONE hoặc BOTTLE
                if label in ['drink', 'bottle']:
                    display_label = 'BOTTLE'
                else:
                    display_label = 'PHONE'
                
                yolo_label    = display_label

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(display_frame, f"{display_label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # =============================================================
        # MODULE 2: LUỒNG SỨC KHỎE & TƯ THẾ HÌNH HỌC (MediaPipe + LSTM)
        # =============================================================
        rgb_frame   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = face_mesh.process(rgb_frame)

        drowsiness_prediction = False
        head_pose_distraction = False
        yawn_detected         = False
        distraction_label     = "Normal"

        pitch, yaw, roll = 0.0, 0.0, 0.0
        avg_ear = 0.25
        mar     = 0.15

        if mesh_results.multi_face_landmarks:
            face_landmarks = mesh_results.multi_face_landmarks[0]

            p_left_outer  = np.array([face_landmarks.landmark[33].x,  face_landmarks.landmark[33].y])
            p_right_outer = np.array([face_landmarks.landmark[263].x, face_landmarks.landmark[263].y])
            face_width    = np.linalg.norm(p_left_outer - p_right_outer)

            avg_ear = (calculate_ear(face_landmarks, LEFT_EYE,  face_width) +
                       calculate_ear(face_landmarks, RIGHT_EYE, face_width)) / 2.0
            mar     = calculate_mar(face_landmarks, MOUTH)
            pitch, yaw, roll, _ = get_head_pose(frame, face_landmarks)

            # Loại bỏ giá trị rác khi thuật toán bị lỗi gập góc
            if abs(pitch) > 90.0 or abs(yaw) > 90.0:
                pitch, yaw = 0.0, 0.0

            # --- PHA HIỆU CHUẨN ĐỘNG (100 frame đầu) ---
            if not is_calibrated:
                calibration_ear_values.append(avg_ear)
                calibration_pitch_values.append(pitch)
                calibration_yaw_values.append(yaw)

                cv2.putText(display_frame,
                            f"CALIBRATING SYSTEM: {len(calibration_ear_values)}/100",
                            (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if len(calibration_ear_values) >= 100:
                    ear_baseline   = np.median(calibration_ear_values)
                    pitch_baseline = np.median(calibration_pitch_values)
                    yaw_baseline   = np.median(calibration_yaw_values)
                    pitch_std_threshold = max(15.0, 3.5 * np.std(calibration_pitch_values))
                    yaw_std_threshold   = max(18.0, 3.5 * np.std(calibration_yaw_values))
                    is_calibrated = True
                    print(f"✅ Hiệu chuẩn xong! EAR baseline={ear_baseline:.3f} | "
                          f"Pitch limit=±{pitch_std_threshold:.1f}° | Yaw limit=±{yaw_std_threshold:.1f}°")

                sequence_buffer.append([0.31, mar, 0.0, 0.0, 0.0])
                perclos_buffer.append(0)  # Chưa hiệu chuẩn → coi là mắt mở
            else:
                pitch_deviation = pitch - pitch_baseline
                yaw_deviation   = yaw  - yaw_baseline

                # --- PERCLOS: cập nhật cửa sổ trượt ---
                eye_closed_flag = 1 if avg_ear < dynamic_threshold else 0
                perclos_buffer.append(eye_closed_flag)
                perclos = (sum(perclos_buffer) / len(perclos_buffer)) * 100.0  # %

                # Phân tâm hình học (head pose)
                if not yolo_detected:
                    if abs(pitch_deviation) > pitch_std_threshold or abs(yaw_deviation) > yaw_std_threshold:
                        head_pose_distraction = True
                        distraction_label     = "LOOKING DOWN/AWAY"

                # Phát hiện ngáp trực tiếp từ MAR
                if mar > YAWN_THRESHOLD:
                    yawn_detected = True

                # Dự đoán LSTM
                alpha = 0.78
                if is_dark:
                    alpha -= 0.04
                dynamic_threshold        = alpha * ear_baseline
                adjusted_ear_for_lstm   = (avg_ear / (ear_baseline + 0.001)) * 0.31

                sequence_buffer.append([adjusted_ear_for_lstm, mar, pitch_deviation, yaw_deviation, roll])

                if len(sequence_buffer) == 30:
                    seq_tensor = torch.tensor([list(sequence_buffer)], dtype=torch.float32).to(DEVICE)
                    with torch.no_grad():
                        drowsy_logits, _ = lstm_model(seq_tensor)
                        _, predicted     = torch.max(drowsy_logits, 1)
                        if predicted.item() == 1 and avg_ear < dynamic_threshold:
                            drowsiness_prediction = True

                # Hiển thị PERCLOS trên HUD
                perclos_color = (0, 255, 0) if perclos < 15 else (0, 165, 255) if perclos < 30 else (0, 0, 255)
                cv2.putText(display_frame, f"PERCLOS: {perclos:.1f}%",
                            (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, perclos_color, 2)

        else:
            if not yolo_detected:
                head_pose_distraction = True
                distraction_label     = "FACE LOST"
            perclos_buffer.append(0)

        # =============================================================
        # HỢP NHẤT TRẠNG THÁI
        # =============================================================
        final_distraction_logic = yolo_detected or head_pose_distraction
        final_drowsy_logic      = drowsiness_prediction or yawn_detected

        # =============================================================
        # MODULE 3: MÁY TRẠNG THÁI (STATE MACHINE) + CẢNH BÁO HUD
        # =============================================================
        status_text    = "STATUS: DRIVING SAFE"
        status_color   = (0, 255, 0)
        draw_alert_bar = False
        alert_bar_text  = ""
        alert_bar_color = (0, 0, 0)

        # Cập nhật counter song song
        if final_distraction_logic:
            distraction_counter = min(30, distraction_counter + 1)
        else:
            distraction_counter = max(0, distraction_counter - 2)

        if final_drowsy_logic:
            drowsy_counter = min(30, drowsy_counter + 1)
        else:
            drowsy_counter = max(0, drowsy_counter - 2)

        # Logic ưu tiên hiển thị HUD
        if distraction_counter >= DEBOUNCE_LIMIT:
            active_label   = yolo_label if yolo_detected else distraction_label
            status_text    = f"STATUS: DISTRACTION ({active_label})"
            status_color   = (0, 165, 255)
            draw_alert_bar = True
            alert_bar_text  = f"!!! WARNING: DISTRACTION ({active_label}) !!!"
            alert_bar_color = (0, 69, 255)

            # --- Audio alert: 1 beep ---
            play_alert("warning")

            # --- Ghi log SQLite (chống spam: tối đa 1 lần/5s) ---
            if now - last_log_time["distraction"] > LOG_COOLDOWN:
                severity = "HIGH" if yolo_detected else "MEDIUM"
                log_event(DB_PATH, f"Distraction ({active_label})", severity, active_label)
                last_log_time["distraction"] = now

        elif drowsy_counter >= DEBOUNCE_LIMIT:
            label_drowsy   = "YAWNING DETECTED" if yawn_detected else "DROWSINESS"
            status_text    = f"STATUS: {label_drowsy}"
            status_color   = (0, 0, 255)
            draw_alert_bar = True
            alert_bar_text  = f"!!! ALERT: {label_drowsy} !!!"
            alert_bar_color = (0, 0, 255)

            # --- Audio alert: 3 beeps (nguy hiểm hơn) ---
            play_alert("danger")

            # --- Ghi log SQLite ---
            if now - last_log_time["drowsy"] > LOG_COOLDOWN:
                severity = "CRITICAL" if not yawn_detected else "HIGH"
                log_event(DB_PATH, label_drowsy, severity, f"EAR={avg_ear:.3f} MAR={mar:.3f}")
                last_log_time["drowsy"] = now

        else:
            status_text  = "STATUS: DRIVING SAFE"
            status_color = (0, 255, 0)

        # =============================================================
        # VẼ HUD
        # =============================================================
        if draw_alert_bar:
            cv2.rectangle(display_frame, (0, 0), (w, 55), alert_bar_color, -1)
            cv2.putText(display_frame, alert_bar_text, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3)
        else:
            cv2.putText(display_frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Thông số debug
        p_dev = pitch - pitch_baseline if is_calibrated else 0.0
        cv2.putText(display_frame,
                    f"Pitch Dev: {p_dev:.1f} | Limit: {pitch_std_threshold:.1f} | EAR: {avg_ear:.2f} | MAR: {mar:.2f}",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(display_frame,
                    f"Counters -> Drowsy: {drowsy_counter} | Distract: {distraction_counter} | Bright: {brightness:.0f}",
                    (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("Driver Monitoring System HUD v2", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Hệ thống đã dừng. Dữ liệu đã được lưu vào:", DB_PATH)

if __name__ == "__main__":
    main_realtime_system()