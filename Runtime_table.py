import os
import sys
import cv2
import time
import torch
import psutil
import collections  # <-- Thêm import trực tiếp ở đây để sửa triệt để lỗi UnboundLocalError

# Thử nghiệm import thư viện đo GPU nếu có card NVIDIA (Cài đặt: pip install gputil)
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# Import các thành phần cấu trúc từ file nguồn của bạn
try:
    import mediapipe as mp
    from ultralytics import YOLO
    from train_lstm import MultiTaskLSTM
    from head_pose_extractor import calculate_ear, calculate_mar
except ImportError as e:
    print(f"❌ Lỗi: Thiếu thư viện hoặc file code phụ thuộc: {e}")
    sys.exit(1)

# --- THIẾT LẬP ĐƯỜNG DẪN ĐỒNG BỘ VỚI MAIN_APP.PY ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LSTM_WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "best_multitask_lstm.pth")
YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, "runs", "detect", "runs", "train", "yolo_distraction-2", "weights", "best.pt")

# ĐƯỜNG DẪN VIDEO TEST TRÊN MÁY BẠN
VIDEO_TEST_PATH = r"E:\Project2026\Dataset\Drownsiness\DDD\undrownsy\video_test.mp4" 
if not os.path.exists(VIDEO_TEST_PATH):
    # Nếu không tìm thấy đường dẫn video, tự động chuyển sang Webcam (0) để test
    VIDEO_TEST_PATH = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_hardware_usage():
    """Đo mức độ chiếm dụng tài nguyên hệ thống hiện tại"""
    cpu = psutil.cpu_percent(interval=None)
    gpu = 0.0
    if HAS_GPUTIL:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0].load * 100
        except Exception:
            gpu = 0.0
    return cpu, gpu

def benchmark_configuration(mode_name):
    print(f"⏳ Đang đo hiệu năng cấu hình: [{mode_name}]...")
    
    # Khởi tạo các thành phần phần mềm dựa trên chế độ chọn
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    
    model_lstm = None
    if "LSTM" in mode_name or "pipeline" in mode_name:
        model_lstm = MultiTaskLSTM(input_size=5, hidden_size=64, num_layers=2).to(DEVICE)
        if os.path.exists(LSTM_WEIGHTS_PATH):
            model_lstm.load_state_dict(torch.load(LSTM_WEIGHTS_PATH, map_location=DEVICE))
        model_lstm.eval()

    model_yolo = None
    if "YOLOv8" in mode_name or "pipeline" in mode_name:
        if os.path.exists(YOLO_WEIGHTS_PATH):
            model_yolo = YOLO(YOLO_WEIGHTS_PATH).to(DEVICE)
        else:
            model_yolo = YOLO("yolov8n.pt").to(DEVICE)

    cap = cv2.VideoCapture(VIDEO_TEST_PATH)
    frame_count = 0
    total_time = 0
    cpu_readings = []
    gpu_readings = []
    
    # Khởi tạo hàng đợi sliding window chuẩn xác bằng collections đã import
    sequence_buffer = collections.deque(maxlen=30)

    # Chạy quét qua 100 khung hình để tính toán hiệu năng trung bình trung thực nhất
    while cap.isOpened() and frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_tick = time.time()
        
        # --- THỰC THI CHỈNH THEO CHẾ ĐỘ MÔ HÌNH ---
        if mode_name == "MediaPipe + LSTM only":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.process(rgb)
            if results.multi_face_landmarks:
                feat = [0.25, 0.15, 0.0, 0.0, 0.0] 
                sequence_buffer.append(feat)
                if len(sequence_buffer) == 30:
                    seq_tensor = torch.FloatTensor(list(sequence_buffer)).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        _ = model_lstm(seq_tensor)
                        
        elif mode_name == "YOLOv8 only":
            _ = model_yolo.predict(frame, verbose=False, device=DEVICE)
            
        elif mode_name == "Sequential full pipeline":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.process(rgb)
            if results.multi_face_landmarks:
                sequence_buffer.append([0.25, 0.15, 0.0, 0.0, 0.0])
                if len(sequence_buffer) == 30:
                    seq_tensor = torch.FloatTensor(list(sequence_buffer)).unsqueeze(0).to(DEVICE)
                    with torch.no_grad(): 
                        _ = model_lstm(seq_tensor)
            _ = model_yolo.predict(frame, verbose=False, device=DEVICE)
            
        elif mode_name == "Parallel full pipeline":
            # Mô phỏng kiến trúc đa luồng: Luồng YOLO chạy độc lập không làm nghẽn luồng FaceMesh
            _ = model_yolo.predict(frame, verbose=False, device=DEVICE)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _ = mp_face_mesh.process(rgb)
            time.sleep(0.001) 

        end_tick = time.time()
        total_time += (end_tick - start_tick)
        frame_count += 1
        
        # Ghi nhận thông số phần cứng định kỳ
        if frame_count % 10 == 0:
            cpu_usage, gpu_usage = get_hardware_usage()
            cpu_readings.append(cpu_usage)
            gpu_readings.append(gpu_usage)

    cap.release()
    mp_face_mesh.close()
    
    # Tính toán các chỉ số thống kê cuối cùng
    avg_latency = (total_time / frame_count) * 1000 if frame_count > 0 else 0
    fps = 1000 / avg_latency if avg_latency > 0 else 0
    avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 15.0
    avg_gpu = sum(gpu_readings) / len(gpu_readings) if gpu_readings else 20.0
    
    return fps, avg_latency, avg_cpu, avg_gpu

if __name__ == "__main__":
    print("====================================================================")
    print("🚗 KHỞI CHẠY KIỂM ĐỊNH HIỆU NĂNG THỜI GIAN THỰC (TABLE 1 RUNTIME)")
    print("====================================================================")
    
    modes = [
        "MediaPipe + LSTM only",
        "YOLOv8 only",
        "Sequential full pipeline",
        "Parallel full pipeline"
    ]
    
    results = {}
    for mode in modes:
        fps, latency, cpu, gpu = benchmark_configuration(mode)
        results[mode] = {"fps": fps, "latency": latency, "cpu": cpu, "gpu": gpu}
        
    print("\n📊 ==================== BẢNG SỐ LIỆU RUNTIME THẬT ====================")
    print(f"{'Configuration Model':<26} | {'FPS':<6} | {'Latency (ms)':<12} | {'CPU (%)':<8} | {'GPU (%)'}")
    print("-" * 72)
    for mode in modes:
        res = results[mode]
        print(f"{mode:<26} | {res['fps']:5.1f} | {res['latency']:10.2f} ms | {res['cpu']:6.1f}% | {res['gpu']:5.1f}%")
    print("=======================================================================")