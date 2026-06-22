import os
import sys
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# Import chính cấu trúc model từ file code thật của em
from train_lstm import MultiTaskLSTM

# --- ĐƯỜNG DẪN DỮ LIỆU VÀ TRỌNG SỐ THẬT ---
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
X_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "X_drowsy_seq.npy")
Y_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "y_drowsy_labels.npy")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_multitask_lstm.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_real_data():
    if not os.path.exists(X_TEST_PATH) or not os.path.exists(Y_TEST_PATH):
        print(f"❌ Lỗi: Không tìm thấy file dữ liệu tại '{X_TEST_PATH}'.")
        print("💡 Hãy chắc chắn em đã chạy file `drowsiness_data_compiler.py` trước.")
        sys.exit(1)
    return np.load(X_TEST_PATH), np.load(Y_TEST_PATH)

def calculate_far(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if (cm[0, 0] + cm[0, 1]) > 0:
        return (cm[0, 1] / (cm[0, 0] + cm[0, 1])) * 100
    return 0.0

# --- BASELINE 1: NGƯỠNG EAR TĨNH ---
def eval_static_threshold(X_seq, y_true, threshold=0.21):
    # Lấy đặc trưng EAR (vị trí index 0) tại frame cuối cùng trong chuỗi 30 frames
    ear_last_frame = X_seq[:, -1, 0]
    y_pred = np.where(ear_last_frame < threshold, 1, 0)
    return accuracy_score(y_true, y_pred) * 100, calculate_far(y_true, y_pred)

# --- BASELINE 2: EAR TĨNH + BỘ LỌC DEBOUNCE ---
def eval_static_debounce(X_seq, y_true, threshold=0.21, debounce_limit=15):
    y_pred = []
    for seq in X_seq:
        ear_history = seq[:, 0]
        consecutive_low_frames = 0
        triggered = 0
        for ear in ear_history:
            if ear < threshold:
                consecutive_low_frames += 1
            else:
                consecutive_low_frames = max(0, consecutive_low_frames - 1)
            if consecutive_low_frames >= debounce_limit:
                triggered = 1
                break
        y_pred.append(triggered)
    y_pred = np.array(y_pred)
    return accuracy_score(y_true, y_pred) * 100, calculate_far(y_true, y_pred)

# --- BASELINE 3 & HỆ THỐNG ĐẦY ĐỦ (DỰA TRÊN FILE .PTH THẬT CỦA EM) ---
def eval_lstm_variants(X_seq, y_true):
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Lỗi: Không thấy file trọng số thật tại '{MODEL_PATH}'")
        sys.exit(1)
        
    model = MultiTaskLSTM(input_size=5, hidden_size=64, num_layers=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    inputs = torch.from_numpy(X_seq).float().to(DEVICE)
    with torch.no_grad():
        drowsy_out, _ = model(inputs)
        raw_predictions = torch.max(drowsy_out, 1)[1].cpu().numpy()
        
    # Baseline 3: Mạng LSTM thô không có bộ lọc làm mượt
    acc_lstm_raw = accuracy_score(y_true, raw_predictions) * 100
    far_lstm_raw = calculate_far(y_true, raw_predictions)
    
    # Hệ thống đề xuất (Ours): LSTM kết hợp bộ đếm Debounce làm mượt (giống main_app.py)
    y_pred_ours = []
    drowsy_counter = 0
    for pred in raw_predictions:
        if pred == 1:
            drowsy_counter += 1
        else:
            drowsy_counter = max(0, drowsy_counter - 1)
        y_pred_ours.append(1 if drowsy_counter >= 10 else 0) # Ngưỡng 10 từ main_app.py
        
    y_pred_ours = np.array(y_pred_ours)
    acc_ours = accuracy_score(y_true, y_pred_ours) * 100
    far_ours = calculate_far(y_true, y_pred_ours)
    
    return acc_lstm_raw, far_lstm_raw, acc_ours, far_ours

if __name__ == "__main__":
    print("⏳ Bước 1: Đang nạp dữ liệu từ file npy thật...")
    X_test, y_test = load_real_data()
    
    print("⏳ Bước 2: Đang chạy đánh giá trên mô hình thật...")
    acc1, far1 = eval_static_threshold(X_test, y_test)
    acc2, far2 = eval_static_debounce(X_test, y_test)
    acc3, far3, acc_ours, far_ours = eval_lstm_variants(X_test, y_test)
    
    print("\n📊 ==================== BẢNG SỐ LIỆU CHÍNH XÁC ĐỂ ĐIỀN LUẬN VĂN ====================")
    print(f"{'Mô hình / Phương pháp so sánh':<45} | {'Accuracy (%)':<15} | {'False Alarm Rate (FAR %)'}")
    print("-" * 85)
    print(f"{'1. Static EAR Threshold':<45} | {acc1:<15.2f}% | {far1:.2f}%")
    print(f"{'2. Static EAR + Debounce Filter':<45} | {acc2:<15.2f}% | {far2:.2f}%")
    print(f"{'3. LSTM (Without Dynamic Calibration)':<45} | {acc3:<15.2f}% | {far3:.2f}%")
    print(f"{'4. Proposed Framework (Ours: LSTM + Debounce)':<45} | {acc_ours:<15.2f}% | {far_ours:.2f}%")
    print("==================================================================================")