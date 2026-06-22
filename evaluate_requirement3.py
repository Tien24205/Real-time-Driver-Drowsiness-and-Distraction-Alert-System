import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import kiến trúc mạng LSTM chính thức từ file code gốc của em
from train_lstm import MultiTaskLSTM

# --- ĐƯỜNG DẪN TỚI CÁC FILE THỰC TẾ TRONG DỰ ÁN CỦA EM ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_RESULTS_CSV = os.path.join(BASE_DIR, "results.csv")
X_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "X_drowsy_seq.npy")
Y_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "y_drowsy_labels.npy")
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_multitask_lstm.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. TRÍCH XUẤT CHỈ SỐ MẤT TẬP TRUNG (YOLOv8 DISTRACTION METRICS)
# ==============================================================================
def get_yolo_metrics():
    print("⏳ Đang bóc tách chỉ số thực nghiệm YOLOv8 từ file results.csv...")
    if not os.path.exists(YOLO_RESULTS_CSV):
        print(f"❌ Lỗi: Không tìm thấy file '{YOLO_RESULTS_CSV}' tại thư mục gốc.")
        return None

    # Đọc file CSV lịch sử log huấn luyện của YOLOv8
    df = pd.read_csv(YOLO_RESULTS_CSV)
    
    # Làm sạch tên cột (xóa khoảng trắng thừa nếu có)
    df.columns = [c.strip() for c in df.columns]
    
    # Lấy dòng dữ liệu ở Epoch cuối cùng (Dòng số 50)
    last_epoch_data = df.iloc[-1]
    
    precision = last_epoch_data['metrics/precision(B)'] * 100
    recall = last_epoch_data['metrics/recall(B)'] * 100
    map50 = last_epoch_data['metrics/mAP50(B)'] * 100
    map50_95 = last_epoch_data['metrics/mAP50-95(B)'] * 100
    
    # Tính toán chỉ số F1-Score từ Precision và Recall gốc
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "mAP50": map50,
        "mAP50_95": map50_95
    }

# ==============================================================================
# 2. ĐÁNH GIÁ CHỈ SỐ BUỒN NGỦ THỰC TẾ (LSTM DROWSINESS METRICS)
# ==============================================================================
def evaluate_lstm_metrics():
    print("⏳ Đang nạp ma trận dữ liệu và mô hình LSTM thật để tính toán hiệu năng...")
    if not os.path.exists(X_TEST_PATH) or not os.path.exists(Y_TEST_PATH):
        print(f"❌ Lỗi: Không thấy dữ liệu chuỗi tại '{X_TEST_PATH}'. Hãy chạy 'drowsiness_data_compiler.py' trước.")
        return None
    if not os.path.exists(LSTM_MODEL_PATH):
        print(f"❌ Lỗi: Không tìm thấy file trọng số '{LSTM_MODEL_PATH}'.")
        return None

    # Nạp dữ liệu mảng numpy thật của em
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    
    # Khởi tạo mạng cấu hình chuẩn 5 đặc trưng đầu vào [EAR, MAR, Pitch, Yaw, Roll]
    model = MultiTaskLSTM(input_size=5, hidden_size=64, num_layers=2).to(DEVICE)
    model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Chuyển đổi sang tensor để đẩy vào GPU/CPU chạy inference
    inputs = torch.from_numpy(X_test).float().to(DEVICE)
    with torch.no_grad():
        drowsy_out, _ = model(inputs)
        y_pred = torch.max(drowsy_out, 1)[1].cpu().numpy()
        
    # Tính toán các chỉ số phân lớp chuẩn khoa học dựa trên nhãn Ground Truth gốc
    acc = accuracy_score(y_test, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    return {
        "accuracy": acc,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100
    }

# ==============================================================================
# 3. IN KẾT QUẢ ĐẦY ĐỦ PHỤC VỤ VIẾT LUẬN VĂN (YÊU CẦU 3)
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("   BÁO CÁO KẾT QUẢ THỰC NGHIỆM CHI TIẾT THEO YÊU CẦU 3 CỦA HỘI ĐỒNG")
    print("="*70)
    
    # Chạy cấu phần YOLOv8
    yolo_res = get_yolo_metrics()
    if yolo_res:
        print("\n🎯 1. BÀI TOÁN PHÁT HIỆN MẤT TẬP TRUNG (MÔ HÌNH YOLOv8):")
        print(f"   + Precision (Độ chính xác)         : {yolo_res['precision']:.2f}%")
        print(f"   + Recall (Tỷ lệ bao phủ)           : {yolo_res['recall']:.2f}%")
        print(f"   + F1-Score (Chỉ số F1 hài hòa)     : {yolo_res['f1']:.2f}%")
        print(f"   + mAP@0.5 (Chỉ số mAP chuẩn)       : {yolo_res['mAP50']:.2f}%")
        print(f"   + mAP@0.5:0.95 (mAP nghiêm ngặt)   : {yolo_res['mAP50_95']:.2f}%")
        
    # Chạy cấu phần LSTM
    lstm_res = evaluate_lstm_metrics()
    if lstm_res:
        print("\n🎯 2. BÀI TOÁN PHÁT HIỆN BUỒN NGỦ (MÔ HÌNH LSTM PHÂN TÍCH CHUỖI):")
        print(f"   + Accuracy (Độ chính xác tổng thể) : {lstm_res['accuracy']:.2f}%")
        print(f"   + Precision (Độ chính xác lớp bệnh): {lstm_res['precision']:.2f}%")
        print(f"   + Recall (Tỷ lệ phát hiện nhầm ít) : {lstm_res['recall']:.2f}%")
        print(f"   + F1-Score (Chỉ số cân bằng F1)    : {lstm_res['f1']:.2f}%")
        
    print("\n" + "="*70)
    print("💡 HƯỚNG DẪN: Sao chép các số liệu thực tế ở trên để điền vào phần ")
    print("   đánh giá riêng biệt của hai mô hình trong chương 4/5 file LaTeX.")
    print("="*70 + "\n")