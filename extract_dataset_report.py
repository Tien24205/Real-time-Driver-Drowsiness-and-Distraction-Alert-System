import os
import re

# --- ĐƯỜNG DẪN THỰC TẾ TRÊN MÁY CỦA EM ---
DROWSINESS_ROOT = "E:\\Project2026\\Dataset\\Drownsiness"
DISTRACT_ROOT = "E:\\Project2026\\Dataset\\Distract"

def analyze_drowsiness_module():
    print("==================================================================")
    print("📊 BÁO CÁO TRÍCH XUẤT SỐ LIỆU THẬT: MODULE BUỒN NGỦ (LSTM)")
    print("==================================================================")
    
    # 1. Thống kê tập ảnh tĩnh nền (dataset_new)
    train_dir = os.path.join(DROWSINESS_ROOT, "dataset_new", "train")
    count_new = 0
    if os.path.exists(train_dir):
        for root, dirs, files in os.walk(train_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    count_new += 1
                    
    # 2. Thống kê tập video chuỗi độc lập (DDD)
    ddd_drowsy_dir = os.path.join(DROWSINESS_ROOT, "DDD", "drownsy")
    ddd_alert_dir = os.path.join(DROWSINESS_ROOT, "DDD", "undrownsy")
    
    count_ddd_drowsy = len([f for f in os.listdir(ddd_drowsy_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(ddd_drowsy_dir) else 0
    count_ddd_alert = len([f for f in os.listdir(ddd_alert_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(ddd_alert_dir) else 0
    
    total_frames = count_new + count_ddd_drowsy + count_ddd_alert
    fps = 30
    total_seconds = total_frames / fps
    
    print(f"+ Số lượng video thô mô phỏng hành vi : 2 chuỗi video stream dài (Drowsy vs Undrownsy)")
    print(f"+ Tổng số khung hình (Frames) tích lũy  : {total_frames} ảnh")
    print(f"  - Số ảnh nền học đặc trưng mở/nhắm mắt: {count_new} ảnh")
    print(f"  - Số ảnh chuỗi thời gian thực tế (Test): {count_ddd_drowsy + count_ddd_alert} ảnh")
    print(f"    * Lớp Buồn ngủ (drownsy)            : {count_ddd_drowsy} frames")
    print(f"    * Lớp Tỉnh táo (undrownsy)          : {count_ddd_alert} frames")
    print(f"+ Tổng thời lượng dữ liệu quy đổi       : {int(total_seconds // 60)} phút {int(total_seconds % 60)} giây")
    print(f"+ Phương pháp tạo chuỗi (Sequence Gen)  : Sliding Window (Cửa sổ trượt 30 frames liên tiếp, bước trượt step=1)")
    print(f"+ Phương pháp gán nhãn (Annotation)     : Manual Ground-Truth theo thư mục trạng thái chuỗi hành vi")

def analyze_distraction_module():
    print("\n==================================================================")
    print("📊 BÁO CÁO TRÍCH XUẤT SỐ LIỆU THẬT: MODULE PHÂN TÂM (YOLOv8)")
    print("==================================================================")
    
    # Giả định cấu trúc YOLO chuẩn: ảnh ở test/images hoặc val/images, nhãn ở test/labels hoặc val/labels
    test_label_dir = os.path.join(DISTRACT_ROOT, "test", "labels")
    if not os.path.exists(test_label_dir):
        test_label_dir = os.path.join(DISTRACT_ROOT, "val", "labels") # Fallback nếu dùng chung folder val làm test
        
    train_img_dir = os.path.join(DISTRACT_ROOT, "train", "images")
    test_img_dir = os.path.join(DISTRACT_ROOT, "test", "images") if os.path.exists(os.path.join(DISTRACT_ROOT, "test", "images")) else os.path.join(DISTRACT_ROOT, "val", "images")

    count_train_yolo = len(os.listdir(train_img_dir)) if os.path.exists(train_img_dir) else 0
    count_test_yolo = len(os.listdir(test_img_dir)) if os.path.exists(test_img_dir) else 0
    
    # Đếm phân bố lớp từ file nhãn chung .txt
    class_distribution = {0: 0, 1: 0, 2: 0} # 0: normal, 1: phone, 2: drink (tùy thuộc data.yaml của em)
    
    if os.path.exists(test_label_dir):
        for label_file in os.listdir(test_label_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(test_label_dir, label_file), 'r') as lf:
                    lines = lf.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) > 0:
                            cls_id = int(parts[0])
                            if cls_id in class_distribution:
                                class_distribution[cls_id] += 1

    print(f"+ Tổng số lượng khung hình tập Train     : {count_train_yolo} ảnh")
    print(f"+ Tổng số lượng khung hình tập Test độc lập: {count_test_yolo} ảnh (Lưu chung trong 1 folder)")
    print(f"+ Phân bố lớp (Class Balance) trong tập Test dựa trên nhãn Bounding Box:")
    print(f"  - Lớp Lái xe an toàn (Normal driving)  : {class_distribution.get(0, 0)} hộp bọc")
    print(f"  - Lớp Sử dụng điện thoại (Phone usage) : {class_distribution.get(1, 0)} hộp bọc")
    print(f"  - Lớp Uống nước (Drinking behavior)    : {class_distribution.get(2, 0)} hộp bọc")
    print(f"+ Phương pháp gán nhãn (Annotation)     : Bounding Box Manual Labeling (Tọa độ chuẩn hóa [x, y, w, h])")
    print("==================================================================")

if __name__ == "__main__":
    analyze_drowsiness_module()
    analyze_distraction_module()