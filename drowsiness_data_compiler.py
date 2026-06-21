import os
import cv2
import numpy as np
import mediapipe as mp
# Import các hàm trích xuất từ file Tuần 1 của em
from head_pose_extractor import calculate_ear, calculate_mar, get_head_pose

# Khởi tạo MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5
)

# Chỉ số mốc chuẩn của MediaPipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH = [78, 81, 13, 308, 14, 312]

def extract_features_from_img(img_path):
    """Đọc ảnh và trích xuất vector 5 chỉ số số"""
    image = cv2.imread(img_path)
    if image is None: 
        return None
        
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        # Tính toán EAR, MAR và Head Pose
        avg_ear = (calculate_ear(face_landmarks, LEFT_EYE) + calculate_ear(face_landmarks, RIGHT_EYE)) / 2.0
        mar = calculate_mar(face_landmarks, MOUTH)
        pitch, yaw, roll, _ = get_head_pose(image, face_landmarks)
        
        return [avg_ear, mar, pitch, yaw, roll]
    return None

def process_drowsiness_dataset(base_path, seq_len=30):
    """Hàm quét tổng lực dựa trên cấu trúc chuẩn xác của em"""
    X_data = []
    y_data = [] # Nhãn buồn ngủ: 0 = Tỉnh táo, 1 = Buồn ngủ

    # 1. XỬ LÝ FOLDER: dataset_new (chứa train và test)
    dataset_new_path = os.path.join(base_path, "dataset_new")
    if os.path.exists(dataset_new_path):
        print("\n>>> Đang bóc tách dữ liệu từ thư mục 'dataset_new'...")
        for sub_folder in ['train', 'test']:
            sub_folder_path = os.path.join(dataset_new_path, sub_folder)
            if not os.path.exists(sub_folder_path): continue
            
            for category in os.listdir(sub_folder_path):
                category_path = os.path.join(sub_folder_path, category)
                if not os.path.isdir(category_path): continue
                
                # Ánh xạ nhãn trạng thái sinh lý
                if category.lower() in ['closed', 'yawn']:
                    label = 1
                elif category.lower() in ['open', 'no_yawn']:
                    label = 0
                else:
                    continue
                
                print(f"  -> Quét '{sub_folder}/{category}' -> Gán nhãn Buồn ngủ: {label}")
                for img_name in os.listdir(category_path):
                    img_file = os.path.join(category_path, img_name)
                    features = extract_features_from_img(img_file)
                    if features is not None:
                        # Augmentation lặp lại 30 lần tạo chuỗi thời gian
                        X_data.append([features] * seq_len)
                        y_data.append(label)

   # ==================================================================
    # 2. XỬ LÝ FOLDER: DDD (chứa drowsy/drownsy và undrowsy/undrownsy)
    # ==================================================================
    ddd_path = os.path.join(base_path, "DDD")
    if os.path.exists(ddd_path):
        print("\n>>> Đang bóc tách dữ liệu từ thư mục 'DDD'...")
        for category in os.listdir(ddd_path):
            category_path = os.path.join(ddd_path, category)
            if not os.path.isdir(category_path): continue
            
            # CHUYỂN SANG DÙNG "IN" ĐỂ CHỐNG SAI CHÍNH TẢ CHỮ "N"
            category_lower = category.lower()
            
            if 'undrow' in category_lower: # Khớp cho cả 'undrowsy' và 'undrownsy'
                label = 0
            elif 'drow' in category_lower:  # Khớp cho cả 'drowsy' và 'drownsy'
                label = 1
            else:
                continue
                
            print(f"  -> Khớp thư mục: 'DDD/{category}' -> Gán nhãn Buồn ngủ: {label}")
            
            # Đọc ảnh bên trong thư mục con này
            for img_name in os.listdir(category_path):
                img_file = os.path.join(category_path, img_name)
                
                # Bỏ qua các file ẩn hoặc file cấu hình hệ thống nếu có
                if img_name.startswith('.'): continue 
                
                features = extract_features_from_img(img_file)
                if features is not None:
                    X_data.append([features] * seq_len)
                    y_data.append(label)

    return np.array(X_data, dtype=np.float32), np.array(y_data, dtype=np.int64)

if __name__ == "__main__":
    # ĐƯỜNG DẪN THỰC TẾ TRÊN MÁY TÍNH CỦA EM
    # Hãy thay thế đường dẫn này trỏ thẳng vào folder "Drowsiness" mẹ
    DROWSINESS_ROOT = "E:\Project2026\Dataset\Drownsiness"

    if os.path.exists(DROWSINESS_ROOT):
        X, y = process_drowsiness_dataset(DROWSINESS_ROOT, seq_len=30)
        
        if len(X) > 0:
            # Lưu trữ dữ liệu số đã xử lý sạch sẽ vào thư mục data/processed
            os.makedirs("data/processed", exist_ok=True)
            np.save("data/processed/X_drowsy_seq.npy", X)
            np.save("data/processed/y_drowsy_labels.npy", y)
            
            print("\n=======================================================")
            print("🎉 HOÀN THÀNH ĐÓNG GÓI MODULE DỮ LIỆU BUỒN NGỦ!")
            print(f"Tổng số chuỗi mẫu (Sequences) thu thập được: {X.shape[0]}")
            print(f"Kích thước mảng X (Đặc trưng chuỗi): {X.shape} -> (Mẫu, 30 Khung hình, 5 Chỉ số)")
            print(f"Kích thước mảng y (Nhãn tương ứng): {y.shape}")
            print("=======================================================")
        else:
            print("Lỗi: Không bóc tách được đặc trưng nào từ các ảnh. Hãy kiểm tra định dạng ảnh.")
    else:
        print(f"Lỗi: Không tìm thấy thư mục gốc Drowsiness tại đường dẫn: {DROWSINESS_ROOT}")