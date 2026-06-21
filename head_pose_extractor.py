import cv2
import mediapipe as mp
import numpy as np
import math

# Khởi tạo MediaPipe Face Mesh với cấu hình tối ưu chi tiết (Refine Landmarks)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

def calculate_ear(landmarks, eye_indices, face_width_reference=None):
    """
    Tính EAR chuẩn hóa giúp loại bỏ hoàn toàn sự ảnh hưởng khi tài xế ngồi xa/gần camera
    """
    # Trích xuất tọa độ 3D của 6 mốc mắt
    p = np.array([[landmarks.landmark[i].x, landmarks.landmark[i].y, landmarks.landmark[i].z] for i in eye_indices])
    
    # Khoảng cách chiều dọc thô của mắt
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    # Khoảng cách chiều ngang thô của mắt
    h = np.linalg.norm(p[0] - p[3])
    
    raw_ear = (v1 + v2) / (2.0 * h)
    
    # Kỹ thuật chuẩn hóa khoảng cách (Distance Normalization):
    # face_width_reference càng nhỏ (ngồi xa) -> scale_factor càng nhỏ -> bù đắp lượng pixel bị thu hẹp
    if face_width_reference is not None:
        scale_factor = face_width_reference * 5.0 
        return raw_ear / (scale_factor + 0.1)
        
    return raw_ear

def calculate_mar(landmarks, mouth_indices):
    """
    Tính chỉ số mở miệng MAR (Mouth Aspect Ratio) để phát hiện ngáp
    """
    p = np.array([[landmarks.landmark[i].x, landmarks.landmark[i].y, landmarks.landmark[i].z] for i in mouth_indices])
    v1 = np.linalg.norm(p[1] - p[5]) # Mốc 81 và 312
    v2 = np.linalg.norm(p[2] - p[4]) # Mốc 13 và 14
    h = np.linalg.norm(p[0] - p[3])  # Mốc 78 và 308
    return (v1 + v2) / (2.0 * h)

def get_head_pose(frame, landmarks):
    """
    Ước lượng góc xoay đầu (Pitch, Yaw, Roll) tự động thích ứng theo kích thước camera thực tế
    """
    h, w, _ = frame.shape
    
    # 1. Định nghĩa các điểm mốc 3D tiêu chuẩn của khuôn mặt (Generic 3D Model Points)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Mũi (Nose tip)
        (0.0, -330.0, -65.0),        # Cằm (Chin)
        (-225.0, 170.0, -135.0),     # Khóe mắt trái ngoài
        (225.0, 170.0, -135.0),      # Khóe mắt phải ngoài
        (-150.0, -150.0, -125.0),    # Khóe miệng trái
        (150.0, -150.0, -125.0)      # Khóe miệng phải
    ], dtype=np.float32)

    # 2. Trích xuất các điểm mốc 2D tương ứng từ MediaPipe Face Mesh (Đã nhân tương thích với pixel w, h)
    image_points = np.array([
        (landmarks.landmark[1].x * w, landmarks.landmark[1].y * h),     
        (landmarks.landmark[152].x * w, landmarks.landmark[152].y * h), 
        (landmarks.landmark[33].x * w, landmarks.landmark[33].y * h),   
        (landmarks.landmark[263].x * w, landmarks.landmark[263].y * h), 
        (landmarks.landmark[61].x * w, landmarks.landmark[61].y * h),   
        (landmarks.landmark[291].x * w, landmarks.landmark[291].y * h)  
    ], dtype=np.float32)

    # 3. TỰ ĐỘNG CẬP NHẬT MA TRẬN CAMERA THEO KHUNG HÌNH THỰC TẾ
    # Tránh lỗi nhảy góc đầu ảo khi tài xế dịch chuyển vị trí xa/gần
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1)) 

    # 4. Giải bài toán PnP để tìm ma trận xoay
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0, None

    # 5. Chuyển đổi sang góc Euler (Pitch, Yaw, Roll)
    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rmat[2, 1], rmat[2, 2])
        y = math.atan2(-rmat[2, 0], sy)
        z = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        x = math.atan2(-rmat[1, 2], rmat[1, 1])
        y = math.atan2(-rmat[2, 0], sy)
        z = 0

    return math.degrees(x), math.degrees(y), math.degrees(z), translation_vector


# --- CHƯƠNG TRÌNH CHÍNH ĐỂ KIỂM THỬ TÍCH HỢP ---
if __name__ == "__main__":
    # Chỉ số các điểm mốc chuẩn của MediaPipe Face Mesh Refined
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [263, 387, 385, 362, 380, 373]
    MOUTH = [78, 81, 13, 308, 14, 312] # Viền môi trong

    cap = cv2.VideoCapture(0)
    print("Đang khởi động Camera tích hợp 5 chỉ số... Nhấn 'q' để thoát.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 1. Tính toán EAR (Trung bình cộng 2 mắt)
                left_ear = calculate_ear(face_landmarks, LEFT_EYE)
                right_ear = calculate_ear(face_landmarks, RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # 2. Tính toán MAR
                mar = calculate_mar(face_landmarks, MOUTH)
                
                # 3. Tính toán Head Pose
                pitch, yaw, roll, nose_2d = get_head_pose(frame, face_landmarks)

                # --- ĐÂY CHÍNH LÀ VECTOR ĐẶC TRƯNG MÀ MÔ HÌNH SẼ HỌC ---
                feature_vector = [avg_ear, mar, pitch, yaw, roll]
                print(f"Feature Vector: [EAR: {avg_ear:.3f}, MAR: {mar:.3f}, P: {pitch:.1f}, Y: {yaw:.1f}, R: {roll:.1f}]")

                # Vẽ các chỉ số hiển thị trực quan lên màn hình UI màn hình máy tính
                cv2.putText(frame, f"EAR (Eye): {avg_ear:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"MAR (Mouth): {mar:.3f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Pitch (Cui/Ngua): {int(pitch)}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw (Trai/Phai): {int(yaw)}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll (Nghieng): {int(roll)}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Vẽ cảnh báo nhanh dựa trên ngưỡng cứng (Chỉ để Test thuật toán)
                if avg_ear < 0.20:
                    cv2.putText(frame, "EYES CLOSED!", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if mar > 0.50:
                    cv2.putText(frame, "YAWNING!", (350, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Driver Feature Extractor - Week 1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()