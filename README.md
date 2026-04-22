# Real-time Driver Drowsiness and Distraction Alert System
Đồ án 2026
# I. Cài đặt:
# 1.1Tạo môi trường ảo (Virtual Environment)
Việc sử dụng môi trường ảo giúp tránh xung đột với các thư viện khác trong máy.
# Tạo môi trường ảo có tên là env_cv
python -m venv env_cv
# Kích hoạt môi trường (Windows)
.\env_cv\Scripts\activate
# Kích hoạt môi trường (macOS/Linux)
source env_cv/bin/activate

# 1.2Cài đặt các thư viện cần thiết (Đảm bảo bạn đã kích hoạt môi trường ảo (có chữ (env_cv) ở đầu dòng lệnh)), sau đó chạy:
pip install --upgrade pip opencv-python mediapipe==0.10.11 scipy
