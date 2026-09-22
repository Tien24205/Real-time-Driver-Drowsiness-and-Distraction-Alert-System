import os
from ultralytics import YOLO

from paths import DISTRACT_DIR


def train_distraction_yolo():
    yaml_path = os.path.join(DISTRACT_DIR, "data.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"Lỗi: Không tìm thấy file data.yaml tại {yaml_path}")
        return

    print("🚀 Đang khởi tạo mô hình YOLOv8 Nano (Siêu nhẹ, tối ưu cho chạy Real-time)...")
    # Chúng ta dùng bản 'yolov8n.pt' để đảm bảo tốc độ FPS cực cao khi chạy Real-time trên Laptop
    model = YOLO("yolov8n.pt") 

    print("🏋️‍♂️ Bắt đầu tiến trình huấn luyện Module Phân tâm...")
    # Tiến hành huấn luyện
    model.train(
        data=yaml_path,
        epochs=50,          # Số epoch huấn luyện (có thể tăng lên 100 nếu muốn độ chính xác cao hơn)
        imgsz=640,          # Kích thước ảnh chuẩn đầu vào
        batch=16,           # Batch size, giảm xuống 8 nếu card đồ họa (VRAM) bị tràn
        device=0,           # Chạy bằng GPU (Cuda). Nếu máy không có card rời, đổi thành 'cpu'
        project="runs/train",
        name="yolo_distraction",
        workers=2           # Số luồng nạp dữ liệu
    )
    
    print("🎉 HUÂN LUYỆN YOLO HOÀN TẤT!")
    print("Trọng số xuất sắc nhất được lưu tại: runs/train/yolo_distraction/weights/best.pt")

if __name__ == "__main__":
    train_distraction_yolo()