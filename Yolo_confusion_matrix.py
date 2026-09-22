import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ultralytics import YOLO

from paths import YOLO_CUSTOM as MODEL_PATH, DATA_YAML as DATA_YAML_PATH


def generate_real_yolo_confusion_matrix():
    # Kiểm tra sự tồn tại của file weights và file cấu hình yaml
    if not os.path.exists(MODEL_PATH):
        print(f"[Lỗi] Không tìm thấy file mô hình tại: {MODEL_PATH}")
        return
    if not os.path.exists(DATA_YAML_PATH):
        print(f"[Lỗi] Không tìm thấy file cấu hình dữ liệu tại: {DATA_YAML_PATH}")
        return

    print("--- Đang khởi tạo mô hình YOLOv8 và quét tập dữ liệu Validation ---")
    
    # Tải mô hình weights gốc từ thư mục train yolo_distraction-2 của bạn
    model = YOLO(MODEL_PATH)
    
    # Chạy validation trực tiếp dựa trên khai báo trong file data.yaml của bạn
    results = model.val(data=DATA_YAML_PATH, split="val") 
    
    # 2. Trích xuất ma trận dữ liệu thực tế từ đối tượng kết quả của Ultralytics
    raw_cm = results.confusion_matrix.matrix
    
    # Tên 3 lớp tương ứng bám sát theo danh mục thiết kế trong Thesis
    class_names = ['phone', 'drink', 'normal']
    
    # Lọc lấy ma trận gốc 3x3 tương tác trực tiếp giữa 3 lớp này
    cm = raw_cm[:3, :3]
    
    # 3. Chuẩn hóa ma trận theo tỷ lệ phần trăm (%) trên từng dòng Ground Truth
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)

    # 4. Trực quan hóa ma trận nhầm lẫn bằng Seaborn Heatmap
    plt.figure(figsize=(7.5, 6))
    
    # Thiết lập nhãn: Dòng trên hiển thị số lượng mẫu thực tế, dòng dưới hiển thị % tỉ lệ
    labels = np.asarray([
        f"{int(val)}\n({pct:.2%})" for val, pct in zip(cm.flatten(), cm_normalized.flatten())
    ]).reshape(3, 3)
    
    sns.heatmap(
        cm_normalized, 
        annot=labels, 
        fmt="", 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names,
        vmin=0, 
        vmax=1,
        cbar=True
    )
    
    # 5. Đặt tiêu đề biểu đồ theo chuẩn văn phong luận văn học thuật tiếng Anh
    plt.title("Confusion Matrix - YOLOv8 Spatial Distraction Detection\n(Evaluated on Independent Testing Pool)", fontsize=12, pad=15)
    plt.ylabel("Actual Ground Truth", fontsize=11)
    plt.xlabel("Predicted State", fontsize=11)
    
    plt.tight_layout()
    
    # 6. Xuất ảnh với độ phân giải cao 300 DPI để nộp bài và đẩy lên Overleaf
    output_image = "yolo_confusion_matrix.png"
    plt.savefig(output_image, dpi=300)
    plt.show()
    
    print(f"\n[Thành công] Đã trích xuất dữ liệu thực tế và lưu đồ thị tại: '{os.path.abspath(output_image)}'")

if __name__ == "__main__":
    generate_real_yolo_confusion_matrix()