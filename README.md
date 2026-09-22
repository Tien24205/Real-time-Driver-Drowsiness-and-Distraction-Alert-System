# Real-time Driver Drowsiness and Distraction Alert System

Đồ án 2026 — phát hiện buồn ngủ (MediaPipe + LSTM) và phân tâm (YOLOv8) theo thời gian thực,
cảnh báo bằng âm thanh và ghi log SQLite, kèm dashboard thống kê.

## 1. Cài đặt

```bash
python -m venv env_cv
.\env_cv\Scripts\activate          # Windows
source env_cv/bin/activate         # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

> `main_app.py` dùng `winsound` để phát cảnh báo nên **chỉ chạy được trên Windows**.
> Các script train/evaluate thì chạy được trên mọi hệ điều hành.

## 2. File cần có (không nằm trong git)

| File | Cách tạo |
|---|---|
| `models/best_multitask_lstm.pth` | `python train_lstm.py` |
| `data/processed/X_drowsy_seq.npy`, `y_drowsy_labels.npy` | `python drowsiness_data_compiler.py` |
| `yolov8n.pt` | Ultralytics tự tải về lần chạy đầu |
| `runs/.../best.pt` | `python train_yolo.py` |

Dataset thô mặc định tìm ở `E:\Project2026\Dataset`. Máy khác thì đặt biến môi trường:

```bash
set DATASET_ROOT=D:\duong\dan\Dataset
```

Mọi đường dẫn khác tập trung ở [paths.py](paths.py).

## 3. Sử dụng

```bash
python main_app.py                 # chạy hệ thống giám sát (nhấn 'q' để thoát)
streamlit run dashboard.py         # dashboard thống kê vi phạm
```

## 4. Huấn luyện & đánh giá

```bash
python drowsiness_data_compiler.py # ảnh -> chuỗi đặc trưng .npy
python train_lstm.py               # train LSTM (split cố định seed=42)
python evaluate.py                 # đo LSTM + baselines trên tập val giữ riêng
python evaluate.py --full          # thêm số trên toàn bộ data (có rò rỉ train)
python Runtime_table.py            # đo FPS/latency/CPU/GPU + xuất biểu đồ
```

**Quan trọng:** `evaluate.py` tái lập tập val bằng `SEED` trong `paths.py`. Nếu file `.pth`
hiện tại được train trước khi seed này được thêm vào thì tập val đó không phải tập val gốc —
hãy chạy lại `python train_lstm.py` một lần để số liệu hợp lệ.

## 5. Hạn chế đã biết

- `drowsiness_data_compiler.py` tạo chuỗi bằng cách **lặp lại một ảnh tĩnh 30 lần**
  (`[features] * seq_len`), nên tập dữ liệu **không chứa thông tin thời gian thật**.
  Vì vậy LSTM không tận dụng được tính chuỗi, và mọi baseline debounce trong chuỗi
  đều cho kết quả trùng với ngưỡng tĩnh.
- `MultiTaskLSTM.distract_head` chưa được huấn luyện (loss chỉ tính trên nhánh drowsy).
- `generate_ear_plot.py` vẽ từ dữ liệu mô phỏng — chỉ dùng làm hình minh hoạ nguyên lý.
