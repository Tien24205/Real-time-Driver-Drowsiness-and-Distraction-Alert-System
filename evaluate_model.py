"""
evaluate_model.py
=================
Script đánh giá mô hình MultiTaskLSTM (phát hiện buồn ngủ) trên tập dữ liệu test.

Đầu ra:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix (in ra terminal)
  - File: evaluation_report.txt  (bảng số liệu chi tiết)
  - File: confusion_matrix.png   (ảnh Confusion Matrix đẹp dùng cho thesis)

Cách chạy:
    python evaluate_model.py
"""

import os
import sys
import json
import datetime
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------
# Đường dẫn (tương đối theo vị trí file này)
# ---------------------------------------------------------------
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(BASE_DIR, "models", "best_multitask_lstm.pth")
X_TEST_PATH      = os.path.join(BASE_DIR, "data", "processed", "X_drowsy_seq.npy")
Y_TEST_PATH      = os.path.join(BASE_DIR, "data", "processed", "y_drowsy_labels.npy")
REPORT_OUT_PATH  = os.path.join(BASE_DIR, "evaluation_report.txt")
CM_IMG_OUT_PATH  = os.path.join(BASE_DIR, "confusion_matrix.png")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------
# Import kiến trúc model
# ---------------------------------------------------------------
# Thêm thư mục hiện tại vào sys.path để import từ train_lstm.py
sys.path.insert(0, BASE_DIR)
from train_lstm import MultiTaskLSTM

# ---------------------------------------------------------------
# Hàm tính các chỉ số đánh giá thủ công (không phụ thuộc sklearn)
# ---------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list):
    """
    Tính Accuracy, Precision, Recall, F1 theo từng class và Macro average.
    Trả về dict kết quả và ma trận nhầm lẫn (confusion matrix).
    """
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    accuracy = np.trace(cm) / cm.sum()

    per_class = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[name] = {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "support":   int(cm[i, :].sum())
        }

    macro_precision = np.mean([v["precision"] for v in per_class.values()])
    macro_recall    = np.mean([v["recall"]    for v in per_class.values()])
    macro_f1        = np.mean([v["f1"]        for v in per_class.values()])

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro": {
            "precision": macro_precision,
            "recall":    macro_recall,
            "f1":        macro_f1
        }
    }, cm


def print_confusion_matrix(cm: np.ndarray, class_names: list):
    """In confusion matrix dạng bảng văn bản ra terminal."""
    col_w = max(12, max(len(n) for n in class_names) + 2)
    header = " " * col_w + "".join(f"{'Pred: ' + n:^{col_w}}" for n in class_names)
    print(header)
    print("-" * len(header))
    for i, row_name in enumerate(class_names):
        row = f"{'True: ' + row_name:<{col_w}}" + "".join(f"{cm[i,j]:^{col_w}}" for j in range(len(class_names)))
        print(row)


def save_confusion_matrix_image(cm: np.ndarray, class_names: list, out_path: str):
    """Vẽ và lưu Confusion Matrix dạng ảnh đẹp (dùng matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # Backend không cần GUI
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=12)
        ax.set_yticklabels(class_names, fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=13)
        ax.set_ylabel("True Label", fontsize=13)
        ax.set_title("Confusion Matrix — Drowsiness Detection (LSTM)", fontsize=13, pad=15)

        thresh = cm.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\n✅ Ảnh Confusion Matrix đã lưu tại: {out_path}")
    except ImportError:
        print("\n⚠️  Không tìm thấy matplotlib. Bỏ qua bước lưu ảnh Confusion Matrix.")
        print("   Cài đặt: pip install matplotlib")


# ---------------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------------
def evaluate():
    print("=" * 60)
    print("   ĐÁNH GIÁ MÔ HÌNH LSTM - PHÁT HIỆN BUỒN NGỦ")
    print("=" * 60)

    # 1. Kiểm tra file
    for path, name in [(MODEL_PATH, "Model weights"), (X_TEST_PATH, "X features"), (Y_TEST_PATH, "y labels")]:
        if not os.path.exists(path):
            print(f"❌ Không tìm thấy {name}: {path}")
            sys.exit(1)

    # 2. Nạp dữ liệu
    print("\n📂 Đang nạp dữ liệu...")
    X = np.load(X_TEST_PATH).astype(np.float32)  # (N, 30, 5)
    y = np.load(Y_TEST_PATH).astype(np.int64)     # (N,)

    print(f"   X.shape = {X.shape}  |  y.shape = {y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        label = "Alert" if cls == 0 else "Drowsy"
        print(f"   Class {cls} ({label}): {cnt} mẫu ({cnt/len(y)*100:.1f}%)")

    # 3. Nạp mô hình
    print(f"\n🧠 Đang nạp mô hình từ: {MODEL_PATH}")
    model = MultiTaskLSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()

    # 4. Inference theo batch (tránh OOM)
    BATCH_SIZE = 256
    all_preds  = []
    all_probs  = []

    print(f"\n🔍 Đang chạy inference trên {len(X)} mẫu (batch_size={BATCH_SIZE})...")
    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            end        = min(start + BATCH_SIZE, len(X))
            batch_x    = torch.tensor(X[start:end]).to(DEVICE)
            drowsy_out, _ = model(batch_x)
            probs      = F.softmax(drowsy_out, dim=1).cpu().numpy()
            preds      = np.argmax(probs, axis=1)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs[:, 1].tolist())  # Xác suất class "Drowsy"

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # 5. Tính chỉ số
    class_names = ["Alert (0)", "Drowsy (1)"]
    metrics, cm = compute_metrics(y, y_pred, class_names)

    # 6. In ra terminal
    print("\n" + "=" * 60)
    print("   KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 60)
    print(f"\n📊 Accuracy tổng thể: {metrics['accuracy']*100:.2f}%\n")

    print(f"{'Metric':<20} {'Alert (0)':>12} {'Drowsy (1)':>12} {'Macro Avg':>12}")
    print("-" * 60)
    for metric in ["precision", "recall", "f1"]:
        row = f"{metric.capitalize():<20}"
        for cls in class_names:
            row += f"{metrics['per_class'][cls][metric]*100:>11.2f}%"
        row += f"{metrics['macro'][metric]*100:>11.2f}%"
        print(row)
    print("-" * 60)
    sup_row = f"{'Support':<20}"
    for cls in class_names:
        sup_row += f"{metrics['per_class'][cls]['support']:>12}"
    print(sup_row)

    print("\n📊 Confusion Matrix:")
    print_confusion_matrix(cm, class_names)

    # 7. Lưu ảnh Confusion Matrix
    save_confusion_matrix_image(cm, ["Alert", "Drowsy"], CM_IMG_OUT_PATH)

    # 8. Lưu báo cáo text
    report_lines = [
        "=" * 60,
        "   EVALUATION REPORT — Drowsiness Detection LSTM",
        f"   Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"   Model: {MODEL_PATH}",
        "=" * 60,
        f"\nTotal samples  : {len(y)}",
        f"Accuracy       : {metrics['accuracy']*100:.2f}%\n",
        f"{'Metric':<20} {'Alert (0)':>12} {'Drowsy (1)':>12} {'Macro Avg':>12}",
        "-" * 60,
    ]
    for metric in ["precision", "recall", "f1"]:
        row = f"{metric.capitalize():<20}"
        for cls in class_names:
            row += f"{metrics['per_class'][cls][metric]*100:>11.2f}%"
        row += f"{metrics['macro'][metric]*100:>11.2f}%"
        report_lines.append(row)

    report_lines.append("-" * 60)
    report_lines.append(f"\nConfusion Matrix:\n  True\\Pred  Alert    Drowsy")
    report_lines.append(f"  Alert      {cm[0,0]:>5}    {cm[0,1]:>5}")
    report_lines.append(f"  Drowsy     {cm[1,0]:>5}    {cm[1,1]:>5}")
    report_lines.append(f"\nImages saved: {CM_IMG_OUT_PATH}")

    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n📄 Báo cáo đã lưu tại: {REPORT_OUT_PATH}")
    print("\n✅ ĐÁNH GIÁ HOÀN TẤT!\n")


if __name__ == "__main__":
    evaluate()
