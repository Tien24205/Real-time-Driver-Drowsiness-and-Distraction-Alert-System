"""Danh gia toan bo he thong - tat ca so lieu deu do that.

Thay the: evaluate_model.py, evaluate_baselines_real.py, evaluate_requirement3.py,
          ablation_study.py, Metrics_Table.py

Chay:
    python evaluate.py              # danh gia tren tap val (20% giu rieng)
    python evaluate.py --full       # them cot tren toan bo du lieu (co ro ri train)

LUU Y: tap val duoc tai lap bang SEED trong paths.py. Neu file .pth duoc train
TRUOC khi seed nay duoc them vao, tap val nay KHONG phai tap val goc -> hay chay
lai `python train_lstm.py` mot lan de so lieu hop le.
"""
import os
import sys
import argparse
import csv

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from paths import X_PATH, Y_PATH, LSTM_WEIGHTS, YOLO_CUSTOM, SEED
from train_lstm import MultiTaskLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EAR_THRESHOLD = 0.21
DEBOUNCE_LIMIT = 15


# ----------------------------------------------------------------------------
# Chi so
# ----------------------------------------------------------------------------
def metrics(y_true, y_pred):
    """Tra ve (accuracy, precision, recall, f1, FAR) theo %, tat ca tu confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn) * 100
    prec = tp / (tp + fp) * 100 if tp + fp else 0.0
    rec = tp / (tp + fn) * 100 if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    far = fp / (fp + tn) * 100 if fp + tn else 0.0
    return acc, prec, rec, f1, far


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------
def predict_static_ear(X):
    """Nguong EAR tinh tren khung hinh cuoi cua chuoi."""
    return (X[:, -1, 0] < EAR_THRESHOLD).astype(int)


def predict_ear_debounce(X):
    """EAR + debounce TRONG tung chuoi 30 frame (khong bac qua cac mau doc lap)."""
    below = X[:, :, 0] < EAR_THRESHOLD
    preds = np.zeros(len(X), dtype=int)
    for i, row in enumerate(below):
        counter = 0
        for flag in row:
            counter = counter + 1 if flag else max(0, counter - 1)
            if counter >= DEBOUNCE_LIMIT:
                preds[i] = 1
                break
    return preds


def predict_random_forest(X_train, y_train, X_val):
    clf = RandomForestClassifier(n_estimators=100, max_depth=12,
                                 random_state=SEED, n_jobs=-1)
    clf.fit(X_train.reshape(len(X_train), -1), y_train)
    return clf.predict(X_val.reshape(len(X_val), -1))


def predict_lstm(X, batch=512):
    model = MultiTaskLSTM().to(DEVICE)
    model.load_state_dict(torch.load(LSTM_WEIGHTS, map_location=DEVICE, weights_only=True))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            logits, _ = model(torch.from_numpy(X[i:i + batch]).float().to(DEVICE))
            out.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------
# YOLO: doc tu log huan luyen that
# ----------------------------------------------------------------------------
def report_yolo():
    """Doc chi so tu results.csv ma Ultralytics ghi ra khi train."""
    csv_path = os.path.normpath(
        os.path.join(os.path.dirname(YOLO_CUSTOM), "..", "results.csv"))
    if not os.path.exists(csv_path):
        print(f"\n[YOLOv8] Bo qua - khong thay {csv_path}")
        return

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    last = {k.strip(): v for k, v in rows[-1].items()}

    prec = float(last["metrics/precision(B)"]) * 100
    rec = float(last["metrics/recall(B)"]) * 100
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print(f"\n[YOLOv8 - phat hien phan tam]  (sau {len(rows)} epoch)")
    print(f"  Precision {prec:6.2f}%   Recall {rec:6.2f}%   F1 {f1:6.2f}%"
          f"   mAP50 {float(last['metrics/mAP50(B)']) * 100:6.2f}%"
          f"   mAP50-95 {float(last['metrics/mAP50-95(B)']) * 100:6.2f}%")


# ----------------------------------------------------------------------------
def split_indices(n):
    """Tai lap dung tap train/val ma train_lstm.py da dung (cung SEED)."""
    train_size = int(0.8 * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED)).numpy()
    return perm[:train_size], perm[train_size:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="danh gia them tren toan bo du lieu (co ro ri tap train)")
    args = ap.parse_args()

    for path in (X_PATH, Y_PATH, LSTM_WEIGHTS):
        if not os.path.exists(path):
            sys.exit(f"Khong tim thay: {path}")

    X = np.load(X_PATH).astype(np.float32)
    y = np.load(Y_PATH).astype(np.int64)
    tr, va = split_indices(len(X))
    X_tr, y_tr, X_va, y_va = X[tr], y[tr], X[va], y[va]

    print(f"Tong {len(X)} chuoi | train {len(tr)} | val {len(va)}  (seed={SEED})")
    print(f"Phan bo val: tinh tao {int((y_va == 0).sum())} | buon ngu {int((y_va == 1).sum())}")

    configs = [
        ("EAR threshold (tinh)", predict_static_ear(X_va)),
        ("EAR + debounce trong chuoi", predict_ear_debounce(X_va)),
        ("Random Forest", predict_random_forest(X_tr, y_tr, X_va)),
        ("LSTM (de xuat)", predict_lstm(X_va)),
    ]

    print(f"\n[LSTM - phat hien buon ngu]  danh gia tren tap val giu rieng")
    header = f"{'Cau hinh':<30} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'FAR':>7}"
    print(header)
    print("-" * len(header))
    for name, pred in configs:
        acc, prec, rec, f1, far = metrics(y_va, pred)
        print(f"{name:<30} | {acc:6.2f}% | {prec:6.2f}% | {rec:6.2f}% | {f1:6.2f}% | {far:6.2f}%")

    tn, fp, fn, tp = confusion_matrix(y_va, configs[-1][1], labels=[0, 1]).ravel()
    print(f"\nConfusion matrix (LSTM):  TN {tn}  FP {fp}  FN {fn}  TP {tp}")

    if args.full:
        acc, prec, rec, f1, far = metrics(y, predict_lstm(X))
        print(f"\n[!] Toan bo {len(X)} mau (GOM CA TAP TRAIN - chi de tham khao, "
              f"khong dung trong bao cao):")
        print(f"    Acc {acc:.2f}%  Prec {prec:.2f}%  Rec {rec:.2f}%  F1 {f1:.2f}%  FAR {far:.2f}%")

    report_yolo()


if __name__ == "__main__":
    main()
