"""Duong dan dung chung cho toan bo project.

Ghi de bang bien moi truong khi chay tren may khac:
    set DATASET_ROOT=D:\data\Dataset
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset tho (chi can khi compile du lieu / train lai)
DATASET_ROOT   = os.environ.get("DATASET_ROOT", r"E:\Project2026\Dataset")
DROWSINESS_DIR = os.path.join(DATASET_ROOT, "Drownsiness")
DISTRACT_DIR   = os.path.join(DATASET_ROOT, "Distract")

# Du lieu da xu ly
X_PATH = os.path.join(BASE_DIR, "data", "processed", "X_drowsy_seq.npy")
Y_PATH = os.path.join(BASE_DIR, "data", "processed", "y_drowsy_labels.npy")

# Trong so
LSTM_WEIGHTS = os.path.join(BASE_DIR, "models", "best_multitask_lstm.pth")
YOLO_WEIGHTS = os.path.join(BASE_DIR, "yolov8n.pt")
YOLO_CUSTOM  = os.path.join(BASE_DIR, "runs", "detect", "runs", "train",
                            "yolo_distraction-2", "weights", "best.pt")
DATA_YAML    = os.path.join(BASE_DIR, "data.yaml")

# Database
DB_PATH       = os.path.join(BASE_DIR, "driver_safety.db")
DB_BACKUP_DIR = os.path.join(BASE_DIR, "driver_safety_backups")

# Seed dung chung cho train/test split -> so lieu tai lap duoc
SEED = 42
