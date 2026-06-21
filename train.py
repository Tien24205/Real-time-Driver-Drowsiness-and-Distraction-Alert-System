import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# 1. Đọc dữ liệu từ file CSV đã trích xuất
try:
    df = pd.read_csv("drowsiness_features.csv")
except FileNotFoundError:
    print("❌ Lỗi: Chưa có file drowsiness_features.csv. Hãy chạy extract_features.py trước.")
    exit()

X = df[['ear', 'mar']]
y = df['label']

# 2. Chia tập dữ liệu (80% train, 20% test)
# stratify=y giúp đảm bảo tỉ lệ nhãn Drowsy trong tập train và test là như nhau
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Xây dựng Pipeline tối ưu
# C=10: Tăng mức độ phạt khi phân loại sai (giúp tăng accuracy trên tập khó)
# class_weight='balanced': Tự động điều chỉnh trọng số nếu số lượng ảnh 2 lớp lệch nhau
model = make_pipeline(
    StandardScaler(), 
    SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True)
)

print("⏳ Đang huấn luyện mô hình SVM tối ưu...")
model.fit(X_train, y_train)

# 4. Kiểm tra nhanh độ chính xác trên tập test nội bộ
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"✅ Huấn luyện xong!")
print(f"   - Độ chính xác tập Train: {train_acc*100:.2f}%")
print(f"   - Độ chính xác tập Test: {test_acc*100:.2f}%")

# 5. Lưu mô hình
joblib.dump(model, "drowsiness_model.pkl")
print("💾 Đã lưu mô hình tại: drowsiness_model.pkl")