import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from paths import X_PATH, Y_PATH, LSTM_WEIGHTS, SEED

# --- CẤU HÌNH HỆ THỐNG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

# 1. ĐỊNH NGHĨA DATASET CUSTOM CHO PYTORCH
class DriverSequenceDataset(Dataset):
    def __init__(self, x_path, y_path):
        # Tải dữ liệu mảng numpy đã nén từ bước trước của em
        self.X = torch.from_numpy(np.load(x_path)).float()
        self.y = torch.from_numpy(np.load(y_path)).long()
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. ĐỊNH NGHĨA KIẾN TRÚC MẠNG MULTI-TASK LSTM
class MultiTaskLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super(MultiTaskLSTM, self).__init__()
        
        # Tầng học đặc trưng chuỗi thời gian (Temporal Backbone)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Nhánh 1: Phân loại Buồn ngủ (Drowsiness Branch) -> 2 Lớp (0: Tỉnh táo, 1: Buồn ngủ)
        self.drowsy_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
        
        # Nhánh 2: Phân loại Phân tâm (Distraction Branch) -> Thiết kế sẵn cấu trúc đầu ra để chờ dữ liệu sau
        self.distract_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3) 
        )

    def forward(self, x):
        # Đầu vào x có shape: (Batch_size, 30, 5)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Lấy đặc trưng của khung hình cuối cùng trong chuỗi thời gian (Khung hình thứ 30)
        last_time_step_feat = lstm_out[:, -1, :] # shape: (Batch_size, hidden_size)
        
        # Đẩy song song qua 2 nhánh độc lập
        drowsy_logits = self.drowsy_head(last_time_step_feat)
        distract_logits = self.distract_head(last_time_step_feat)
        
        return drowsy_logits, distract_logits

# 3. TIẾN TRÌNH HUÂN LUYỆN CHÍNH
def train_model():
    x_data_path, y_data_path = X_PATH, Y_PATH

    if not (os.path.exists(x_data_path) and os.path.exists(y_data_path)):
        print("Lỗi: Không tìm thấy file dữ liệu nén. Vui lòng chạy lại drowsiness_data_compiler.py!")
        return

    # Khởi tạo dataset và tự động chia tập dữ liệu Train / Validation (80% / 20%)
    full_dataset = DriverSequenceDataset(x_data_path, y_data_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    # Seed co dinh -> evaluate.py tai lap dung tap val nay
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n==========================================")
    print(f"📊 Khởi tạo thành công!")
    print(f"   - Tập Huấn luyện (Train): {len(train_dataset)} mẫu")
    print(f"   - Tập Đánh giá (Validation): {len(val_dataset)} mẫu")
    print(f"   - Thiết bị chạy: {DEVICE}")
    print(f"==========================================")

    # Khởi tạo mạng, hàm Loss và Bộ tối ưu Adam
    model = MultiTaskLSTM().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    # Vòng lặp huấn luyện qua từng Epoch
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            # Forward pass (Đẩy dữ liệu qua mạng)
            drowsy_out, _ = model(batch_x) 
            loss = criterion(drowsy_out, batch_y)
            
            # Backward pass & Tối ưu hóa trọng số
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(drowsy_out, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()
            
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = (correct_train / total_train) * 100
        
        # ĐÁNH GIÁ TRÊN TẬP VALIDATION (VAL)
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                drowsy_out, _ = model(batch_x)
                loss = criterion(drowsy_out, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(drowsy_out, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = (correct_val / total_val) * 100
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] -> Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% || Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        # Cơ chế Checkpoint: Chỉ lưu lại mô hình nếu đạt chỉ số Loss thấp nhất trên tập Validation
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs(os.path.dirname(LSTM_WEIGHTS), exist_ok=True)
            torch.save(model.state_dict(), LSTM_WEIGHTS)
            print("  --> Đã lưu trọng số xuất sắc nhất (Best Checkpoint)!")

    print("\n🎉 HUÂN LUYỆN HOÀN TẤT! File trọng số đã lưu thành công tại: models/best_multitask_lstm.pth")

if __name__ == "__main__":
    train_model()