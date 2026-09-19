import matplotlib.pyplot as plt
import numpy as np

# 1. Trích xuất dữ liệu từ bảng trong bài báo của bạn
metrics = ['FPS\n(Higher is better)', 'Latency (ms)\n(Lower is better)', 'CPU (%)\n(Lower is better)', 'GPU (%)\n(Lower is better)']
sequential_pipeline = [53.4, 18.72, 13.9, 16.5]
parallel_pipeline = [60.2, 16.61, 10.2, 14.8]

# 2. Cấu hình vị trí và độ rộng của cột
x = np.arange(len(metrics))
width = 0.35  

# 3. Khởi tạo biểu đồ
fig, ax = plt.subplots(figsize=(10, 6))

# Vẽ cột cho Sequential (Màu đỏ/cam nhạt) và Parallel (Màu xanh dương)
rects1 = ax.bar(x - width/2, sequential_pipeline, width, label='Sequential Pipeline', color='#e74c3c')
rects2 = ax.bar(x + width/2, parallel_pipeline, width, label='Parallel Pipeline (Proposed)', color='#2980b9')

# 4. Thêm các thông tin, nhãn dán, tiêu đề
ax.set_ylabel('Scores / Percentages', fontsize=12, fontweight='bold')
ax.set_title('Real-Time Computation and Hardware Resource Profiling', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11)

# Xóa bớt viền biểu đồ (Spines) để nhìn thoáng và hiện đại hơn
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 5. Gắn giá trị số trực tiếp lên đỉnh mỗi cột
def autolabel(rects):
    """Gắn text giá trị lên đầu mỗi cột bar."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # Đẩy text lên 3 điểm so với đỉnh cột
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# 6. Tối ưu layout và lưu thành file ảnh PNG nét căng (300 dpi)
fig.tight_layout()
plt.savefig('runtime_performance.png', dpi=300, bbox_inches='tight')

# Hiển thị lên màn hình
plt.show()

print("✅ Đã tạo thành công ảnh biểu đồ: runtime_performance.png")