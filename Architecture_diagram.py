import os
from graphviz import Digraph

# Cấu hình đường dẫn di động Graphviz trên máy em
graphviz_paths = [
    r'E:\graphviz\Graphviz-15.0.0-win32\bin',
    r'C:\Program Files\Graphviz\bin'
]

for path in graphviz_paths:
    if os.path.exists(path):
        os.environ["PATH"] += os.pathsep + path
        break

def generate_minimal_structure():
    print("⏳ Generating highly-optimized minimalist architecture diagram...")
    
    # rankdir='TB' (Từ trên xuống) kết hợp splines='ortho' để ép các mũi tên thành đường thẳng vuông góc
    dot = Digraph(comment='Minimal DMS Architecture', format='png')
    dot.attr(rankdir='TB', size='7,9', dpi='300', splines='ortho', nodesep='0.6', ranksep='0.5')
    
    # Định dạng các khối tối giản, viền mảnh chuyên nghiệp
    dot.attr('node', fontname='Arial', fontsize='10', shape='box', style='filled', penwidth='1.2')
    dot.attr('edge', fontname='Arial', fontsize='9', arrowsize='0.7', color='#37474F', penwidth='1.1')

    # ==============================================================================
    # CORE SYSTEM NODES (Simplified & Grouped)
    # ==============================================================================
    # Tầng 1: Input
    dot.node('cam', '📷 Video Stream\n(Camera - 30 FPS)', fillcolor='#E3F2FD', color='#1E88E5')

    # Tầng 2: Parallel Model Input Layers (Cân bằng cấu trúc ngang hàng)
    dot.node('mp', 'MediaPipe Face Mesh\n(Landmark Extraction)', fillcolor='#FFF3E0', color='#FB8C00')
    dot.node('yolo', 'YOLOv8 Nano\n(Spatial Detector)', fillcolor='#E8F5E9', color='#4CAF50', penwidth='1.8')

    # Tầng 3: Temporal Process Layers
    dot.node('lstm', 'LSTM Classifier\n(Temporal Sequence Network)', fillcolor='#E8F5E9', color='#4CAF50', penwidth='1.8')
    
    # Tầng 4: Optimization Filters
    dot.node('calib', 'Dynamic Calibration\n(Alpha = 0.78)', fillcolor='#F3E5F5', color='#9C27B0')
    dot.node('debounce', 'Two-Stage Debounce\n(Threshold >= 15f)', fillcolor='#F3E5F5', color='#9C27B0')

    # Tầng 5: Decision Logic & Output Actuators
    dot.node('logic', 'System Fusion Logic\n(Decision Engine)', fillcolor='#FFF9C4', color='#FBC02D')
    dot.node('alarm', '🚨 Acoustic Alarm\n(winsound Trigger)', fillcolor='#FFEBEE', color='#E53935')
    dot.node('db', '💾 SQLite Database\n(Analytics Logs)', fillcolor='#F5F5F5', color='#9E9E9E')

    # ==============================================================================
    # RECTILINEAR DATAFLOW CONNECTIONS (STRAIGHT ARROWS)
    # ==============================================================================
    # Phân nhánh dữ liệu từ Camera
    dot.edge('cam', 'mp')
    dot.edge('cam', 'yolo')
    
    # Luồng Xử lý Buồn ngủ (Drowsiness Channel - Nhánh thẳng đứng bên trái)
    dot.edge('mp', 'lstm', label=' 5-D Vector String')
    dot.edge('lstm', 'calib')
    dot.edge('calib', 'debounce')
    
    # Hội tụ tín hiệu về Bộ kiểm soát trung tâm
    dot.edge('debounce', 'logic', label=' Drowsy Signal')
    dot.edge('yolo', 'logic', label=' Distract Signal')
    
    # Xuất tín hiệu ngoại vi
    dot.edge('logic', 'alarm')
    dot.edge('logic', 'db')

    # Thực thi xuất file ảnh vật lý
    try:
        output_name = 'system_structure_minimal'
        dot.render(output_name, cleanup=True)
        print(f"🎉 SUCCESS: Minimalist block diagram saved at: '{output_name}.png'")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == '__main__':
    generate_minimal_structure()