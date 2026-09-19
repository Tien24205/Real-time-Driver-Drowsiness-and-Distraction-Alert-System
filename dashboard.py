import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import os
import datetime

st.set_page_config(
    page_title="Hệ thống Giám sát Tài xế v2",
    page_icon="🚗",
    layout="wide"
)

# ─── CSS ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .stMetric { background-color: #1e1e2e; padding: 15px; border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
    .stMetric label { color: #a0a8c0 !important; }
    .block-container { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ─── HEADER ─────────────────────────────────────────────────────────────
st.title("📊 Bảng điều khiển Trợ lý An toàn Tài xế")
st.caption("Cập nhật theo thời gian thực từ hệ thống camera giám sát.")

# ─── TÌM DATABASE ───────────────────────────────────────────────────────
DB_DIR = os.path.dirname(__file__)
DB_CANDIDATES = [
    os.path.join(DB_DIR, "driver_safety.db"),
    r"E:\Project2026\driver_safety.db",
]
DEFAULT_DB_PATH = os.path.join(DB_DIR, "driver_safety.db")
DB_PATH = next((p for p in DB_CANDIDATES if os.path.exists(p)), DEFAULT_DB_PATH)
DB_BACKUP_DIR = os.path.join(DB_DIR, "driver_safety_backups")

# ─── HỖ TRỢ DATABASE ──────────────────────────────────────────────────

def list_backup_files(backup_dir: str):
    if not os.path.isdir(backup_dir):
        return []
    return sorted([
        f for f in os.listdir(backup_dir) if f.lower().endswith(".db")
    ], reverse=True)


@st.cache_data(ttl=10)   # Tự refresh mỗi 10 giây
def get_data(db_path):
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM safety_events ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

if not os.path.exists(DB_PATH):
    st.warning("⚠️ Chưa có file `driver_safety.db` để đọc. Hãy chạy `main_app.py` để tạo và ghi dữ liệu.")


df = get_data(DB_PATH)

if df.empty:
    st.warning("⚠️ Chưa có dữ liệu vi phạm trong database hiện tại.")
    st.info("📌 Hãy chạy `main_app.py` để ghi dữ liệu mới và làm mới dashboard.")


df["timestamp"] = pd.to_datetime(df["timestamp"])

# ─── PHẦN 1: KPI METRICS ─────────────────────────────────────────────────
total_events   = len(df)
drowsy_count   = len(df[df["event_type"].str.contains("Drowsy|Drowsiness|YAWNING", case=False, na=False)])
distract_count = len(df[df["event_type"].str.contains("Distract", case=False, na=False)])
critical_count = len(df[df["severity"] == "CRITICAL"])

risk_level = "🟢 Thấp"
if critical_count > 3 or total_events > 30:
    risk_level = "🔴 Cao (Nguy hiểm)"
elif drowsy_count > 5 or total_events > 15:
    risk_level = "🟡 Trung bình"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Tổng vi phạm",      total_events)
col2.metric("Buồn ngủ / Ngáp",   drowsy_count)
col3.metric("Phân tâm",          distract_count)
col4.metric("Mức nghiêm trọng",  critical_count, help="Số sự kiện có severity=CRITICAL")
col5.metric("Cấp độ rủi ro",     risk_level)

st.divider()

# ─── PHẦN 2: BIỂU ĐỒ PHÂN TÍCH ──────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📈 Diễn biến vi phạm theo thời gian")
    fig_scatter = px.scatter(
        df, x="timestamp", y="event_type", color="severity",
        color_discrete_map={
            "CRITICAL": "#FF4B4B",
            "HIGH":     "#FF8C00",
            "MEDIUM":   "#FFD700",
            "LOW":      "#00CC88"
        },
        title="Lịch sử các sự kiện vi phạm",
        labels={"timestamp": "Thời gian", "event_type": "Loại vi phạm"},
        height=350
    )
    fig_scatter.update_traces(marker_size=10)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right:
    st.subheader("🍩 Tỉ lệ loại vi phạm")
    type_counts = df["event_type"].value_counts().reset_index()
    type_counts.columns = ["event_type", "count"]
    fig_pie = px.pie(
        type_counts, values="count", names="event_type",
        hole=0.45, height=350,
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig_pie.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

# ─── PHẦN 3: BIỂU ĐỒ SEVERITY ────────────────────────────────────────────
st.subheader("📊 Phân bố mức độ nghiêm trọng (Severity)")
sev_order  = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
sev_counts = df["severity"].value_counts().reindex(sev_order, fill_value=0).reset_index()
sev_counts.columns = ["severity", "count"]

fig_bar = px.bar(
    sev_counts, x="severity", y="count", color="severity",
    color_discrete_map={
        "CRITICAL": "#FF4B4B", "HIGH": "#FF8C00", "MEDIUM": "#FFD700", "LOW": "#00CC88"
    },
    title="Số lượng vi phạm theo mức độ nghiêm trọng",
    labels={"severity": "Mức độ", "count": "Số lượng"},
    text="count", height=300
)
fig_bar.update_traces(textposition="outside")
fig_bar.update_layout(showlegend=False)
st.plotly_chart(fig_bar, use_container_width=True)

# ─── PHẦN 4: LỊCH SỬ CHI TIẾT ────────────────────────────────────────────
with st.expander("📄 Xem bảng dữ liệu chi tiết (100 sự kiện gần nhất)"):
    display_df = df.head(100)[["timestamp", "event_type", "severity", "detail"]].copy()
    display_df.columns = ["Thời gian", "Loại vi phạm", "Mức độ", "Chi tiết"]
    st.dataframe(display_df, use_container_width=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────
st.divider()
st.caption(f"📁 Database: `{DB_PATH}` — Tổng {total_events} bản ghi")

# ─── XEM DATABASE CŨ ─────────────────────────────────────────────────────
with st.expander("🕰️ Xem database cũ từ backup"):
    backup_files = list_backup_files(DB_BACKUP_DIR)
    if not backup_files:
        st.info("Không tìm thấy file backup nào trong thư mục `driver_safety_backups`.")
    else:
        selected_backup = st.selectbox("Chọn file backup", backup_files)
        if selected_backup:
            backup_path = os.path.join(DB_BACKUP_DIR, selected_backup)
            try:
                with sqlite3.connect(backup_path) as conn:
                    old_df = pd.read_sql_query("SELECT * FROM safety_events ORDER BY timestamp DESC", conn)
                old_df["timestamp"] = pd.to_datetime(old_df["timestamp"])
                st.write(f"### 📦 Nội dung file backup: `{selected_backup}`")
                st.dataframe(old_df.head(100), use_container_width=True)
            except Exception as e:
                st.error(f"Không thể đọc file backup: {e}")