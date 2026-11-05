import streamlit as st
import joblib
from pathlib import Path
from mlp_numpy import NumpyMLPClassifier, NumpyMLPRegressor
from views.diabetes_view import render_diabetes_view
from views.house_price_view import render_house_price_view
from views.ecommerce_view import render_ecommerce_view
from views.benchmark_view import render_benchmark_view
from views.about_view import render_about_view

st.set_page_config(
    page_title="Hệ Thống Demo Dự Đoán Đa Mô Hình (ML & Deep Learning)",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3c72;
    }
    div[data-testid="stSidebarNav"] {
        display: none;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 8px;
        padding-bottom: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bundles():
    artifacts_dir = Path("artifacts_models")
    d_path = artifacts_dir / "diabetes_bundle.joblib"
    h_path = artifacts_dir / "house_price_bundle.joblib"
    e_path = artifacts_dir / "ecommerce_bundle.joblib"

    if not d_path.exists() or not h_path.exists() or not e_path.exists():
        st.error("Không tìm thấy các file model trong thư mục artifacts_models. Đang tự động huấn luyện lại...")
        import train_models
        train_models.train_diabetes()
        train_models.train_house_price()
        train_models.train_ecommerce()

    diabetes_bundle = joblib.load(d_path)
    house_bundle = joblib.load(h_path)
    ecommerce_bundle = joblib.load(e_path)

    return diabetes_bundle, house_bundle, ecommerce_bundle

try:
    diabetes_bundle, house_bundle, ecommerce_bundle = load_bundles()
except Exception as ex:
    st.error(f"Lỗi tải tài nguyên mô hình: {ex}")
    st.stop()

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 10px 0 16px 0;">
            <h2 style="margin: 0; color: #1e3c72;">🚀 AI PREDICTOR</h2>
            <p style="margin: 4px 0 0 0; color: #666; font-size: 13px;">Assignment 03 - Machine & Deep Learning Suite</p>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "📌 Điều hướng các bài toán:",
        [
            "🩺 1. Chẩn đoán Bệnh Tiểu Đường",
            "🏠 2. Dự đoán Giá Bất Động Sản",
            "🛍️ 3. Phân tích Đánh giá E-Commerce",
            "📊 4. So sánh Hiệu năng Mô hình",
            "ℹ️ 5. Giới thiệu & Kiến trúc Mạng"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ Tổng quan hệ thống")
    st.info("""
    - **12 Mô hình sẵn sàng:** 3 bài toán × (3 ML + 1 Deep Learning NumPy).
    - **Mô hình Deep Learning:** MLP tự lập trình bằng NumPy với Lan truyền ngược (Backprop).
    - **Tốc độ phản hồi:** < 10ms/lần dự đoán.
    """)

    st.markdown("---")
    st.caption("© 2026 Deep Learning Demo | Assignment 03")

# Main Page Routing
if "1. Chẩn đoán Bệnh Tiểu Đường" in page:
    render_diabetes_view(diabetes_bundle)
elif "2. Dự đoán Giá Bất Động Sản" in page:
    render_house_price_view(house_bundle)
elif "3. Phân tích Đánh giá E-Commerce" in page:
    render_ecommerce_view(ecommerce_bundle)
elif "4. So sánh Hiệu năng Mô hình" in page:
    render_benchmark_view(diabetes_bundle, house_bundle, ecommerce_bundle)
elif "5. Giới thiệu & Kiến trúc Mạng" in page:
    render_about_view()

