import streamlit as st
import pandas as pd
import numpy as np

def render_benchmark_view(diabetes_bundle, house_bundle, ecommerce_bundle):
    st.markdown("""
        <div style="background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%); padding: 22px; border-radius: 12px; color: white; margin-bottom: 24px;">
            <h2 style="margin: 0; color: white;">📊 Bảng Tổng Hợp So Sánh Hiệu Năng Toàn Diện</h2>
            <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 15px;">
                Đánh giá và so sánh chi tiết giữa <b>3 thuật toán Machine Learning truyền thống</b> và <b>Mạng nơ-ron Deep Learning MLP viết bằng NumPy</b> trên cả 3 bài toán.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("1. 🩺 Bài toán 1: Chẩn đoán Bệnh Tiểu Đường (Classification)")
    d_metrics = diabetes_bundle["meta"]["metrics"]
    d_rows = []
    d_f1_dict = {}
    for name, data in d_metrics.items():
        tm = data["test"]
        d_f1_dict[name] = tm["F1"]
        d_rows.append({
            "Mô hình": name,
            "Phân loại thuật toán": "Deep Learning MLP (NumPy)" if "Deep" in name else "Machine Learning",
            "Accuracy": f"{tm['Accuracy'] * 100:.2f}%",
            "Precision": f"{tm['Precision'] * 100:.2f}%",
            "Recall": f"{tm['Recall'] * 100:.2f}%",
            "F1-Score": f"{tm['F1']:.4f}",
            "ROC-AUC": f"{tm['ROC-AUC']:.4f}",
            "PR-AUC": f"{tm['PR-AUC']:.4f}"
        })
    col_d1, col_d2 = st.columns([3, 2])
    with col_d1:
        st.dataframe(pd.DataFrame(d_rows), use_container_width=True, hide_index=True)
    with col_d2:
        st.caption("So sánh F1-Score (Test Set):")
        st.bar_chart(pd.Series(d_f1_dict, name="F1-Score"))

    st.markdown("---")
    st.subheader("2. 🏠 Bài toán 2: Dự đoán Giá Bất Động Sản (Regression)")
    h_metrics = house_bundle["meta"]["metrics"]
    h_rows = []
    h_r2_dict = {}
    for name, data in h_metrics.items():
        tm = data["test"]
        h_r2_dict[name] = tm["R2"]
        h_rows.append({
            "Mô hình": name,
            "Phân loại thuật toán": "Deep Learning MLP (NumPy)" if "Deep" in name else "Machine Learning",
            "MAE (Tỷ VNĐ)": f"{tm['MAE']:.4f}",
            "RMSE (Tỷ VNĐ)": f"{tm['RMSE']:.4f}",
            "MSE": f"{tm['MSE']:.4f}",
            "Hệ số R²": f"{tm['R2']:.4f}"
        })
    col_h1, col_h2 = st.columns([3, 2])
    with col_h1:
        st.dataframe(pd.DataFrame(h_rows), use_container_width=True, hide_index=True)
    with col_h2:
        st.caption("So sánh Hệ số xác định R² (Càng gần 1.0 càng tốt):")
        st.bar_chart(pd.Series(h_r2_dict, name="Hệ số R²"))

    st.markdown("---")
    st.subheader("3. 🛍️ Bài toán 3: E-Commerce Customer Interest Discovery (Classification)")
    e_metrics = ecommerce_bundle["meta"]["metrics"]
    e_rows = []
    e_f1_dict = {}
    for name, data in e_metrics.items():
        tm = data["test"]
        e_f1_dict[name] = tm["F1"]
        e_rows.append({
            "Mô hình": name,
            "Phân loại thuật toán": "Deep Learning MLP (NumPy)" if "Deep" in name else "Machine Learning",
            "Accuracy": f"{tm['Accuracy'] * 100:.2f}%",
            "Precision": f"{tm['Precision'] * 100:.2f}%",
            "Recall": f"{tm['Recall'] * 100:.2f}%",
            "F1-Score": f"{tm['F1']:.4f}",
            "ROC-AUC": f"{tm['ROC-AUC']:.4f}",
            "MSE Loss": f"{tm['MSE']:.4f}"
        })
    col_e1, col_e2 = st.columns([3, 2])
    with col_e1:
        st.dataframe(pd.DataFrame(e_rows), use_container_width=True, hide_index=True)
    with col_e2:
        st.caption("So sánh F1-Score (Test Set):")
        st.bar_chart(pd.Series(e_f1_dict, name="F1-Score"))

    st.markdown("---")
    st.subheader("💡 Nhận xét và Phân tích chuyên sâu")
    st.markdown("""
- **Machine Learning dạng cây (Decision Tree, Random Forest):**
  - Xử lý rất tốt các tương tác phi tuyến và cấu trúc dữ liệu bảng (tabular data), đặc biệt với bài toán **Giá Nhà** (Random Forest đạt R² ~ 0.84) và **Tiểu đường** (F1 ~ 0.80).
- **Mô hình tuyến tính (Logistic / Linear Regression):**
  - Thời gian huấn luyện và suy luận cực nhanh. Với bài toán **E-Commerce TF-IDF**, Logistic Regression thể hiện ưu thế vượt trội (F1 > 0.91) nhờ không gian đặc trưng văn bản có số chiều lớn và tính chất thưa (sparse linear boundary).
- **Mạng nơ-ron Deep Learning MLP tự code bằng NumPy:**
  - Hoạt động ổn định trên cả 3 bài toán nhờ chuẩn hóa đầu vào kỹ lưỡng (`StandardScaler`, `OneHotEncoder`, `TF-IDF`) và cơ chế mini-batch gradient descent kết hợp kích hoạt ReLU + Sigmoid/Linear.
  - Kết quả đạt tương đương với các thư viện tiêu chuẩn mà không phụ thuộc vào bất kỳ framework Deep Learning nặng nào (như PyTorch hay TensorFlow).
    """)

