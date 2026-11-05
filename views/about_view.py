import streamlit as st

def render_about_view():
    st.markdown("""
        <div style="background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); padding: 24px; border-radius: 12px; color: white; margin-bottom: 24px;">
            <h2 style="margin: 0; color: white;">ℹ️ Giới Thiệu Đồ Án & Kiến Trúc Deep Learning NumPy</h2>
            <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 15px;">
                Assignment 03 - Introduction to Deep Learning | Phát triển hệ thống demo dự đoán đa mô hình
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("1. 🎯 Mục tiêu đồ án")
    st.markdown("""
Ứng dụng web này hiện thực hóa toàn bộ pipeline từ tiền xử lý, huấn luyện đến suy luận (Inference) và so sánh trực quan giữa các mô hình **Machine Learning cổ điển** và **Mạng nơ-ron Deep Learning MLP viết từ đầu bằng NumPy** trên 3 bài toán thực tế:
1. **Chẩn đoán Bệnh Tiểu Đường (Diabetes Classification):** Dự đoán nguy cơ mắc tiểu đường dựa trên 100,000 hồ sơ bệnh nhân.
2. **Dự đoán Giá Bất Động Sản (Real Estate Regression):** Định giá nhà đất tại Hà Nội & TP. Hồ Chí Minh với hơn 74,000 bất động sản.
3. **Phân tích Review Thương Mại Điện Tử (E-Commerce Customer Interest):** Phân loại cảm xúc và dự đoán khách hàng có khuyên dùng (Recommend) sản phẩm thời trang nữ hay không qua TF-IDF và đặc trưng khách hàng.
    """)

    st.markdown("---")
    st.subheader("2. 🧠 Chi tiết toán học Mạng nơ-ron Deep Learning MLP (NumPy)")
    st.markdown("""
Mô hình Deep Learning trong cả 3 notebook được xây dựng **thuần túy bằng NumPy** mà không sử dụng các framework cấp cao như PyTorch hay TensorFlow:
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📐 Cấu trúc mạng (Architecture)")
        st.markdown(r"""
- **Lớp đầu vào (Input Layer):** Kích thước $d$ ($78$ cho Tiểu đường, $70$ cho Giá nhà, $133$ cho E-commerce).
- **Lớp ẩn 1 (Hidden Layer 1):** $16$ nơ-ron, hàm kích hoạt **ReLU** $f(z) = \max(0, z)$.
- **Lớp ẩn 2 (Hidden Layer 2):** $8$ nơ-ron, hàm kích hoạt **ReLU**.
- **Lớp đầu ra (Output Layer):**
  - Bài toán Phân loại (Classification): $1$ nơ-ron với hàm kích hoạt **Sigmoid** $\sigma(z) = \frac{1}{1 + e^{-z}}$.
  - Bài toán Hồi quy (Regression): $1$ nơ-ron Tuyến tính (Linear), chuẩn hóa mục tiêu $y_{norm} = \frac{y - \mu}{\sigma}$.
        """)

    with col2:
        st.markdown("#### 🔄 Quá trình Lan truyền ngược (Backpropagation)")
        st.markdown(r"""
- **Phân loại (Binary Cross-Entropy):**
  $$\mathcal{L} = -\frac{1}{N}\sum [y \ln \hat{y} + (1 - y)\ln(1 - \hat{y})]$$
  $$dz_3 = \frac{\hat{y} - y}{N}$$
- **Hồi quy (Mean Squared Error):**
  $$\mathcal{L} = \frac{1}{N}\sum (\hat{y} - y)^2$$
  $$dz_3 = \frac{2(\hat{y} - y)}{N}$$
- **Quy tắc chuỗi (Chain Rule):**
  $$dW_l = h_{l-1}^T dz_l, \quad db_l = \sum dz_l$$
  $$dz_{l-1} = (dz_l W_l^T) \odot \mathbb{I}(z_{l-1} > 0)$$
        """)

    st.markdown("---")
    st.subheader("3. 🛠️ Công nghệ sử dụng trong Web Demo")
    st.markdown("""
- **Backend & ML Core:** Python 3.13, NumPy, Pandas, Scikit-Learn, Joblib.
- **Deep Learning Engine:** Custom Multi-Layer Perceptron (NumpyMLPClassifier, NumpyMLPRegressor) với forward pass và backward pass tự viết.
- **Web Interface:** Streamlit với kiến trúc module hóa đa trang, hỗ trợ Form động, nạp mẫu thử 1-click, xử lý file CSV hàng loạt và biểu đồ trực quan.
    """)
