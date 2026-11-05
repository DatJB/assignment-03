# Assignment 03 - Introduction to Deep Learning

## 🌟 Web Demo Dự Đoán Đa Mô Hình (ML & Deep Learning)

Hệ thống ứng dụng Web Demo trực quan được xây dựng bằng **Streamlit**, tích hợp toàn bộ **12 mô hình** (4 mô hình cho mỗi bài toán gồm 3 mô hình Machine Learning cổ điển và 1 mô hình Deep Learning MLP tự xây dựng bằng NumPy từ đầu).

### 🚀 Khởi chạy Web Demo
```bash
streamlit run app.py
```
Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`

---

## 📌 Các tính năng chính của Web Demo

1. **🩺 1. Chẩn đoán Bệnh Tiểu Đường (Diabetes Classification):**
   - **Mô hình hỗ trợ:** Logistic Regression, Decision Tree, Random Forest, Deep Learning MLP (NumPy).
   - **Tùy chọn:** Dự đoán đơn mô hình hoặc so sánh đồng thời cả 4 mô hình.
   - **Tính năng nổi bật:**
     - Nạp nhanh mẫu thử lâm sàng chỉ với 1 click (Khỏe mạnh, Tiền tiểu đường, Nguy cơ cao).
     - Phân tích chi tiết các yếu tố nguy cơ (BMI, HbA1c, Đường huyết, Tim mạch).
     - Dự đoán hàng loạt qua upload file CSV và tải kết quả.

2. **🏠 2. Dự đoán Giá Bất Động Sản (House Price Regression):**
   - **Mô hình hỗ trợ:** Linear Regression, Decision Tree Regressor, Random Forest Regressor, Deep Learning MLP (NumPy).
   - **Tính năng nổi bật:**
     - Tự động lọc danh sách Quận/Huyện linh hoạt theo Thành phố đã chọn (Hà Nội & TP. Hồ Chí Minh).
     - Nạp mẫu nhà đất có sẵn (Nhà hẻm TP.HCM, Nhà mặt phố Hà Nội, Biệt thự Quận 7).
     - Hiển thị giá dự đoán theo Tỷ VNĐ, VNĐ đầy đủ và đơn giá theo m².
     - Biểu đồ đối chiếu giá dự đoán giữa 4 mô hình.
     - Hỗ trợ định giá hàng loạt từ file CSV.

3. **🛍️ 3. E-commerce Customer-Interest Discovery (NLP & Classification):**
   - **Mô hình hỗ trợ:** Logistic Regression, Decision Tree, Random Forest, Deep Learning MLP (NumPy).
   - **Tính năng nổi bật:**
     - Phân tích văn bản review khách hàng qua vector hóa TF-IDF (100 từ vựng).
     - Nạp mẫu đánh giá tích cực, tiêu cực hoặc trung lập.
     - Trích xuất và trực quan hóa các từ khóa TF-IDF chính có trong review.
     - Phân loại hàng loạt đánh giá từ file CSV.

4. **📊 4. Bảng So Sánh Hiệu Năng Toàn Diện:**
   - Bảng tổng hợp các chỉ số đánh giá trên tập Test độc lập:
     - Phân loại: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC.
     - Hồi quy: MAE, MSE, RMSE, Hệ số xác định R².
   - Biểu đồ cột so sánh trực quan hiệu năng giữa Machine Learning và Deep Learning.

5. **ℹ️ 5. Giới Thiệu & Kiến Trúc Deep Learning NumPy:**
   - Chi tiết cấu trúc mạng nơ-ron đa tầng: $d \rightarrow 16 \rightarrow 8 \rightarrow 1$.
   - Công thức toán học quá trình Lan truyền thuận (Forward Pass) và Lan truyền ngược (Backpropagation).

---

## 📂 Cấu trúc thư mục dự án

```text
assignment-03/
├── app.py                      # Ứng dụng Streamlit chính điều hướng đa trang
├── mlp_numpy.py                # Lớp mô hình Deep Learning MLP (Classifier & Regressor) viết bằng NumPy
├── train_models.py             # Script huấn luyện và đóng gói mô hình
├── artifacts_models/           # Các bundle mô hình, preprocessor và siêu dữ liệu đã huấn luyện
│   ├── diabetes_bundle.joblib
│   ├── house_price_bundle.joblib
│   └── ecommerce_bundle.joblib
├── views/                      # Các trang giao diện module hóa
│   ├── __init__.py
│   ├── diabetes_view.py        # Trang Chẩn đoán Tiểu đường
│   ├── house_price_view.py     # Trang Dự đoán Giá nhà
│   ├── ecommerce_view.py       # Trang Phân tích E-Commerce
│   ├── benchmark_view.py       # Trang So sánh Hiệu năng
│   └── about_view.py           # Trang Kiến trúc & Lý thuyết
├── data/                       # Dữ liệu huấn luyện
│   ├── diabetes_dataset.csv
│   ├── hanoi_hcm_real_estate.csv
│   └── Womens Clothing E-Commerce Reviews.csv
├── diabetes/
│   └── diabetes3.ipynb         # Notebook thí nghiệm bài toán 1
├── house-price/
│   └── house_price_3.ipynb     # Notebook thí nghiệm bài toán 2
└── ecommerce/
    └── ecommerce3.ipynb        # Notebook thí nghiệm bài toán 3
```
