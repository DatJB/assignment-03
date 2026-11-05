import streamlit as st
import pandas as pd
import numpy as np

def render_house_price_view(bundle):
    models = bundle["models"]
    scaler = bundle["scaler"]
    meta = bundle["meta"]
    samples = meta["samples"]
    metrics = bundle["meta"]["metrics"]
    feature_names = meta["feature_names"]
    districts_by_city = meta["districts_by_city"]

    st.markdown("""
        <div style="background: linear-gradient(135deg, #134e5e 0%, #71b280 100%); padding: 22px; border-radius: 12px; color: white; margin-bottom: 24px;">
            <h2 style="margin: 0; color: white;">🏠 Dự Đoán Giá Bất Động Sản (House Price Regression)</h2>
            <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 15px;">
                Định giá bất động sản tại <b>Hà Nội</b> và <b>TP. Hồ Chí Minh</b> theo diện tích, số tầng, số phòng, vị trí và pháp lý. Hỗ trợ đối chiếu mô hình Hồi quy Machine Learning với <b>Deep Learning MLP (NumPy)</b>.
            </p>
        </div>
    """, unsafe_allow_html=True)

    tab_predict, tab_compare, tab_batch = st.tabs([
        "🎯 Dự đoán trực tiếp",
        "📊 So sánh hiệu năng mô hình",
        "📁 Dự đoán hàng loạt (CSV)"
    ])

    with tab_predict:
        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            selected_model = st.selectbox(
                "🤖 Chọn mô hình hồi quy:",
                ["Tất cả mô hình (So sánh đồng thời)", "Linear Regression (ML)", "Decision Tree (ML)", "Random Forest (ML)", "Deep Learning MLP (NumPy)"],
                help="Chọn 1 mô hình hoặc chạy cả 4 để so sánh giá dự đoán"
            )
        with col_ctrl2:
            st.markdown("<p style='font-weight: 600; margin-bottom: 6px;'>🧪 Tải căn nhà mẫu thử:</p>", unsafe_allow_html=True)
            preset_choice = st.selectbox("Chọn mẫu nhà:", list(samples.keys()), label_visibility="collapsed")
            if st.button("⚡ Nạp dữ liệu căn nhà này", use_container_width=True):
                sample_data = samples[preset_choice]
                st.session_state["h_city"] = sample_data["city"]
                st.session_state["h_district"] = sample_data["district"]
                st.session_state["h_type"] = sample_data["type_of_house"]
                st.session_state["h_legal"] = sample_data["legal"]
                st.session_state["h_area"] = float(sample_data["area"])
                st.session_state["h_bedrooms"] = int(sample_data["bedrooms"])
                st.session_state["h_toilets"] = int(sample_data["toilets"])
                st.session_state["h_floors"] = int(sample_data["floors"])
                st.rerun()

        st.markdown("---")
        st.subheader("📋 Nhập thông tin chi tiết bất động sản")

        # Session defaults
        if "h_city" not in st.session_state:
            st.session_state["h_city"] = "TP Hồ Chí Minh"
        if "h_district" not in st.session_state:
            st.session_state["h_district"] = "Quận Bình Thạnh"
        if "h_type" not in st.session_state:
            st.session_state["h_type"] = "Nhà hẻm, ngõ"
        if "h_legal" not in st.session_state:
            st.session_state["h_legal"] = "Sổ hồng"
        if "h_area" not in st.session_state:
            st.session_state["h_area"] = 65.0
        if "h_bedrooms" not in st.session_state:
            st.session_state["h_bedrooms"] = 3
        if "h_toilets" not in st.session_state:
            st.session_state["h_toilets"] = 2
        if "h_floors" not in st.session_state:
            st.session_state["h_floors"] = 2

        col_loc1, col_loc2, col_spec1, col_spec2 = st.columns(4)

        with col_loc1:
            city = st.selectbox("Thành phố:", meta["cities"], index=meta["cities"].index(st.session_state["h_city"]) if st.session_state["h_city"] in meta["cities"] else 0)

        with col_loc2:
            available_districts = districts_by_city.get(city, meta["districts_by_city"][meta["cities"][0]])
            dist_idx = available_districts.index(st.session_state["h_district"]) if st.session_state["h_district"] in available_districts else 0
            district = st.selectbox("Quận / Huyện:", available_districts, index=dist_idx)

        with col_spec1:
            type_of_house = st.selectbox("Loại hình BĐS:", meta["types_of_house"], index=meta["types_of_house"].index(st.session_state["h_type"]) if st.session_state["h_type"] in meta["types_of_house"] else 0)

        with col_spec2:
            legal = st.selectbox("Tình trạng pháp lý:", meta["legals"], index=meta["legals"].index(st.session_state["h_legal"]) if st.session_state["h_legal"] in meta["legals"] else 0)

        col_num1, col_num2, col_num3, col_num4 = st.columns(4)
        with col_num1:
            area = st.number_input("Diện tích đất (m²):", min_value=10.0, max_value=1000.0, value=float(st.session_state["h_area"]), step=5.0)
        with col_num2:
            bedrooms = st.number_input("Số phòng ngủ:", min_value=1, max_value=20, value=int(st.session_state["h_bedrooms"]), step=1)
        with col_num3:
            toilets = st.number_input("Số phòng vệ sinh:", min_value=1, max_value=20, value=int(st.session_state["h_toilets"]), step=1)
        with col_num4:
            floors = st.number_input("Tổng số tầng:", min_value=1, max_value=30, value=int(st.session_state["h_floors"]), step=1)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Dự Đoán Giá Bất Động Sản", type="primary", use_container_width=True):
            raw_input = {
                "area": area,
                "bedrooms": bedrooms,
                "toilets": toilets,
                "Total Floors": floors,
                "Type of House": type_of_house,
                "Legal Documents": legal,
                "district": district,
                "city": city
            }
            df_raw = pd.DataFrame([raw_input])
            df_encoded = pd.get_dummies(df_raw, columns=["Type of House", "Legal Documents", "district", "city"], dtype=float)
            df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0.0)
            X_input_scaled = scaler.transform(df_aligned).astype(np.float32)

            st.markdown("### 💰 Kết Quả Ước Tính Giá Nhà")

            model_map = {
                "Linear Regression (ML)": "Linear Regression",
                "Decision Tree (ML)": "Decision Tree",
                "Random Forest (ML)": "Random Forest",
                "Deep Learning MLP (NumPy)": "Deep Learning MLP"
            }

            if selected_model == "Tất cả mô hình (So sánh đồng thời)":
                res_rows = []
                price_dict = {}
                for display_name, internal_name in model_map.items():
                    m = models[internal_name]
                    pred_billion = float(m.predict(X_input_scaled)[0])
                    # Guard against negative price in linear models
                    pred_billion = max(0.1, pred_billion)
                    price_dict[internal_name] = pred_billion
                    price_million_m2 = (pred_billion * 1000) / area
                    res_rows.append({
                        "Mô hình": internal_name,
                        "Loại mô hình": "Deep Learning (NumPy MLP)" if "Deep" in internal_name else "Machine Learning",
                        "Giá dự đoán (Tỷ VNĐ)": f"{pred_billion:.3f} tỷ",
                        "Giá quy đổi (VNĐ)": f"{pred_billion * 1e9:,.0f} đ",
                        "Đơn giá theo m²": f"{price_million_m2:.1f} triệu/m²"
                    })

                res_df = pd.DataFrame(res_rows)
                st.dataframe(res_df, use_container_width=True, hide_index=True)

                st.markdown("#### 📊 Biểu đồ so sánh giá dự đoán giữa 4 mô hình (Tỷ VNĐ):")
                st.bar_chart(pd.Series(price_dict, name="Giá nhà ước tính (Tỷ VNĐ)"))

                avg_price = np.mean(list(price_dict.values()))
                st.info(f"💡 **Mức giá đồng thuận trung bình giữa các mô hình:** **{avg_price:.3f} tỷ VNĐ** (~ {avg_price * 1e9:,.0f} VNĐ, tương đương {(avg_price*1000)/area:.1f} triệu VNĐ/m²).")

            else:
                target_model_name = model_map[selected_model]
                model = models[target_model_name]
                pred_billion = float(model.predict(X_input_scaled)[0])
                pred_billion = max(0.1, pred_billion)
                price_million_m2 = (pred_billion * 1000) / area

                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.metric(
                        label=f"Giá ước tính ({target_model_name})",
                        value=f"{pred_billion:.3f} Tỷ VNĐ"
                    )
                with col_res2:
                    st.metric(
                        label="Giá đầy đủ (VNĐ)",
                        value=f"{pred_billion * 1e9:,.0f} đ"
                    )
                with col_res3:
                    st.metric(
                        label="Đơn giá trên m²",
                        value=f"{price_million_m2:.1f} Triệu/m²"
                    )

                st.markdown(f"""
                    <div style="background-color: #f0f7f4; border-left: 5px solid #2e7d32; padding: 14px 18px; border-radius: 4px; margin-top: 15px;">
                        <b>Đặc điểm định giá:</b> Căn nhà tại <b>{district}, {city}</b> diện tích <b>{area} m²</b> ({floors} tầng, {bedrooms} PN, {toilets} WC, pháp lý {legal}) được mô hình <b>{target_model_name}</b> định giá khoảng <b>{pred_billion:.3f} tỷ VNĐ</b>.
                    </div>
                """, unsafe_allow_html=True)

    with tab_compare:
        st.subheader("📊 Bảng đánh giá và so sánh các mô hình trên tập Test")
        st.caption("Các metrics tính trên tập Test độc lập (15% dữ liệu):")

        metric_rows = []
        for m_name, m_data in metrics.items():
            test_m = m_data["test"]
            metric_rows.append({
                "Mô hình": m_name,
                "Loại": "Deep Learning (NumPy MLP)" if "Deep" in m_name else "Machine Learning",
                "MAE (Sai số trung bình tuyệt đối)": f"{test_m['MAE']:.4f} tỷ",
                "RMSE (Căn bậc hai MSE)": f"{test_m['RMSE']:.4f} tỷ",
                "MSE": f"{test_m['MSE']:.4f}",
                "Hệ số xác định R²": f"{test_m['R2']:.4f}"
            })
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

        st.markdown("#### 🧠 Chi tiết mạng nơ-ron Deep Learning MLP (NumPy):")
        st.code("""
Kiến trúc Mạng nơ-ron:
- Input Layer: 70 đặc trưng (đã chuẩn hóa StandardScaler & One-Hot Encoding)
- Dense Layer 1: 16 neurons + ReLU Activation
- Dense Layer 2: 8 neurons + ReLU Activation
- Output Layer: 1 neuron (Linear Output)
- Chuẩn hóa đầu ra (y): (y - y_mean) / y_std trong quá trình học
- Hàm mất mát: Mean Squared Error (MSE Loss)
- Cập nhật tham số: Mini-batch Gradient Descent tự động với lan truyền ngược (Backpropagation)
        """, language="text")

    with tab_batch:
        st.subheader("📁 Dự đoán hàng loạt giá nhà từ file CSV")
        st.write("Tải lên danh sách các căn nhà để ước tính giá đồng loạt.")

        sample_batch = pd.DataFrame([
            samples["Nhà hẻm TP.HCM (Bình Thạnh)"],
            samples["Nhà mặt phố Hà Nội (Cầu Giấy)"],
            samples["Biệt thự Villa TP.HCM (Quận 7)"]
        ])
        csv_data = sample_batch.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tải file CSV mẫu chuẩn",
            data=csv_data,
            file_name="house_price_sample_template.csv",
            mime="text/csv"
        )

        uploaded_file = st.file_uploader("Chọn file CSV nhà đất:", type=["csv"], key="h_upload")
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Xem trước file tải lên:")
                st.dataframe(batch_df.head(), use_container_width=True)

                if st.button("🚀 Chạy dự đoán giá toàn bộ file", key="h_run_batch"):
                    clean_batch = pd.DataFrame()
                    clean_batch["area"] = batch_df["area"]
                    clean_batch["bedrooms"] = batch_df["bedrooms"]
                    clean_batch["toilets"] = batch_df["toilets"]
                    clean_batch["Total Floors"] = batch_df["floors"] if "floors" in batch_df.columns else batch_df["Total Floors"]
                    clean_batch["Type of House"] = batch_df["type_of_house"] if "type_of_house" in batch_df.columns else batch_df["Type of House"]
                    clean_batch["Legal Documents"] = batch_df["legal"] if "legal" in batch_df.columns else batch_df["Legal Documents"]
                    clean_batch["district"] = batch_df["district"]
                    clean_batch["city"] = batch_df["city"]

                    df_enc = pd.get_dummies(clean_batch, columns=["Type of House", "Legal Documents", "district", "city"], dtype=float)
                    df_aligned = df_enc.reindex(columns=feature_names, fill_value=0.0)
                    X_b_scaled = scaler.transform(df_aligned).astype(np.float32)

                    target_m = models["Deep Learning MLP" if "Deep" in selected_model else "Random Forest"]
                    preds_billion = np.maximum(0.1, target_m.predict(X_b_scaled))

                    batch_df["Giá_dự_đoán_tỷ_VND"] = np.round(preds_billion, 3)
                    batch_df["Giá_VNĐ"] = np.round(preds_billion * 1e9, 0)
                    batch_df["Đơn_giá_triệu_m2"] = np.round((preds_billion * 1000) / batch_df["area"], 1)

                    st.success(f"Đã định giá thành công cho {len(batch_df)} bất động sản!")
                    st.dataframe(batch_df, use_container_width=True)

                    out_csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("💾 Tải kết quả định giá (CSV)", data=out_csv, file_name="house_price_predictions_result.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Lỗi khi xử lý file CSV: {str(e)}")

