import streamlit as st
import pandas as pd
import numpy as np

def render_ecommerce_view(bundle):
    models = bundle["models"]
    vectorizer = bundle["vectorizer"]
    scaler = bundle["scaler"]
    meta = bundle["meta"]
    samples = meta["samples"]
    metrics = meta["metrics"]
    cat_feature_names = meta["cat_feature_names"]

    st.markdown("""
        <div style="background: linear-gradient(135deg, #654ea3 0%, #eaafc8 100%); padding: 22px; border-radius: 12px; color: white; margin-bottom: 24px;">
            <h2 style="margin: 0; color: white;">🛍️ E-Commerce Customer Interest Discovery (Review Recommendation)</h2>
            <p style="margin: 6px 0 0 0; opacity: 0.95; font-size: 15px;">
                Dự đoán xu hướng và sự quan tâm của khách hàng: Khách hàng có <b>khuyên dùng sản phẩm (Recommend)</b> hay không dựa trên nội dung đánh giá văn bản (TF-IDF NLP), độ tuổi và phân loại sản phẩm. Hỗ trợ so sánh trực quan các mô hình ML và <b>Deep Learning MLP (NumPy)</b>.
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
                "🤖 Chọn mô hình phân loại:",
                ["Tất cả mô hình (So sánh đồng thời)", "Logistic Regression (ML)", "Decision Tree (ML)", "Random Forest (ML)", "Deep Learning MLP (NumPy)"],
                help="Chọn 1 mô hình cụ thể hoặc đối chiếu đồng thời cả 4 mô hình"
            )
        with col_ctrl2:
            st.markdown("<p style='font-weight: 600; margin-bottom: 6px;'>🧪 Tải đánh giá mẫu thử:</p>", unsafe_allow_html=True)
            preset_choice = st.selectbox("Chọn mẫu review:", list(samples.keys()), label_visibility="collapsed")
            if st.button("⚡ Nạp đánh giá mẫu này", use_container_width=True):
                sample_data = samples[preset_choice]
                st.session_state["e_title"] = sample_data["title"]
                st.session_state["e_review"] = sample_data["review"]
                st.session_state["e_age"] = int(sample_data["age"])
                st.session_state["e_division"] = sample_data["division"]
                st.session_state["e_department"] = sample_data["department"]
                st.session_state["e_class"] = sample_data["class_name"]
                st.rerun()

        st.markdown("---")
        st.subheader("📋 Nhập thông tin khách hàng và nội dung Review")

        # Session defaults
        if "e_title" not in st.session_state:
            st.session_state["e_title"] = "Love this elegant dress!"
        if "e_review" not in st.session_state:
            st.session_state["e_review"] = "The fabric feels wonderful, fits perfectly and looks very flattering. Highly recommended to everyone!"
        if "e_age" not in st.session_state:
            st.session_state["e_age"] = 32
        if "e_division" not in st.session_state:
            st.session_state["e_division"] = "General"
        if "e_department" not in st.session_state:
            st.session_state["e_department"] = "Dresses"
        if "e_class" not in st.session_state:
            st.session_state["e_class"] = "Dresses"

        col_cust1, col_cust2, col_cust3, col_cust4 = st.columns(4)
        with col_cust1:
            age = st.slider("Tuổi khách hàng:", min_value=18, max_value=95, value=int(st.session_state["e_age"]))
        with col_cust2:
            division = st.selectbox("Division:", meta["divisions"], index=meta["divisions"].index(st.session_state["e_division"]) if st.session_state["e_division"] in meta["divisions"] else 0)
        with col_cust3:
            department = st.selectbox("Department:", meta["departments"], index=meta["departments"].index(st.session_state["e_department"]) if st.session_state["e_department"] in meta["departments"] else 0)
        with col_cust4:
            class_name = st.selectbox("Class:", meta["classes"], index=meta["classes"].index(st.session_state["e_class"]) if st.session_state["e_class"] in meta["classes"] else 0)

        title = st.text_input("Tiêu đề đánh giá (Title):", value=st.session_state["e_title"], placeholder="Ví dụ: Great quality, exactly as pictured...")
        review_text = st.text_area("Nội dung chi tiết (Review Text):", value=st.session_state["e_review"], height=110, placeholder="Nhập cảm nhận của bạn về sản phẩm...")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Phân Tích & Dự Đoán Xu Hướng", type="primary", use_container_width=True):
            full_review = (title + " " + review_text).strip()
            if not full_review:
                full_review = "no review provided"

            # Preprocess
            X_text = vectorizer.transform([full_review]).toarray()
            X_age = scaler.transform(pd.DataFrame([[age]], columns=["Age"]))
            df_cat = pd.DataFrame([{
                "Division Name": division,
                "Department Name": department,
                "Class Name": class_name
            }])
            df_cat_enc = pd.get_dummies(df_cat, dtype=float).reindex(columns=cat_feature_names, fill_value=0.0)
            X_input = np.hstack([X_text, X_age, df_cat_enc.values]).astype(np.float32)

            st.markdown("### 🛍️ Kết Quả Phân Tích Đánh Giá")

            model_map = {
                "Logistic Regression (ML)": "Logistic Regression",
                "Decision Tree (ML)": "Decision Tree",
                "Random Forest (ML)": "Random Forest",
                "Deep Learning MLP (NumPy)": "Deep Learning MLP"
            }

            if selected_model == "Tất cả mô hình (So sánh đồng thời)":
                res_rows = []
                prob_dict = {}
                for display_name, internal_name in model_map.items():
                    m = models[internal_name]
                    proba = float(m.predict_proba(X_input)[0, 1])
                    pred = int(proba >= 0.5)
                    prob_dict[internal_name] = proba * 100
                    res_rows.append({
                        "Mô hình": internal_name,
                        "Loại mô hình": "Deep Learning (NumPy MLP)" if "Deep" in internal_name else "Machine Learning",
                        "Kết luận": "👍 SẼ KHUYÊN DÙNG (Recommend)" if pred == 1 else "👎 KHÔNG KHUYÊN DÙNG",
                        "Tỷ lệ Recommend (%)": f"{proba * 100:.2f}%",
                        "Độ tin cậy": "Rất cao" if proba > 0.85 or proba < 0.15 else ("Cao" if proba > 0.70 or proba < 0.30 else "Trung bình")
                    })

                res_df = pd.DataFrame(res_rows)
                st.dataframe(res_df, use_container_width=True, hide_index=True)

                st.markdown("#### 📊 Biểu đồ so sánh xác suất khuyên dùng giữa 4 mô hình (%):")
                st.bar_chart(pd.Series(prob_dict, name="Tỷ lệ Recommend (%)"))

                avg_prob = np.mean(list(prob_dict.values()))
                if avg_prob >= 50:
                    st.success(f"🎉 **Đồng thuận tích cực:** Đa số mô hình dự đoán khách hàng **hài lòng và sẽ khuyên dùng** sản phẩm này (Tỷ lệ trung bình: {avg_prob:.1f}%).")
                else:
                    st.warning(f"⚠️ **Cảnh báo không hài lòng:** Các mô hình nhận định khách hàng **sẽ không khuyên dùng** sản phẩm này (Tỷ lệ recommend trung bình chỉ {avg_prob:.1f}%).")

            else:
                target_model_name = model_map[selected_model]
                model = models[target_model_name]
                proba = float(model.predict_proba(X_input)[0, 1])
                pred = int(proba >= 0.5)

                res_c1, res_c2 = st.columns([1, 1])
                with res_c1:
                    if pred == 1:
                        st.markdown(f"""
                            <div style="background-color: #e8f5e9; border: 2px solid #4caf50; border-radius: 10px; padding: 18px; text-align: center;">
                                <h3 style="color: #2e7d32; margin: 0;">👍 KHÁCH HÀNG KHUYÊN DÙNG</h3>
                                <p style="font-size: 16px; color: #333; margin-top: 8px;">Mô hình <b>{target_model_name}</b> dự đoán khách hàng có mức độ hài lòng cao và <b>RECOMMEND</b> sản phẩm này.</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div style="background-color: #ffebee; border: 2px solid #ef5350; border-radius: 10px; padding: 18px; text-align: center;">
                                <h3 style="color: #c62828; margin: 0;">👎 KHÔNG KHUYÊN DÙNG</h3>
                                <p style="font-size: 16px; color: #333; margin-top: 8px;">Mô hình <b>{target_model_name}</b> dự đoán khách hàng <b>KHÔNG RECOMMEND</b> sản phẩm.</p>
                            </div>
                        """, unsafe_allow_html=True)

                with res_c2:
                    st.metric(label="Xác suất Khuyên dùng (Recommend Probability)", value=f"{proba * 100:.2f}%")
                    st.progress(min(max(proba, 0.0), 1.0))
                    confidence = "Rất cao" if proba > 0.85 or proba < 0.15 else ("Cao" if proba > 0.70 or proba < 0.30 else "Trung bình")
                    st.caption(f"Độ tự tin của mô hình: **{confidence}** (Ngưỡng phân loại chuẩn: 50.0%)")

            # Word token analysis
            st.markdown("#### 🔍 Phân tích các từ khóa TF-IDF nhận diện trong đánh giá:")
            feature_words = vectorizer.get_feature_names_out()
            row_weights = X_text[0]
            matched_indices = np.where(row_weights > 0)[0]

            if len(matched_indices) > 0:
                top_words = sorted([(feature_words[i], row_weights[i]) for i in matched_indices], key=lambda x: x[1], reverse=True)[:10]
                kw_df = pd.DataFrame(top_words, columns=["Từ khóa (Vocabulary)", "Trọng số TF-IDF"])
                st.dataframe(kw_df, use_container_width=True, hide_index=True)
            else:
                st.caption("Các từ trong review này không nằm trong từ điển 100 đặc trưng TF-IDF phổ biến nhất của tập dữ liệu huấn luyện.")

    with tab_compare:
        st.subheader("📊 Bảng đánh giá và so sánh các mô hình trên tập Test")
        st.caption("Các metrics tính trên tập kiểm thử Test Set (15% dữ liệu):")

        metric_rows = []
        for m_name, m_data in metrics.items():
            test_m = m_data["test"]
            metric_rows.append({
                "Mô hình": m_name,
                "Loại": "Deep Learning (NumPy MLP)" if "Deep" in m_name else "Machine Learning",
                "Accuracy": f"{test_m['Accuracy'] * 100:.2f}%",
                "Precision": f"{test_m['Precision'] * 100:.2f}%",
                "Recall": f"{test_m['Recall'] * 100:.2f}%",
                "F1-Score": f"{test_m['F1']:.4f}",
                "ROC-AUC": f"{test_m['ROC-AUC']:.4f}",
                "MSE": f"{test_m['MSE']:.4f}"
            })
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

        st.markdown("#### 🧠 Chi tiết mạng nơ-ron Deep Learning MLP (NumPy):")
        st.code("""
Kiến trúc Mạng nơ-ron:
- Input Layer: 133 đặc trưng (100 TF-IDF features + 1 Scaled Age + One-Hot Category features)
- Dense Layer 1: 16 neurons + ReLU Activation
- Dense Layer 2: 8 neurons + ReLU Activation
- Output Layer: 1 neuron + Sigmoid Activation
- Hàm mất mát: Binary Cross-Entropy
- Tối ưu hóa: Mini-batch Gradient Descent tự viết bằng NumPy thuần túy với lan truyền ngược (Backpropagation)
        """, language="text")

    with tab_batch:
        st.subheader("📁 Dự đoán hàng loạt từ file CSV")
        st.write("Tải lên danh sách review để phân loại và khám phá xu hướng khách hàng.")

        sample_batch = pd.DataFrame([
            samples["Đánh giá rất tích cực (Recommend)"],
            samples["Đánh giá tiêu cực (Not Recommend)"],
            samples["Đánh giá trung lập / băn khoăn"]
        ])
        csv_data = sample_batch.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tải file CSV mẫu chuẩn",
            data=csv_data,
            file_name="ecommerce_sample_template.csv",
            mime="text/csv"
        )

        uploaded_file = st.file_uploader("Chọn file CSV đánh giá:", type=["csv"], key="e_upload")
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Xem trước file tải lên:")
                st.dataframe(batch_df.head(), use_container_width=True)

                if st.button("🚀 Chạy phân tích toàn bộ file", key="e_run_batch"):
                    title_col = batch_df["title"] if "title" in batch_df.columns else batch_df["Title"]
                    text_col = batch_df["review"] if "review" in batch_df.columns else batch_df["Review Text"]
                    reviews_combined = (title_col.fillna("") + " " + text_col.fillna("")).tolist()

                    X_b_text = vectorizer.transform(reviews_combined).toarray()
                    age_col = batch_df["age"] if "age" in batch_df.columns else batch_df["Age"]
                    X_b_age = scaler.transform(pd.DataFrame(age_col.values, columns=["Age"]))

                    cat_df = pd.DataFrame({
                        "Division Name": batch_df["division"] if "division" in batch_df.columns else batch_df["Division Name"],
                        "Department Name": batch_df["department"] if "department" in batch_df.columns else batch_df["Department Name"],
                        "Class Name": batch_df["class_name"] if "class_name" in batch_df.columns else batch_df["Class Name"]
                    })
                    df_b_cat_enc = pd.get_dummies(cat_df, dtype=float).reindex(columns=cat_feature_names, fill_value=0.0)
                    X_b_input = np.hstack([X_b_text, X_b_age, df_b_cat_enc.values]).astype(np.float32)

                    target_m = models["Deep Learning MLP" if "Deep" in selected_model else "Logistic Regression"]
                    probas = target_m.predict_proba(X_b_input)[:, 1]
                    preds = (probas >= 0.5).astype(int)

                    batch_df["Xác_suất_Recommend_%"] = np.round(probas * 100, 2)
                    batch_df["Dự_đoán"] = np.where(preds == 1, "Khuyên dùng (Recommend)", "Không khuyên dùng")

                    st.success(f"Đã phân loại thành công cho {len(batch_df)} đánh giá!")
                    st.dataframe(batch_df, use_container_width=True)

                    out_csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("💾 Tải kết quả phân tích (CSV)", data=out_csv, file_name="ecommerce_predictions_result.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Lỗi khi xử lý file CSV: {str(e)}")

