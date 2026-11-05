import streamlit as st
import pandas as pd
import numpy as np

def render_diabetes_view(bundle):
    models = bundle["models"]
    preprocessor = bundle["preprocessor"]
    meta = bundle["meta"]
    samples = meta["samples"]
    metrics = meta["metrics"]

    st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 22px; border-radius: 12px; color: white; margin-bottom: 24px;">
            <h2 style="margin: 0; color: white;">🩺 Chẩn Đoán Bệnh Tiểu Đường (Diabetes Classification)</h2>
            <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 15px;">
                Phân loại nguy cơ mắc bệnh tiểu đường dựa trên thông tin lâm sàng và tiền sử bệnh. Hỗ trợ so sánh trực tiếp giữa các mô hình <b>Machine Learning</b> và mạng nơ-ron đa tầng <b>Deep Learning MLP (NumPy)</b>.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab_predict, tab_compare, tab_batch = st.tabs([
        "🎯 Dự đoán trực tiếp",
        "📊 So sánh hiệu năng mô hình",
        "📁 Dự đoán hàng loạt (CSV)"
    ])

    with tab_predict:
        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            selected_model = st.selectbox(
                "🤖 Chọn mô hình dự đoán:",
                ["Tất cả mô hình (So sánh đồng thời)", "Logistic Regression (ML)", "Decision Tree (ML)", "Random Forest (ML)", "Deep Learning MLP (NumPy)"],
                help="Chọn 1 mô hình cụ thể hoặc chạy tất cả để đối chiếu kết quả"
            )
        with col_ctrl2:
            st.markdown("<p style='font-weight: 600; margin-bottom: 6px;'>🧪 Tải dữ liệu mẫu thử:</p>", unsafe_allow_html=True)
            preset_choice = st.selectbox("Chọn hồ sơ mẫu:", list(samples.keys()), label_visibility="collapsed")
            if st.button("⚡ Nạp dữ liệu mẫu này", use_container_width=True):
                sample_data = samples[preset_choice]
                st.session_state["d_gender"] = sample_data["gender"]
                st.session_state["d_age"] = sample_data["age"]
                st.session_state["d_location"] = sample_data["location"]
                st.session_state["d_race"] = sample_data["race"]
                st.session_state["d_hypertension"] = sample_data["hypertension"]
                st.session_state["d_heart_disease"] = sample_data["heart_disease"]
                st.session_state["d_smoking"] = sample_data["smoking_history"]
                st.session_state["d_bmi"] = float(sample_data["bmi"])
                st.session_state["d_hba1c"] = float(sample_data["hbA1c_level"])
                st.session_state["d_glucose"] = float(sample_data["blood_glucose_level"])
                st.rerun()

        st.markdown("---")
        st.subheader("📋 Nhập thông số lâm sàng bệnh nhân")

        # Session state defaults
        if "d_gender" not in st.session_state:
            st.session_state["d_gender"] = "Female"
        if "d_age" not in st.session_state:
            st.session_state["d_age"] = 45
        if "d_location" not in st.session_state:
            st.session_state["d_location"] = "California"
        if "d_race" not in st.session_state:
            st.session_state["d_race"] = "Asian"
        if "d_hypertension" not in st.session_state:
            st.session_state["d_hypertension"] = 0
        if "d_heart_disease" not in st.session_state:
            st.session_state["d_heart_disease"] = 0
        if "d_smoking" not in st.session_state:
            st.session_state["d_smoking"] = "never"
        if "d_bmi" not in st.session_state:
            st.session_state["d_bmi"] = 26.5
        if "d_hba1c" not in st.session_state:
            st.session_state["d_hba1c"] = 5.8
        if "d_glucose" not in st.session_state:
            st.session_state["d_glucose"] = 110.0

        col_in1, col_in2, col_in3 = st.columns(3)

        with col_in1:
            st.markdown("##### 👤 Nhân khẩu học")
            gender = st.selectbox("Giới tính:", meta["genders"], index=meta["genders"].index(st.session_state["d_gender"]) if st.session_state["d_gender"] in meta["genders"] else 0)
            age = st.slider("Tuổi:", min_value=1, max_value=100, value=int(st.session_state["d_age"]))
            race_options = ["Asian", "Caucasian", "AfricanAmerican", "Hispanic", "Other"]
            race = st.selectbox("Chủng tộc:", race_options, index=race_options.index(st.session_state["d_race"]) if st.session_state["d_race"] in race_options else 0)
            location = st.selectbox("Khu vực / Bang:", meta["locations"], index=meta["locations"].index(st.session_state["d_location"]) if st.session_state["d_location"] in meta["locations"] else 0)

        with col_in2:
            st.markdown("##### 🫀 Tiền sử bệnh lý")
            hypertension_val = st.radio("Tăng huyết áp (Hypertension):", ["Không (0)", "Có (1)"], index=int(st.session_state["d_hypertension"]))
            hypertension = 1 if "Có" in hypertension_val else 0

            heart_disease_val = st.radio("Bệnh tim mạch (Heart Disease):", ["Không (0)", "Có (1)"], index=int(st.session_state["d_heart_disease"]))
            heart_disease = 1 if "Có" in heart_disease_val else 0

            smoking_history = st.selectbox("Tiền sử hút thuốc:", meta["smoking_options"], index=meta["smoking_options"].index(st.session_state["d_smoking"]) if st.session_state["d_smoking"] in meta["smoking_options"] else 0)

        with col_in3:
            st.markdown("##### 🧪 Chỉ số xét nghiệm")
            bmi = st.number_input("Chỉ số BMI (kg/m²):", min_value=10.0, max_value=70.0, value=float(st.session_state["d_bmi"]), step=0.1)
            hba1c_level = st.number_input("Chỉ số HbA1c (%):", min_value=3.0, max_value=16.0, value=float(st.session_state["d_hba1c"]), step=0.1)
            blood_glucose_level = st.number_input("Đường huyết lúc đói (mg/dL):", min_value=40.0, max_value=350.0, value=float(st.session_state["d_glucose"]), step=1.0)

        # Feature preparation
        cardio_risk = hypertension + heart_disease
        glucose_hba1c_interaction = blood_glucose_level * hba1c_level

        row_dict = {
            "year": 2020,
            "gender": gender,
            "age": age,
            "location": location,
            "race:AfricanAmerican": 1 if race == "AfricanAmerican" else 0,
            "race:Asian": 1 if race == "Asian" else 0,
            "race:Caucasian": 1 if race == "Caucasian" else 0,
            "race:Hispanic": 1 if race == "Hispanic" else 0,
            "race:Other": 1 if race == "Other" else 0,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking_history,
            "bmi": bmi,
            "hbA1c_level": hba1c_level,
            "blood_glucose_level": blood_glucose_level,
            "cardio_risk": cardio_risk,
            "glucose_hba1c_interaction": glucose_hba1c_interaction
        }

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Thực hiện Dự đoán", type="primary", use_container_width=True):
            input_df = pd.DataFrame([row_dict])
            X_input = preprocessor.transform(input_df).astype(np.float32)

            st.markdown("### 🎯 Kết Quả Dự Đoán")

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
                        "Loại mô hình": "Deep Learning (NumPy MLP)" if "Deep" in internal_name else "Machine Learning (Scikit-Learn)",
                        "Kết luận": "⚠️ NGUY CƠ TIỂU ĐƯỜNG" if pred == 1 else "✅ Âm tính (Bình thường)",
                        "Xác suất mắc bệnh": f"{proba * 100:.2f}%",
                        "Mức độ rủi ro": "Cao" if proba > 0.65 else ("Trung bình" if proba > 0.35 else "Thấp")
                    })

                res_df = pd.DataFrame(res_rows)
                st.dataframe(res_df, use_container_width=True, hide_index=True)

                st.markdown("#### 📊 So sánh xác suất dự đoán giữa các mô hình:")
                st.bar_chart(pd.Series(prob_dict, name="Xác suất mắc bệnh (%)"))

                avg_prob = np.mean(list(prob_dict.values()))
                if avg_prob >= 50:
                    st.error(f"⚠️ **Cảnh báo trung bình:** Các mô hình đánh giá bệnh nhân có **nguy cơ cao mắc tiểu đường** (Trung bình {avg_prob:.1f}%). Nên kiểm tra y tế chuyên sâu.")
                else:
                    st.success(f"✅ **Đánh giá chung:** Đa số các mô hình kết luận bệnh nhân nằm trong **ngưỡng an toàn** (Trung bình nguy cơ {avg_prob:.1f}%).")

            else:
                target_model_name = model_map[selected_model]
                model = models[target_model_name]
                proba = float(model.predict_proba(X_input)[0, 1])
                pred = int(proba >= 0.5)

                res_c1, res_c2 = st.columns([1, 1])
                with res_c1:
                    if pred == 1:
                        st.markdown(f"""
                            <div style="background-color: #ffebee; border: 2px solid #ef5350; border-radius: 10px; padding: 18px; text-align: center;">
                                <h3 style="color: #c62828; margin: 0;">⚠️ CẢNH BÁO: DƯƠNG TÍNH</h3>
                                <p style="font-size: 16px; color: #333; margin-top: 8px;">Mô hình <b>{target_model_name}</b> dự đoán bệnh nhân có <b>NGUY CƠ MẮC BỆNH TIỂU ĐƯỜNG</b>.</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div style="background-color: #e8f5e9; border: 2px solid #66bb6a; border-radius: 10px; padding: 18px; text-align: center;">
                                <h3 style="color: #2e7d32; margin: 0;">✅ KẾT QUẢ: ÂM TÍNH</h3>
                                <p style="font-size: 16px; color: #333; margin-top: 8px;">Mô hình <b>{target_model_name}</b> dự đoán bệnh nhân <b>KHÔNG MẮC TIỂU ĐƯỜNG</b>.</p>
                            </div>
                        """, unsafe_allow_html=True)

                with res_c2:
                    st.metric(label="Xác suất mắc bệnh ước tính", value=f"{proba * 100:.2f}%")
                    st.progress(min(max(proba, 0.0), 1.0))
                    risk_txt = "Cao" if proba > 0.65 else ("Trung bình" if proba > 0.35 else "Thấp")
                    st.caption(f"Đánh giá rủi ro: **{risk_txt}** (Ngưỡng phân loại chuẩn: 50.0%)")

                # Medical factor breakdown
                st.markdown("#### 🔬 Phân tích các yếu tố rủi ro chính:")
                notes = []
                if blood_glucose_level >= 140:
                    notes.append(f"- 🔴 **Đường huyết cao ({blood_glucose_level} mg/dL):** Vượt ngưỡng bình thường (ngưỡng bình thường lúc đói < 100 mg/dL, tiền tiểu đường 100-125 mg/dL).")
                else:
                    notes.append(f"- 🟢 **Đường huyết bình thường ({blood_glucose_level} mg/dL)**")

                if hba1c_level >= 6.5:
                    notes.append(f"- 🔴 **HbA1c cao ({hba1c_level}%):** Vượt ngưỡng chẩn đoán tiểu đường (≥ 6.5%).")
                elif hba1c_level >= 5.7:
                    notes.append(f"- 🟡 **HbA1c ở mức tiền tiểu đường ({hba1c_level}%):** Nằm trong khoảng cảnh báo (5.7% - 6.4%).")
                else:
                    notes.append(f"- 🟢 **HbA1c bình thường ({hba1c_level}%)**")

                if bmi >= 30:
                    notes.append(f"- 🔴 **BMI béo phì ({bmi}):** BMI ≥ 30 làm tăng đáng kể tình trạng kháng insulin.")
                elif bmi >= 25:
                    notes.append(f"- 🟡 **BMI thừa cân ({bmi}):** Khuyến nghị điều chỉnh lối sống và vận động.")
                else:
                    notes.append(f"- 🟢 **BMI bình thường ({bmi})**")

                if cardio_risk > 0:
                    notes.append(f"- ⚠️ **Tiền sử tim mạch/huyết áp:** Bệnh nhân có {cardio_risk} yếu tố nguy cơ tim mạch.")

                for n in notes:
                    st.markdown(n)

    with tab_compare:
        st.subheader("📊 Bảng đánh giá và so sánh các mô hình trên tập Test")
        st.caption("Các metrics được tính toán độc lập trên tập kiểm thử (Test Set 15%):")

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
                "PR-AUC": f"{test_m['PR-AUC']:.4f}"
            })
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

        st.markdown("#### 🧠 Chi tiết kiến trúc Deep Learning MLP (NumPy):")
        st.code("""
Kiến trúc Mạng nơ-ron:
- Input Layer: 78 đặc trưng (đã chuẩn hóa StandardScaler & One-Hot Encoding)
- Dense Layer 1: 16 neurons + ReLU Activation
- Dense Layer 2: 8 neurons + ReLU Activation
- Output Layer: 1 neuron + Sigmoid Activation
- Hàm mất mát: Binary Cross-Entropy
- Thuật toán tối ưu: Mini-batch Gradient Descent (Batch size = 1024, lr = 0.05)
        """, language="text")

    with tab_batch:
        st.subheader("📁 Dự đoán hàng loạt từ file CSV")
        st.write("Tải lên file dữ liệu bệnh nhân để dự đoán hàng loạt với mô hình được chọn.")

        # Download sample template
        sample_batch = pd.DataFrame([
            samples["Khỏe mạnh / Nguy cơ thấp"],
            samples["Tiền tiểu đường (Nguy cơ trung bình)"],
            samples["Nguy cơ cao (Chẩn đoán tiểu đường)"]
        ])
        csv_data = sample_batch.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tải file CSV mẫu chuẩn",
            data=csv_data,
            file_name="diabetes_sample_template.csv",
            mime="text/csv"
        )

        uploaded_file = st.file_uploader("Chọn file CSV dữ liệu:", type=["csv"], key="d_upload")
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Xem trước dữ liệu tải lên:")
                st.dataframe(batch_df.head(), use_container_width=True)

                if st.button("🚀 Chạy dự đoán cho toàn bộ file", key="d_run_batch"):
                    batch_clean = batch_df.copy()
                    if "year" not in batch_clean.columns:
                        batch_clean["year"] = 2020
                    if "cardio_risk" not in batch_clean.columns:
                        batch_clean["cardio_risk"] = batch_clean["hypertension"] + batch_clean["heart_disease"]
                    if "glucose_hba1c_interaction" not in batch_clean.columns:
                        batch_clean["glucose_hba1c_interaction"] = batch_clean["blood_glucose_level"] * batch_clean["hbA1c_level"]
                    for r_col in ["race:AfricanAmerican", "race:Asian", "race:Caucasian", "race:Hispanic", "race:Other"]:
                        if r_col not in batch_clean.columns:
                            if "race" in batch_clean.columns:
                                r_val = r_col.split(":")[1]
                                batch_clean[r_col] = (batch_clean["race"] == r_val).astype(int)
                            else:
                                batch_clean[r_col] = 0

                    X_batch = preprocessor.transform(batch_clean).astype(np.float32)

                    # Predict with selected or best model
                    target_m = models["Deep Learning MLP" if "Deep" in selected_model else "Random Forest"]
                    probas = target_m.predict_proba(X_batch)[:, 1]
                    preds = (probas >= 0.5).astype(int)

                    batch_df["Xác_suất_tiểu_đường_%"] = np.round(probas * 100, 2)
                    batch_df["Dự_đoán"] = np.where(preds == 1, "Dương tính (Tiểu đường)", "Âm tính (Bình thường)")

                    st.success(f"Đã dự đoán thành công cho {len(batch_df)} dòng dữ liệu!")
                    st.dataframe(batch_df, use_container_width=True)

                    out_csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("💾 Tải kết quả dự đoán (CSV)", data=out_csv, file_name="diabetes_predictions_result.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Lỗi khi xử lý file CSV: {str(e)}")

