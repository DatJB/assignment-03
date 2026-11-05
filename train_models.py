import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

from mlp_numpy import NumpyMLPClassifier, NumpyMLPRegressor

ARTIFACTS_DIR = Path("artifacts_models")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

def train_diabetes():
    print("\n" + "="*50)
    print(">>> 1. HUẤN LUYỆN DỰ ÁN: CHẨN ĐOÁN TIỂU ĐƯỜNG")
    print("="*50)
    t0 = time.time()
    data_path = Path("data/diabetes_dataset.csv")
    if not data_path.exists():
        data_path = Path(r"C:\Users\Dat\Downloads\PTHTTM\DATA\diabetes_dataset.csv")

    df = pd.read_csv(data_path).drop_duplicates().copy()
    for col in ["age", "bmi", "hbA1c_level", "blood_glucose_level"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["age"].isna() | df["age"].between(0, 120)) &
            (df["bmi"].isna() | (df["bmi"] > 0)) &
            (df["hbA1c_level"].isna() | (df["hbA1c_level"] > 0)) &
            (df["blood_glucose_level"].isna() | (df["blood_glucose_level"] > 0))].copy()
    df["diabetes"] = df["diabetes"].astype(int)

    X = df.drop(columns=["diabetes"]).copy()
    # Feature engineering as in notebook
    X["cardio_risk"] = X["hypertension"] + X["heart_disease"]
    X["glucose_hba1c_interaction"] = X["blood_glucose_level"] * X["hbA1c_level"]
    y = df["diabetes"].values

    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    numeric_cols = X_train_raw.select_dtypes(include="number").columns.tolist()
    category_cols = X_train_raw.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), category_cols)
    ])

    X_train = preprocessor.fit_transform(X_train_raw).astype(np.float32)
    X_val = preprocessor.transform(X_val_raw).astype(np.float32)
    X_test = preprocessor.transform(X_test_raw).astype(np.float32)

    print(f"Số đặc trưng sau xử lý: {X_train.shape[1]}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "Deep Learning MLP": NumpyMLPClassifier(hidden_dim1=16, hidden_dim2=8, lr=0.05, epochs=60, batch_size=1024, random_state=SEED)
    }

    metrics = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        if name == "Deep Learning MLP":
            model.fit(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)

        # Evaluate on Validation & Test
        y_val_score = model.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_score >= 0.5).astype(int)
        y_test_score = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_score >= 0.5).astype(int)

        metrics[name] = {
            "val": {
                "Accuracy": float(accuracy_score(y_val, y_val_pred)),
                "Precision": float(precision_score(y_val, y_val_pred, zero_division=0)),
                "Recall": float(recall_score(y_val, y_val_pred, zero_division=0)),
                "F1": float(f1_score(y_val, y_val_pred, zero_division=0)),
                "ROC-AUC": float(roc_auc_score(y_val, y_val_score)),
                "PR-AUC": float(average_precision_score(y_val, y_val_score)),
                "Confusion_Matrix": confusion_matrix(y_val, y_val_pred).tolist()
            },
            "test": {
                "Accuracy": float(accuracy_score(y_test, y_test_pred)),
                "Precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
                "Recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
                "F1": float(f1_score(y_test, y_test_pred, zero_division=0)),
                "ROC-AUC": float(roc_auc_score(y_test, y_test_score)),
                "PR-AUC": float(average_precision_score(y_test, y_test_score)),
                "Confusion_Matrix": confusion_matrix(y_test, y_test_pred).tolist()
            }
        }
        print(f"    Validation F1: {metrics[name]['val']['F1']:.4f} | Test F1: {metrics[name]['test']['F1']:.4f}")

    # Presets for quick UI testing
    samples = {
        "Khỏe mạnh / Nguy cơ thấp": {
            "gender": "Female", "age": 28, "location": "California",
            "race": "Asian", "hypertension": 0, "heart_disease": 0,
            "smoking_history": "never", "bmi": 21.5, "hbA1c_level": 4.8, "blood_glucose_level": 85
        },
        "Tiền tiểu đường (Nguy cơ trung bình)": {
            "gender": "Male", "age": 45, "location": "Texas",
            "race": "Caucasian", "hypertension": 1, "heart_disease": 0,
            "smoking_history": "former", "bmi": 28.4, "hbA1c_level": 6.2, "blood_glucose_level": 140
        },
        "Nguy cơ cao (Chẩn đoán tiểu đường)": {
            "gender": "Male", "age": 62, "location": "Florida",
            "race": "AfricanAmerican", "hypertension": 1, "heart_disease": 1,
            "smoking_history": "current", "bmi": 34.2, "hbA1c_level": 8.5, "blood_glucose_level": 220
        }
    }

    # Meta information
    meta = {
        "numeric_cols": numeric_cols,
        "category_cols": category_cols,
        "raw_cols": X_train_raw.columns.tolist(),
        "locations": sorted(df["location"].dropna().unique().tolist()),
        "smoking_options": sorted(df["smoking_history"].dropna().unique().tolist()),
        "genders": sorted(df["gender"].dropna().unique().tolist()),
        "samples": samples,
        "metrics": metrics
    }

    out_file = ARTIFACTS_DIR / "diabetes_bundle.joblib"
    joblib.dump({
        "models": models,
        "preprocessor": preprocessor,
        "meta": meta
    }, out_file)
    print(f"Đã lưu bundle Tiểu đường tại {out_file} ({time.time()-t0:.2f}s)")


def train_house_price():
    print("\n" + "="*50)
    print(">>> 2. HUẤN LUYỆN DỰ ÁN: DỰ ĐOÁN GIÁ NHÀ")
    print("="*50)
    t0 = time.time()
    data_path = Path("data/hanoi_hcm_real_estate.csv")
    if not data_path.exists():
        data_path = Path(r"C:\Users\Dat\Downloads\PTHTTM\DATA\hanoi_hcm_real_estate.csv")

    df = pd.read_csv(data_path)
    data = df.copy()
    data["bedrooms"] = pd.to_numeric(data["Bedrooms"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    data["toilets"] = pd.to_numeric(data["Toilets"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    data = data[data["price_vnd"].between(1e8, 1e11) & data["area"].between(10, 1000)].copy()

    numeric_cols = ["area", "bedrooms", "toilets", "Total Floors"]
    categorical_cols = ["Type of House", "Legal Documents", "district", "city"]

    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())
    for col in categorical_cols:
        data[col] = data[col].fillna("Không rõ")

    data["price_billion"] = data["price_vnd"] / 1e9

    X = pd.get_dummies(data[numeric_cols + categorical_cols], columns=categorical_cols, dtype=float)
    y = data["price_billion"].values
    feature_names = X.columns.tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEED)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    print(f"Số đặc trưng sau mã hóa: {X_train.shape[1]}")

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=10, random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=16, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "Deep Learning MLP": NumpyMLPRegressor(hidden_dim1=16, hidden_dim2=8, lr=0.01, epochs=60, batch_size=1024, random_state=SEED)
    }

    metrics = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        if name == "Deep Learning MLP":
            model.fit(X_train_scaled, y_train, X_val_scaled, y_val)
        else:
            model.fit(X_train_scaled, y_train)

        val_pred = model.predict(X_val_scaled)
        test_pred = model.predict(X_test_scaled)

        metrics[name] = {
            "val": {
                "MAE": float(mean_absolute_error(y_val, val_pred)),
                "MSE": float(mean_squared_error(y_val, val_pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_val, val_pred))),
                "R2": float(r2_score(y_val, val_pred))
            },
            "test": {
                "MAE": float(mean_absolute_error(y_test, test_pred)),
                "MSE": float(mean_squared_error(y_test, test_pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, test_pred))),
                "R2": float(r2_score(y_test, test_pred))
            }
        }
        print(f"    Validation MAE: {metrics[name]['val']['MAE']:.4f} tỷ | R2: {metrics[name]['val']['R2']:.4f}")

    # Districts by city
    districts_by_city = {
        city: sorted(data[data["city"] == city]["district"].unique().tolist())
        for city in data["city"].unique()
    }

    samples = {
        "Nhà hẻm TP.HCM (Bình Thạnh)": {
            "city": "TP Hồ Chí Minh", "district": "Quận Bình Thạnh",
            "type_of_house": "Nhà hẻm, ngõ", "legal": "Sổ hồng",
            "area": 55.0, "bedrooms": 3, "toilets": 2, "floors": 2
        },
        "Nhà mặt phố Hà Nội (Cầu Giấy)": {
            "city": "Hà Nội", "district": "Quận Cầu Giấy",
            "type_of_house": "Nhà mặt tiền", "legal": "Sổ đỏ",
            "area": 85.0, "bedrooms": 4, "toilets": 4, "floors": 5
        },
        "Biệt thự Villa TP.HCM (Quận 7)": {
            "city": "TP Hồ Chí Minh", "district": "Quận 7",
            "type_of_house": "Biệt thự, Villa", "legal": "Sổ hồng",
            "area": 250.0, "bedrooms": 5, "toilets": 5, "floors": 3
        }
    }

    meta = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_names": feature_names,
        "cities": sorted(data["city"].unique().tolist()),
        "districts_by_city": districts_by_city,
        "types_of_house": sorted(data["Type of House"].unique().tolist()),
        "legals": sorted(data["Legal Documents"].unique().tolist()),
        "samples": samples,
        "metrics": metrics
    }

    out_file = ARTIFACTS_DIR / "house_price_bundle.joblib"
    joblib.dump({
        "models": models,
        "scaler": scaler,
        "meta": meta
    }, out_file)
    print(f"Đã lưu bundle Giá nhà tại {out_file} ({time.time()-t0:.2f}s)")


def train_ecommerce():
    print("\n" + "="*50)
    print(">>> 3. HUẤN LUYỆN DỰ ÁN: E-COMMERCE CUSTOMER INTEREST")
    print("="*50)
    t0 = time.time()
    data_path = Path("data/Womens Clothing E-Commerce Reviews.csv")
    if not data_path.exists():
        data_path = Path(r"C:\Users\Dat\Downloads\PTHTTM\DATA\Womens Clothing E-Commerce Reviews.csv")

    df = pd.read_csv(data_path).drop_duplicates().copy()
    df["Title"] = df["Title"].fillna("")
    df["Review Text"] = df["Review Text"].fillna("")
    df["Division Name"] = df["Division Name"].fillna("Unknown")
    df["Department Name"] = df["Department Name"].fillna("Unknown")
    df["Class Name"] = df["Class Name"].fillna("Unknown")
    df["review"] = df["Title"] + " " + df["Review Text"]
    df = df[df["Recommended IND"].isin([0, 1])].copy().reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df["Recommended IND"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["Recommended IND"]
    )

    # 1. TF-IDF
    vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
    X_text_train = vectorizer.fit_transform(train_df["review"]).toarray()
    X_text_val = vectorizer.transform(val_df["review"]).toarray()
    X_text_test = vectorizer.transform(test_df["review"]).toarray()

    # 2. Scaler Age
    scaler = StandardScaler()
    X_age_train = scaler.fit_transform(train_df[["Age"]])
    X_age_val = scaler.transform(val_df[["Age"]])
    X_age_test = scaler.transform(test_df[["Age"]])

    # 3. Categorical one-hot
    cat_cols = ["Division Name", "Department Name", "Class Name"]
    X_cat_train = pd.get_dummies(train_df[cat_cols], dtype=float)
    cat_feature_names = X_cat_train.columns.tolist()

    X_cat_val = pd.get_dummies(val_df[cat_cols], dtype=float).reindex(columns=cat_feature_names, fill_value=0)
    X_cat_test = pd.get_dummies(test_df[cat_cols], dtype=float).reindex(columns=cat_feature_names, fill_value=0)

    # 4. Stack
    X_train = np.hstack([X_text_train, X_age_train, X_cat_train.values]).astype(np.float32)
    X_val = np.hstack([X_text_val, X_age_val, X_cat_val.values]).astype(np.float32)
    X_test = np.hstack([X_text_test, X_age_test, X_cat_test.values]).astype(np.float32)

    y_train = train_df["Recommended IND"].to_numpy()
    y_val = val_df["Recommended IND"].to_numpy()
    y_test = test_df["Recommended IND"].to_numpy()

    print(f"Số đặc trưng sau xử lý: {X_train.shape[1]}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=15, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_leaf=8, random_state=SEED, n_jobs=-1),
        "Deep Learning MLP": NumpyMLPClassifier(hidden_dim1=16, hidden_dim2=8, lr=0.05, epochs=50, batch_size=512, random_state=SEED)
    }

    metrics = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        if name == "Deep Learning MLP":
            model.fit(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)

        y_val_score = model.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_score >= 0.5).astype(int)
        y_test_score = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_score >= 0.5).astype(int)

        metrics[name] = {
            "val": {
                "Accuracy": float(accuracy_score(y_val, y_val_pred)),
                "Precision": float(precision_score(y_val, y_val_pred, zero_division=0)),
                "Recall": float(recall_score(y_val, y_val_pred, zero_division=0)),
                "F1": float(f1_score(y_val, y_val_pred, zero_division=0)),
                "ROC-AUC": float(roc_auc_score(y_val, y_val_score)),
                "MSE": float(mean_squared_error(y_val, y_val_pred)),
                "Confusion_Matrix": confusion_matrix(y_val, y_val_pred).tolist()
            },
            "test": {
                "Accuracy": float(accuracy_score(y_test, y_test_pred)),
                "Precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
                "Recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
                "F1": float(f1_score(y_test, y_test_pred, zero_division=0)),
                "ROC-AUC": float(roc_auc_score(y_test, y_test_score)),
                "MSE": float(mean_squared_error(y_test, y_test_pred)),
                "Confusion_Matrix": confusion_matrix(y_test, y_test_pred).tolist()
            }
        }
        print(f"    Validation F1: {metrics[name]['val']['F1']:.4f} | Test F1: {metrics[name]['test']['F1']:.4f}")

    samples = {
        "Đánh giá rất tích cực (Recommend)": {
            "title": "Absolutely gorgeous dress!",
            "review": "I love this dress so much! The fabric is high quality, fit is perfect and so flattering. Got so many compliments!",
            "age": 34, "division": "General", "department": "Dresses", "class_name": "Dresses"
        },
        "Đánh giá tiêu cực (Not Recommend)": {
            "title": "Terrible quality and poor fit",
            "review": "Very disappointed with this purchase. The material feels cheap and scratchy. It ripped after one wash. Do not buy!",
            "age": 42, "division": "General Petite", "department": "Tops", "class_name": "Blouses"
        },
        "Đánh giá trung lập / băn khoăn": {
            "title": "Cute style but runs small",
            "review": "The color is pretty but sizing is completely off. Much tighter than expected, order a size up.",
            "age": 29, "division": "General", "department": "Bottoms", "class_name": "Pants"
        }
    }

    meta = {
        "cat_feature_names": cat_feature_names,
        "divisions": sorted(df["Division Name"].unique().tolist()),
        "departments": sorted(df["Department Name"].unique().tolist()),
        "classes": sorted(df["Class Name"].unique().tolist()),
        "samples": samples,
        "metrics": metrics
    }

    out_file = ARTIFACTS_DIR / "ecommerce_bundle.joblib"
    joblib.dump({
        "models": models,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "meta": meta
    }, out_file)
    print(f"Đã lưu bundle E-commerce tại {out_file} ({time.time()-t0:.2f}s)")

if __name__ == "__main__":
    t_start = time.time()
    train_diabetes()
    train_house_price()
    train_ecommerce()
    print("\n" + "="*50)
    print(f"HOÀN TẤT HUẤN LUYỆN TOÀN BỘ 3 BÀI TOÁN TRONG {time.time()-t_start:.2f} GIÂY!")
    print("="*50)

