"""Train and export the diabetes risk model + monitoring artifacts.

Run:
  python train_export.py --input diabetes.xlsx --out_dir .

It will create:
  - model.joblib
  - model_metrics.json
  - feature_importance_top10.csv

Notes:
  - Uses Logistic Regression with class_weight='balanced' to prioritize recall.
  - Includes preprocessing (OneHot + StandardScaler) inside a single Pipeline.
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
    )
    return Pipeline(steps=[("prep", preprocess), ("clf", model)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="diabetes.xlsx")
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_excel(args.input)
    target = "diabetes"

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    pipe = build_pipeline(cat_cols, num_cols)
    pipe.fit(X_train, y_train)

    # Metrics (test)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "class_balance": {
            "positive_rate": float(y.mean()),
            "negative_rate": float(1 - y.mean()),
        },
        "model": {
            "name": "LogisticRegression (class_weight=balanced)",
            "recall": float(recall_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)

    model_path = os.path.join(args.out_dir, "model.joblib")
    metrics_path = os.path.join(args.out_dir, "model_metrics.json")
    fi_path = os.path.join(args.out_dir, "feature_importance_top10.csv")

    joblib.dump(pipe, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance via absolute standardized coefficients
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]

    ohe = prep.named_transformers_["cat"].named_steps["onehot"]
    feature_names = num_cols + list(ohe.get_feature_names_out(cat_cols))
    coefs = clf.coef_.ravel()

    fi = (
        pd.DataFrame({"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)})
        .sort_values("abs_coef", ascending=False)
        .head(10)
    )
    fi.to_csv(fi_path, index=False)

    print("✅ Export completed")
    print("-", model_path)
    print("-", metrics_path)
    print("-", fi_path)


if __name__ == "__main__":
    main()
