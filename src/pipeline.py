from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_data() -> pd.DataFrame:
    bunch = load_diabetes(as_frame=True)
    return bunch.frame.copy()


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_eval(X_train, X_test, y_train, y_test):
    models = {
        "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(
            n_estimators=400, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    preds = {}

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge_cv = cross_val_score(
        models["ridge"], X_train, y_train,
        scoring="neg_mean_squared_error",
        cv=cv
    )
    ridge_cv_rmse = (-ridge_cv.mean()) ** 0.5
    results["ridge_cv_rmse"] = ridge_cv_rmse

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        mse = mean_squared_error(y_test, y_hat)
        rmse = mse**0.5
        results[name] = rmse
        preds[name] = y_hat

    return results, preds


def save_residual_plot(y_test, y_hat, out_path: Path):
    residuals = y_test - y_hat
    plt.figure()
    plt.scatter(y_hat, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual plot")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def run_pipeline():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    results, preds = train_and_eval(X_train, X_test, y_train, y_test)

    print("RMSE results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    best_name = min(results, key=results.get)
    Path("reports/best_model.txt").write_text(best_name + "\n")
    save_residual_plot(y_test, preds[best_name], Path("reports/residuals.png"))

    Path("reports/results.txt").write_text(
        "\n".join([f"{k}: {v:.6f}" for k, v in results.items()]) + "\n"
    )
