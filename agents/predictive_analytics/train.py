"""Train development-only churn, cash-flow, and platform-volume forecast artefacts."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, roc_auc_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from shared.training import ensure_output_dir, ensure_synthetic, write_json, write_model_card  # noqa: E402


CHURN_FEATURES = ["days_since_last_transaction", "monthly_transaction_count_trend", "product_count", "complaint_count_12m", "channel_usage_score"]
CASHFLOW_FEATURES = ["balance_lag_1", "balance_mean_7", "balance_mean_30", "balance_trend_7", "day_of_week"]


def train(activity_path: str, balances_path: str, volume_path: str, output_dir: str) -> dict[str, float | int | bool]:
    activity = pd.read_csv(activity_path)
    balances = pd.read_csv(balances_path)
    volume = pd.read_csv(volume_path)
    for name, frame in [("activity", activity), ("balances", balances), ("volume", volume)]:
        ensure_synthetic(frame, name)
    X_train, X_test, y_train, y_test = train_test_split(activity[CHURN_FEATURES], activity.churned_next_90_days, test_size=0.2, random_state=42, stratify=activity.churned_next_90_days)
    churn = lgb.LGBMClassifier(objective="binary", n_estimators=180, learning_rate=0.05, num_leaves=20, min_child_samples=45, random_state=42, verbosity=-1)
    churn.fit(X_train, y_train)
    churn_probability = churn.predict_proba(X_test)[:, 1]

    balance_frame = balances.sort_values(["customer_id", "ds"]).copy()
    balance_frame["balance_lag_1"] = balance_frame.groupby("customer_id").balance_ngn.shift(1)
    balance_frame["balance_mean_7"] = balance_frame.groupby("customer_id").balance_ngn.transform(lambda values: values.rolling(7).mean())
    balance_frame["balance_mean_30"] = balance_frame.groupby("customer_id").balance_ngn.transform(lambda values: values.rolling(30).mean())
    balance_frame["balance_trend_7"] = balance_frame.groupby("customer_id").balance_ngn.transform(lambda values: values.diff(7))
    balance_frame["target_balance_30d"] = balance_frame.groupby("customer_id").balance_ngn.shift(-30)
    balance_frame["day_of_week"] = pd.to_datetime(balance_frame.ds).dt.dayofweek
    balance_frame = balance_frame.dropna().reset_index(drop=True)
    split_index = int(len(balance_frame) * 0.80)
    cash_train, cash_test = balance_frame.iloc[:split_index], balance_frame.iloc[split_index:]
    cashflow = Ridge(alpha=4.0).fit(cash_train[CASHFLOW_FEATURES], cash_train.target_balance_30d)
    cash_prediction = np.maximum(0, cashflow.predict(cash_test[CASHFLOW_FEATURES]))

    volume_frame = volume.sort_values("ds").copy()
    volume_frame["lag_1"] = volume_frame.transaction_volume.shift(1)
    volume_frame["lag_7"] = volume_frame.transaction_volume.shift(7)
    volume_frame["lag_30"] = volume_frame.transaction_volume.shift(30)
    volume_frame["day_of_week"] = pd.to_datetime(volume_frame.ds).dt.dayofweek
    volume_frame = volume_frame.dropna().reset_index(drop=True)
    volume_features = ["lag_1", "lag_7", "lag_30", "day_of_week"]
    volume_train, volume_test = volume_frame.iloc[:-60], volume_frame.iloc[-60:]
    volume_model = Ridge(alpha=3.0).fit(volume_train[volume_features], volume_train.transaction_volume)
    volume_prediction = np.maximum(0, volume_model.predict(volume_test[volume_features]))
    metrics = {
        "churn_auc_roc": round(float(roc_auc_score(y_test, churn_probability)), 4),
        "cashflow_mae_ngn": round(float(mean_absolute_error(cash_test.target_balance_30d, cash_prediction)), 2),
        "cashflow_mape": round(float(mean_absolute_percentage_error(cash_test.target_balance_30d, cash_prediction)), 4),
        "volume_mae": round(float(mean_absolute_error(volume_test.transaction_volume, volume_prediction)), 2),
        "volume_mape": round(float(mean_absolute_percentage_error(volume_test.transaction_volume, volume_prediction)), 4),
        "synthetic_only": True,
    }
    output = ensure_output_dir(output_dir)
    with (output / "churn_lgbm.pkl").open("wb") as handle:
        pickle.dump({"model": churn, "feature_columns": CHURN_FEATURES}, handle)
    with (output / "cashflow_ridge.pkl").open("wb") as handle:
        pickle.dump({"model": cashflow, "feature_columns": CASHFLOW_FEATURES}, handle)
    with (output / "volume_autoreg_ridge.pkl").open("wb") as handle:
        pickle.dump({"model": volume_model, "feature_columns": volume_features}, handle)
    write_json(output / "evaluation_report.json", metrics)
    write_model_card(
        output, "Predictive Analytics", "LightGBM churn, Ridge cash-flow, autoregressive Ridge volume forecast", "synthetic-1.0.0",
        activity_path, CHURN_FEATURES + CASHFLOW_FEATURES, metrics,
        ["Forecast paths are generated from artificial balance and volume patterns.", "A predicted churn probability is a relationship-management prompt, not a customer treatment decision.", "All forecasting models require backtesting, drift monitoring, and approved real-data retraining before deployment."],
        "Provide advisory churn prioritisation and chart-ready balance/volume forecasts to authorised users.",
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity-path", default="data/synthetic/customer_activity.csv")
    parser.add_argument("--balances-path", default="data/synthetic/daily_balances.csv")
    parser.add_argument("--volume-path", default="data/synthetic/platform_daily_volume.csv")
    parser.add_argument("--output-dir", default="agents/predictive_analytics/models")
    args = parser.parse_args()
    print(train(args.activity_path, args.balances_path, args.volume_path, args.output_dir))
