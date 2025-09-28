import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import numpy as np

def train_model_and_get_predictions():
    """
    trains machine learning model & returns a full analysis
    """
    # Load dataset
    df = pd.read_csv("data/ai4i2020.csv")
    df.columns = [col.replace("[","").replace("]","").replace(" ","_") for col in df.columns]

    # Define the specific sensor features to be used for training and analysis.
    # This whitelisting approach prevents data leakage from failure-type flags.
    sensor_features = [
        "Air_temperature_K",
        "Process_temperature_K",
        "Rotational_speed_rpm",
        "Torque_Nm",
        "Tool_wear_min",
    ]
    target_col = "Machine_failure"

    # Create the feature set (X) and target (y)
    X = df[sensor_features]
    y = df[target_col]

    # For correlation analysis, combine the features and the target
    analysis_df = df[sensor_features + [target_col]]

    # --- Model Training ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Analysis ---
    # 1. Evaluation Metrics
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    # 2. Feature Importance
    feature_importance = model.get_booster().get_score(importance_type='weight')
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    # 3. Correlation with Target (calculated on the clean analysis_df)
    correlation = analysis_df.corr()[target_col].sort_values(ascending=False)
    
    # 4. Summary Statistics
    summary_stats = X.describe().to_dict()

    return {
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
        "feature_importance": dict(sorted_importance),
        "correlation": correlation.to_dict(),
        "summary_stats": summary_stats
    }
