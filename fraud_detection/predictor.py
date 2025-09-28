import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

def train_fraud_model_and_get_analysis():
    """
    trains fraud detection model & returns a full analysis
    """
    # Load dataset
    data_path = Path(__file__).parent.parent / "data" / "synthetic_fraud_dataset.csv"
    df = pd.read_csv(data_path)

    # quick feature eng
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour_of_Day'] = df['Timestamp'].dt.hour
    df['Day_of_Week'] = df['Timestamp'].dt.dayofweek

    # drop unnecessary cols
    df = df.drop(['Transaction_ID', 'User_ID', 'Timestamp', 'IP_Address_Flag', 'Risk_Score'], axis=1)

    # label encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # define features & target
    features = [col for col in df.columns if col != 'Fraud_Label']
    X = df[features]
    y = df['Fraud_Label']

    # --- Model Training ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # handle class imbalance
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'scale_pos_weight': [scale_pos_weight]
    }

    # Initialize the XGBClassifier
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False
    )

    # Set up GridSearchCV to optimize for F1-score
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # --- Analysis ---
    # 1. Evaluation Metrics
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    # 2. Feature Importance
    feature_importance = best_model.get_booster().get_score(importance_type='weight')
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    # 3. Correlation with Target
    correlation = df.corr()['Fraud_Label'].sort_values(ascending=False)
    
    # 4. Summary Statistics
    summary_stats = X.describe().to_dict()

    return {
        "best_params": grid_search.best_params_,
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
        "feature_importance": dict(sorted_importance),
        "correlation": correlation.to_dict(),
        "summary_stats": summary_stats
    }
