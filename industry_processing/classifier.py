import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# Load dataset
df = pd.read_csv("ai4i2020.csv")
df.columns = [col.replace("[","").replace("]","").replace(" ","_") for col in df.columns]

X = df.drop(["Machine_failure", "Product_ID", "Type"], axis=1)
y = df["Machine_failure"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define XGBoost model for binary classification
model = xgb.XGBClassifier(
    objective="binary:logistic",  # outputs probability of class 1
    eval_metric="logloss",
    use_label_encoder=False
)

# Train
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

