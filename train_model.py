import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("/home/user/Desktop/college_dropout_risk/dropout_risk_cleaned.csv")

# Split features and target
X = df.drop("dropout", axis=1)   # target column name
y = df["dropout"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline (scaler + model)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier())
])

# Train model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "dropout_risk.pkl")

print("✅ Model saved successfully!")
