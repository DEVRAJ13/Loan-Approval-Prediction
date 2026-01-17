import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

# Load data
data = pd.read_csv("data.csv")

X = data.drop("approved", axis=1)
y = data["approved"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline (scaling + model)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Loan approval model trained & saved")
