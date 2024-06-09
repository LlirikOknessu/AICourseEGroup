import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Load validation data and model
X_val = pd.read_csv('data/prepared/X_val.csv')
y_val = pd.read_csv('data/prepared/y_val.csv')
model = joblib.load('data/models/linear_regression_model.joblib')

# Make predictions and evaluate
predictions = model.predict(X_val)
mse = mean_squared_error(y_val, predictions)
r2 = r2_score(y_val, predictions)

print(f"Validation MSE: {mse}")
print(f"Validation R2: {r2}")
