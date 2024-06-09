import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Load full dataset
X_full = pd.read_csv('data/prepared/X_full.csv')
y_full = pd.read_csv('data/prepared/y_full.csv')

# Train the model
model = LinearRegression()
model.fit(X_full, y_full)

# Save the model
joblib.dump(model, 'data/models/linear_regression_model_prod.joblib')
