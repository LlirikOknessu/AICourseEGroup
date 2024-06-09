import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Load prepared data
X_train = pd.read_csv('data/prepared/X_train.csv')
y_train = pd.read_csv('data/prepared/y_train.csv')

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'data/models/linear_regression_model.joblib')
