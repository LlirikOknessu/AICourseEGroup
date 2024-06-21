import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


data = pd.read_csv('data/raw/insurance.csv')

print(data.head())

print(data.describe())

numeric_cols = ['age', 'bmi', 'children']
categorical_cols = ['sex', 'smoker', 'region']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values by replacing them with the mean
    ('scaler', StandardScaler())  # Standardize the data
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values by replacing them with the most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-Hot encode the data
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X = data.drop('charges', axis=1)
y = data['charges']

X_preprocessed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

y_train_log = np.log(y_train)
y_test_log = np.log(y_test)

model = LinearRegression()

model.fit(X_train, y_train_log)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test_log, y_pred)
rmse = mean_squared_error(y_test_log, y_pred, squared=False)
r2 = r2_score(y_test_log, y_pred)

print(mse, rmse, r2)

y_pred_original = np.exp(y_pred)

# 基于中位数的基准预测
median_pred = y_test.median()  # 获取中位数值

# 为每个测试样本生成基准预测
median_preds = [median_pred for _ in range(len(y_test))]

median_preds_log = np.log(median_preds)

mse_baseline = mean_squared_error(y_test_log, median_preds_log)
rmse_baseline = mean_squared_error(y_test_log, median_preds_log, squared=False)
r2_baseline = r2_score(y_test_log, median_preds_log)

print(mse_baseline, rmse_baseline, r2_baseline)

df_comparison = pd.DataFrame(
    {
        'predicted': y_pred_original,
        'real': y_test,
        'median': median_preds,
    }
)

# 重制索引
df_comparison = df_comparison.reset_index(drop=True)

plt.figure(figsize=(10,6))
plt.plot(df_comparison.index, df_comparison['predicted'], label='predict', marker='o')
plt.plot(df_comparison.index, df_comparison['real'], label='real', marker='x')
plt.plot(df_comparison.index, df_comparison['median'], label='median', linestyle='-.')

plt.title('predict vs real')
plt.xlabel('index')
plt.ylabel('charges')
plt.legend()
plt.show()
