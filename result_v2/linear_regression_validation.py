import argparse
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from pathlib import Path
import yaml


def parser_args():
    parser = argparse.ArgumentParser(description='Validate linear regression model')
    parser.add_argument('--input_dir', '-id', type=str, required=True, help='path to input data directory')
    parser.add_argument('--model_path', '-im', type=str, required=True, help='path to the trained model')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False, help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    with open(args.params, 'r', encoding='utf-8') as f:
        params_all = yaml.safe_load(f)
    params = params_all['linear_regression']

    input_dir = Path(args.input_dir)
    model_path = Path(args.model_path)

    X_val = pd.read_csv(input_dir / 'X_val.csv')
    y_val = pd.read_csv(input_dir / 'y_val.csv')

    model = joblib.load(model_path)
    y_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R-squared (R²): {r2}')
