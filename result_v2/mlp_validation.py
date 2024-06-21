import argparse
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from pathlib import Path
import yaml

def parser_args():
    parser = argparse.ArgumentParser(description='Validate MLP model')
    parser.add_argument('--input_dir', '-id', type=str, required=True, help='path to input data directory')
    parser.add_argument('--model_path', '-im', type=str, required=True, help='path to the trained model')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False, help='file with dvc stage params')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args()
    with open(args.params, 'r', encoding='utf-8') as f:
        params_all = yaml.safe_load(f)
    params = params_all['mlp']

    input_dir = Path(args.input_dir)
    model_path = Path(args.model_path)

    X_test = pd.read_csv(input_dir / 'X_test.csv')
    y_test = pd.read_csv(input_dir / 'y_test.csv')

    scaler = joblib.load(model_path.parent / f'{model_path.stem}_scaler.pkl')
    X_test_scaled = scaler.transform(X_test)

    model = load_model(model_path)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R-squared (R²): {r2}')
