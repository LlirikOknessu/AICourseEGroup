import pandas as pd
import argparse
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import numpy as np


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--input_model', '-im', type=str, default='data/models/mlp_model.h5',
                        required=False, help='path to load the model')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    input_model = Path(args.input_model)

    X_val = pd.read_csv(input_dir / 'X_val.csv')
    y_val = pd.read_csv(input_dir / 'y_val.csv')

    model = load_model(input_model)

    predictions = np.squeeze(model.predict(X_val))
    y_mean = y_val.mean()
    y_pred_baseline = [y_mean] * len(y_val)

    print(f"Mean apt salary: {y_mean:.4f}")
    print(f"Baseline MAE: {mean_absolute_error(y_val, y_pred_baseline):.4f}")
    print(f"Model MAE: {mean_absolute_error(y_val, predictions):.4f}")
