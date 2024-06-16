import pandas as pd
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error
from joblib import load


def parser_args_for_sac():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--input_model', '-im', type=str, default='data/models/model.joblib',
                        required=False, help='path to load the model')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parser_args_for_sac()

    # Define input directory and model path
    input_dir = Path(args.input_dir)
    input_model = Path(args.input_model)

    # Define paths to validation dataset
    X_val_name = input_dir / 'X_val.csv'
    y_val_name = input_dir / 'y_val.csv'

    # Load the validation dataset
    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)

    # Load the trained model
    reg = load(input_model)

    # Predict on the validation set
    predicted_values = np.squeeze(reg.predict(X_val))

    # Calculate baseline predictions based on normal distribution
    y_mean = y_val.mean()
    y_std = y_val.std()
    y_pred_normal = np.random.normal(loc=y_mean, scale=y_std, size=len(y_val))

    # Calculate baseline predictions based on uniform distribution
    y_min = y_val.min()
    y_max = y_val.max()
    y_pred_uniform = np.random.uniform(low=y_min, high=y_max, size=len(y_val))

    # Print model performance metrics
    print(f"Model R^2 Score: {reg.score(X_val, y_val):.4f}")
    print(f"Mean target value: {y_mean:.4f}")
    print(f"Normal Distribution Baseline MAE: {mean_absolute_error(y_val, y_pred_normal):.4f}")
    print(f"Uniform Distribution Baseline MAE: {mean_absolute_error(y_val, y_pred_uniform):.4f}")
    print(f"Model MAE: {mean_absolute_error(y_val, predicted_values):.4f}")
