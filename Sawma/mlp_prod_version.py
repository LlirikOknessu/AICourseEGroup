import pandas as pd
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save the production model')
    parser.add_argument('--model_name', '-mn', type=str, required=True,
                        help='name of the model file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_name = args.model_name

    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    model_path = output_dir / f"{model_name}.h5"
    model = load_model(model_path)

    predicted_values = np.squeeze(model.predict(X_test))

    y_mean = y_test.mean()
    y_pred_baseline = [y_mean] * len(y_test)

    print(f"Mean target value: {y_mean:.4f}")
    print(f"Baseline MAE: {mean_absolute_error(y_test, y_pred_baseline):.4f}")
    print(f"Model MAE on test data: {mean_absolute_error(y_test, predicted_values):.4f}")
    production_model_path = output_dir / f"{model_name}_prod.h5"
    model.save(production_model_path)
