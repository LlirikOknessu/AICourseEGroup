import pandas as pd
import argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from joblib import dump

# Constants for linear models
LINEAR_MODELS_MAPPER = {'LinearRegression': LinearRegression}


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save model artifacts')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='name of the model to be trained')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_dir / (args.model_name + '.csv')
    output_model_joblib_path = output_dir / (args.model_name + '.joblib')

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    # Load data
    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)  # Ensure y_train is a Series
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)  # Ensure y_test is a Series

    # Train the model
    reg = LINEAR_MODELS_MAPPER[args.model_name]().fit(X_train, y_train)

    # Evaluate the model
    y_mean = y_test.mean()
    y_pred_baseline = np.full_like(y_test, fill_value=y_mean)  # Baseline prediction

    predicted_values = reg.predict(X_test)

    print("Model performance metrics:")
    print("R-squared:", reg.score(X_test, y_test))
    print("Mean actual salary: ", y_mean)
    print("Baseline MAE: ", mean_absolute_error(y_test, y_pred_baseline))
    print("Model MAE: ", mean_absolute_error(y_test, predicted_values))

    # Extract model coefficients
    intercept = reg.intercept_.astype(float)
    coefficients = reg.coef_.astype(float)

    intercept_series = pd.Series(intercept, name='intercept')
    coefficients_series = pd.Series(coefficients, name='coefficients')

    print("Intercept:", intercept_series)
    print("Coefficients:", coefficients_series)

    # Save model coefficients and intercept
    model_coefficients = pd.DataFrame([coefficients_series, intercept_series])
    model_coefficients.to_csv(output_model_path, index=False)

    # Save trained model using joblib
    dump(reg, output_model_joblib_path)
