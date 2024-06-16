import pandas as pd
import argparse
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from joblib import dump

# Define a mapper for linear models
LINEAR_MODELS_MAPPER = {'Ridge': Ridge, 'LinearRegression': LinearRegression}


def parser_args_for_sac():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parser_args_for_sac()

    # Define input and output directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Create output directory if it does not exist
    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_dir / (args.model_name + '_prod.csv')
    output_model_joblib_path = output_dir / (args.model_name + '_prod.joblib')

    # Define paths to full dataset
    X_full_name = input_dir / 'X_full.csv'
    y_full_name = input_dir / 'y_full.csv'

    # Load the full dataset
    X_full = pd.read_csv(X_full_name)
    y_full = pd.read_csv(y_full_name)

    # Train the model on the full dataset
    reg = LINEAR_MODELS_MAPPER.get(args.model_name, LinearRegression)().fit(X_full, y_full)

    # Extract intercept and coefficients if they exist
    if hasattr(reg, 'intercept_'):
        intercept = reg.intercept_.astype(float)
        intercept = pd.Series(intercept, name='intercept')
        print("intercept:", intercept)
    else:
        intercept = pd.Series([], name='intercept')
        print("No intercept found for the model")

    if hasattr(reg, 'coef_'):
        coefficients = reg.coef_.astype(float)
        coefficients = pd.Series(coefficients[0], name='coefficients')
        print("list of coefficients:", coefficients)
    else:
        coefficients = pd.Series([], name='coefficients')
        print("No coefficients found for the model")

    # Save intercept and coefficients
    out_model = pd.DataFrame([coefficients, intercept])
    out_model.to_csv(output_model_path, index=False)

    # Save the trained model
    dump(reg, output_model_joblib_path)
