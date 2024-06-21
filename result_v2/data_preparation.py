import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.ffill(inplace=True)  # Forward fill for missing values
    df.bfill(inplace=True)  # Backward fill for any remaining missing values
    # Convert categorical variables to codes (0, 1, 2, ...)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.Categorical(df[col]).codes
    return df


if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r', encoding='utf-8') as f:
        params_all = yaml.safe_load(f)
    params = params_all['data_preparation']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    # Read the dataset
    data_file = input_dir / 'insurance.csv'
    full_data = pd.read_csv(data_file)

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


    X = full_data.drop('charges', axis=1)
    y = full_data['charges']

    X_preprocessed = preprocessor.fit_transform(X)

    column_names = preprocessor.get_feature_names_out(input_features=X.columns)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y,
                                                        train_size=params.get('train_test_ratio', 0.8),
                                                        random_state=params.get('random_state', 42))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      train_size=params.get('train_val_ratio', 0.8),
                                                      random_state=params.get('random_state', 42))
    y_log = np.log(y)
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)                                    
    y_val_log = np.log(y_val)

    X_full_name = output_dir / 'X_full.csv'
    y_full_name = output_dir / 'y_full.csv'
    X_train_name = output_dir / 'X_train.csv'
    y_train_name = output_dir / 'y_train.csv'
    X_test_name = output_dir / 'X_test.csv'
    y_test_name = output_dir / 'y_test.csv'
    X_val_name = output_dir / 'X_val.csv'
    y_val_name = output_dir / 'y_val.csv'

    X_preprocessed = pd.DataFrame(X_preprocessed, columns=column_names)
    y_log = pd.DataFrame(y_log)
    X_train = pd.DataFrame(X_train, columns=column_names)
    y_train_log = pd.DataFrame(y_train_log)
    X_test = pd.DataFrame(X_test, columns=column_names)
    y_test_log = pd.DataFrame(y_test_log)
    X_val = pd.DataFrame(X_val, columns=column_names)
    y_val_log = pd.DataFrame(y_val_log)

    X_preprocessed.to_csv(X_full_name, index=False)
    y_log.to_csv(y_full_name, index=False)
    X_train.to_csv(X_train_name, index=False)
    y_train_log.to_csv(y_train_name, index=False)
    X_test.to_csv(X_test_name, index=False)
    y_test_log.to_csv(y_test_name, index=False)
    X_val.to_csv(X_val_name, index=False)
    y_val_log.to_csv(y_val_name, index=False)
