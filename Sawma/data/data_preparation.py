import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

def encode_labels(df: pd.DataFrame):
    label_encoder = LabelEncoder()
    df['Country_Label'] = label_encoder.fit_transform(df['Country'])
    df['Status'] = label_encoder.fit_transform(df['Status'])
    return df

def fill_missing_values(df: pd.DataFrame):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    mean_values = df[numeric_columns].mean()
    df[numeric_columns] = df[numeric_columns].fillna(mean_values)
    return df

if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['data_preparation']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    for data_file in input_dir.glob('*.csv'):
        df = pd.read_csv(data_file)
        df = fill_missing_values(df)
        df = encode_labels(df)
        df.drop(columns=['Country'], inplace=True)

        y = df['Life expectancy ']
        X = df.drop(columns=['Life expectancy '])

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=params.get('train_test_ratio', 0.8),
                                                            random_state=params.get('random_state', 1))
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          train_size=params.get('train_val_ratio', 0.8),
                                                          random_state=params.get('random_state', 1))

        X_full_name = output_dir / 'X_full.csv'
        y_full_name = output_dir / 'y_full.csv'
        X_train_name = output_dir / 'X_train.csv'
        y_train_name = output_dir / 'y_train.csv'
        X_test_name = output_dir / 'X_test.csv'
        y_test_name = output_dir / 'y_test.csv'
        X_val_name = output_dir / 'X_val.csv'
        y_val_name = output_dir / 'y_val.csv'

        X.to_csv(X_full_name, index=False)
        y.to_csv(y_full_name, index=False)
        X_train.to_csv(X_train_name, index=False)
        y_train.to_csv(y_train_name, index=False)
        X_test.to_csv(X_test_name, index=False)
        y_test.to_csv(y_test_name, index=False)
        X_val.to_csv(X_val_name, index=False)
        y_val.to_csv(y_val_name, index=False)
