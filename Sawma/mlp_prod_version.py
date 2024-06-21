import pandas as pd
import argparse
from pathlib import Path
import yaml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save the production model')
    parser.add_argument('--model_name', '-mn', type=str, required=True,
                        help='name of the model file')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with model parameters')
    return parser.parse_args()


def load_params(params_file):
    with open(params_file, 'r') as f:
        return yaml.safe_load(f)['mlp']


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_name = args.model_name
    params = load_params(args.params)

    X_full_name = input_dir / 'X_full.csv'
    y_full_name = input_dir / 'y_full.csv'

    X_full = pd.read_csv(X_full_name)
    y_full = pd.read_csv(y_full_name)

    input_shape = (X_full.shape[1],)
    print(f'Feature shape: {input_shape}')

    model = Sequential()
    model.add(Dense(params['64'], input_shape=input_shape, activation='relu'))
    model.add(Dense(params['32'], activation='relu'))
    model.add(Dense(params['16'], activation='relu'))
    model.add(Dense(params['8'], activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'])

    model.fit(X_full, y_full, epochs=params['300'], batch_size=params['8'], verbose=1, validation_split=0.2)

    production_model_path = output_dir / f"{model_name}_prod.h5"
    model.save(production_model_path)
    print(f"Production model saved to {production_model_path}")
