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
                        required=False, help='path to save the model')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['mlp']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    X_train = pd.read_csv(input_dir / 'X_train.csv')
    y_train = pd.read_csv(input_dir / 'y_train.csv')

    input_shape = (X_train.shape[1],)
    model = Sequential()
    model.add(Dense(64, input_shape=input_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'])
    model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=1, validation_split=0.2)

    output_dir.mkdir(exist_ok=True, parents=True)
    model.save(output_dir / f"{params['model_name']}.h5")
