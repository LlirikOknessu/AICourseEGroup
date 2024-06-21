import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
import joblib
import datetime


def parser_args():
    parser = argparse.ArgumentParser(description='Train MLP model')
    parser.add_argument('--input_dir', '-id', type=str, required=True, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, required=True, help='path to save model and results')
    parser.add_argument('--model_name', '-mn', type=str, required=True, help='name of the model to be saved')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


def build_mlp_model(input_shape, params):
    model = Sequential()
    model.add(Dense(params['hidden_units1'], input_shape=input_shape, activation=params['activation'],
                    kernel_regularizer=l2(params['l2_reg'])))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(params['hidden_units2'], activation=params['activation'], kernel_regularizer=l2(params['l2_reg'])))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(params['hidden_units3'], activation=params['activation'], kernel_regularizer=l2(params['l2_reg'])))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss=params['loss'])
    return model


if __name__ == '__main__':
    args = parser_args()
    with open(args.params, 'r', encoding='utf-8') as f:  # 指定UTF-8编码
        params_all = yaml.safe_load(f)
    params = params_all['mlp']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_name = args.model_name

    X_train = pd.read_csv(input_dir / 'X_train.csv')
    y_train = pd.read_csv(input_dir / 'y_train.csv')
    X_val = pd.read_csv(input_dir / 'X_val.csv')
    y_val = pd.read_csv(input_dir / 'X_val.csv')
    X_full = pd.read_csv(input_dir / 'X_full.csv')
    y_full = pd.read_csv(input_dir / 'y_full.csv')

    # Scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_full_scaled = scaler.fit_transform(X_full)  # 使用完整数据重新拟合标准化

    model = build_mlp_model((X_train_scaled.shape[1],), params)

    # Adding TensorBoard callback
    log_dir = output_dir / "logs/fit/" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=str(log_dir), histogram_freq=1)

    model.fit(X_train_scaled, y_train, validation_data=(X_train_scaled, y_train),
              epochs=params['epochs'], batch_size=params['batch_size'],
              callbacks=[ tensorboard_callback])

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / f'{model_name}.h5')
    joblib.dump(scaler, output_dir / f'{model_name}_scaler.pkl')

    # 加入Prod版本
    prod_model = build_mlp_model((X_full_scaled.shape[1],), params)
    prod_model.fit(X_full_scaled, y_full, epochs=params['epochs'], batch_size=params['batch_size'],
                   callbacks=[ tensorboard_callback])

    prod_model.save(output_dir / f'{model_name}_prod.h5')
    joblib.dump(scaler, output_dir / f'{model_name}_prod_scaler.pkl')
