import argparse
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from pathlib import Path
import yaml


def parser_args():
    parser = argparse.ArgumentParser(description='Train full MLP model')
    parser.add_argument('--input_dir', '-id', type=str, required=True, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, required=True, help='path to save model and results')
    parser.add_argument('--model_name', '-mn', type=str, required=True, help='name of the model to be saved')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    with open(args.params, 'r', encoding='utf-8') as f:
        params_all = yaml.safe_load(f)
    params = params_all['mlp']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_name = args.model_name

    # 读取完整的数据集
    X_full = pd.read_csv(input_dir / 'X_full.csv')
    y_full = pd.read_csv(input_dir / 'y_full.csv')

    # 标准化数据
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)

    # 构建MLP模型
    model = Sequential()
    model.add(Dense(params['hidden_units'], input_dim=X_full.shape[1], activation='relu'))
    model.add(Dense(params['hidden_units'], activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')

    # 训练模型
    model.fit(X_full_scaled, y_full, epochs=params['epochs'], batch_size=params['batch_size'], verbose=1)

    # 保存模型和标准化参数
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / f'{model_name}_prod.h5')
    joblib.dump(scaler, output_dir / f'{model_name}_prod_scaler.pkl')

    # 保存模型系数（权重）
    weights = model.get_weights()
    for i, weight in enumerate(weights):
        if i % 2 == 0:
            weight_df = pd.DataFrame(weight)
            weight_df.to_csv(output_dir / f'{model_name}_layer_{i//2}_weights.csv', index=False)
