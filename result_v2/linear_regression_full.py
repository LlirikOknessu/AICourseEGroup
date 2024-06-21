import argparse
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path
import yaml


def parser_args():
    parser = argparse.ArgumentParser(description='Train full linear regression model')
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
    params = params_all['linear_regression']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_name = args.model_name

    # 读取完整的数据集
    X_full = pd.read_csv(input_dir / 'X_full.csv')
    y_full = pd.read_csv(input_dir / 'y_full.csv')

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X_full, y_full)

    # 保存模型和系数
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / f'{model_name}_prod.joblib')

    # 将model.coef_扁平化并保存
    coef_df = pd.DataFrame(model.coef_.reshape(-1, 1), index=X_full.columns, columns=['Coefficient'])
    coef_df.to_csv(output_dir / f'{model_name}_prod_coefficients.csv', index=True)
