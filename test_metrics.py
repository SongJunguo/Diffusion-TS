import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics_for_each_variable(samples, reals, masks):
    """
    分别为每个变量计算 MSE 和 MAE。

    Args:
        samples: 预测值，numpy 数组，形状为 (样本数, 时间步, 变量数)。
        reals: 真实值，numpy 数组，形状为 (样本数, 时间步, 变量数)。
        masks: 掩码，numpy 数组，形状为 (样本数, 时间步, 变量数)，其中 0 表示需要计算指标的位置。

    Returns:
        一个字典，键为 "MSE_{变量索引}" 和 "MAE_{变量索引}"，值为对应变量的 MSE 和 MAE。
    """
    num_variables = samples.shape[-1]
    metrics = {}

    for i in range(num_variables):
        y_true = reals[:, :, i][masks[:, :, i] == 0]
        y_pred = samples[:, :, i][masks[:, :, i] == 0]

        if y_true.size > 0:  # 确保有需要计算指标的数据
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)

            metrics[f"MSE_{i}"] = mse
            metrics[f"MAE_{i}"] = mae
        else:
            print(f"Warning: No valid data points for variable {i} to calculate metrics.")

    return metrics

def plot_results(samples, reals, masks, title_prefix, save_dir, sample_interval, show_plots=False):
    """
    为每个变量绘制预测结果图，按采样间隔采样一部分数据进行绘图。

    Args:
        samples: 预测值，numpy 数组，形状为 (样本数, 时间步, 变量数)。
        reals: 真实值，numpy 数组，形状为 (样本数, 时间步, 变量数)。
        masks: 掩码，numpy 数组，形状为 (样本数, 时间步, 变量数)，其中 0 表示需要计算指标的位置。
        title_prefix: 图表标题的前缀。
        save_dir: 保存图表的目录。
        sample_interval: 采样间隔，每隔多少个样本绘制一个图表。
        show_plots: 是否显示图表 (用于调试)。
    """
    num_variables = samples.shape[-1]
    num_samples = samples.shape[0]

    for i in range(num_variables):
        plt.figure(figsize=(19.20, 10.80), dpi=100)  # 设置 4K 分辨率 (3840x2160)

        # 绘制 reals vs. samples
        # plt.subplot(2, 1, 2) # 不需要这个了，因为只剩一个图了
        for j in range(0, num_samples, sample_interval):
            plt.plot(reals[j, :, i], color='blue', alpha=0.5)
            plt.plot(samples[j, :, i], color='red', alpha=0.5)

        # 添加图例
        plt.gca().plot([], [], color='blue', label='Reals')
        plt.gca().plot([], [], color='red', label='Samples')
        plt.legend()

        plt.title(f"{title_prefix} - Variable {i} - Reals vs. Samples")
        plt.xlabel("Time Step")
        plt.ylabel("Value")

        plt.tight_layout()

        # 保存图片
        save_path = os.path.join(save_dir, f"{title_prefix}_variable_{i}.png")
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")

        # 可选：显示图片
        if show_plots:
            plt.show()

        plt.close()
def main():
    parser = argparse.ArgumentParser(description='Calculate MSE and MAE for each variable from saved numpy files.')
    parser.add_argument('--load_dir', type=str, default='/workspace/Diffusion-TS/OUTPUT/trajectory_10_20_big_model/2025-0103-2243',
                        help='directory to load the saved numpy files (default: /workspace/Diffusion-TS/OUTPUT/trajectory_10_20/2025-0101-)')
    parser.add_argument('--mode', type=str, default='predict',
                        help='mode used during saving (e.g., infill, predict) (default: predict)')
    parser.add_argument('--name', type=str, default='trajectory_10_20_big_model',
                        help='name used during saving (default: trajectory_10_20_big_model)')
    parser.add_argument('--sample_interval', type=int, default=200,
                        help='sample interval for plotting (default: 1000)')
    parser.add_argument('--show_plots', action='store_true',
                        help='show plots (default: False)')

    args = parser.parse_args()

    # 确保 'samples' 文件夹存在
    samples_dir = os.path.join(args.load_dir, 'samples')
    # 构建原始数据的文件路径
    samples_file = os.path.join(samples_dir, f'samples_{args.mode}_{args.name}.npy')
    reals_file = os.path.join(samples_dir, f'reals_{args.mode}_{args.name}.npy')
    masks_file = os.path.join(samples_dir, f'masks_{args.mode}_{args.name}.npy')

    # 构建反归一化数据的文件路径
    inverse_samples_file = os.path.join(samples_dir, f'inverse_samples_{args.mode}_{args.name}.npy')
    inverse_reals_file = os.path.join(samples_dir, f'inverse_reals_{args.mode}_{args.name}.npy')

    # 创建用于保存图表和数据的新目录
    plots_dir = os.path.join(args.load_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错

    # 创建文本文件来保存打印输出
    with open(os.path.join(plots_dir, 'metrics.txt'), 'w') as f:
        def print_and_log(*args):
            print(*args)
            print(*args, file=f)

        # 检查原始数据文件是否存在
        if not all(os.path.exists(f) for f in [samples_file, reals_file, masks_file]):
            print_and_log(f"Error: One or more required files for original data not found in {args.load_dir}")
            print_and_log(f"Tried loading:")
            print_and_log(f"  - {samples_file}")
            print_and_log(f"  - {reals_file}")
            print_and_log(f"  - {masks_file}")
            return

        # 加载原始数据
        samples = np.load(samples_file)
        reals = np.load(reals_file)
        masks = np.load(masks_file)

        # 计算原始数据的指标
        print_and_log("Calculating metrics for original data...")
        metrics_original = calculate_metrics_for_each_variable(samples, reals, masks)
        print_and_log(f"Metrics (original):")
        for key, value in metrics_original.items():
            print_and_log(f"{key}: {value}")

        # 绘制原始数据的图表
        plot_results(samples, reals, masks, "Original", plots_dir, args.sample_interval, args.show_plots)

        # 检查反归一化数据文件是否存在
        if all(os.path.exists(f) for f in [inverse_samples_file, inverse_reals_file, masks_file]):
            print_and_log("\nCalculating metrics for inverse transformed data...")
            # 加载反归一化数据
            inverse_samples = np.load(inverse_samples_file)
            inverse_reals = np.load(inverse_reals_file)

            # 计算反归一化数据的指标
            metrics_inverse = calculate_metrics_for_each_variable(inverse_samples, inverse_reals, masks)
            print_and_log(f"Metrics (inverse):")
            for key, value in metrics_inverse.items():
                print_and_log(f"{key}: {value}")

            # 绘制反归一化数据的图表
            plot_results(inverse_samples, inverse_reals, masks, "Inverse", plots_dir, args.sample_interval, args.show_plots)
        else:
            print_and_log("\nInverse transformed data files not found. Skipping plots for inverse data.")

if __name__ == '__main__':
    main()
