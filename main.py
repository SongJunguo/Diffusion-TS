import os  # 导入os模块，用于处理文件路径和目录
import torch  # 导入PyTorch库，用于深度学习
import argparse  # 导入argparse模块，用于解析命令行参数
import numpy as np  # 导入numpy库，用于数值计算

from engine.logger import Logger  # 导入自定义的Logger类，用于记录日志
from engine.solver import Trainer  # 导入自定义的Trainer类，用于训练模型
from Data.build_dataloader import build_dataloader, build_dataloader_cond  # 导入自定义的dataloader构建函数
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one  # 导入自定义的反归一化函数
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config  # 导入自定义的配置加载和随机种子设置等函数


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PyTorch Training Script')  # 创建一个参数解析器
    parser.add_argument('--name', type=str, default=None)  # 添加一个参数--name，用于指定实验名称
    parser.add_argument('--config_file', type=str, default=None,
                        help='path of config file')  # 添加一个参数--config_file，用于指定配置文件路径
    parser.add_argument('--output', type=str, default='OUTPUT',
                        help='directory to save the results')  # 添加一个参数--output，用于指定输出结果的保存目录
    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard for logging')  # 添加一个参数--tensorboard，用于指定是否使用tensorboard记录日志

    # args for random
    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')  # 添加一个参数--cudnn_deterministic，用于指定是否设置cudnn为确定性模式
    parser.add_argument('--seed', type=int, default=12345,
                        help='seed for initializing training.')  # 添加一个参数--seed，用于指定随机种子
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')  # 添加一个参数--gpu，用于指定使用的GPU ID

    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')  # 添加一个参数--train，用于指定是否进行训练
    parser.add_argument('--sample', type=int, default=0,
                        choices=[0, 1], help='Condition or Uncondition.')  # 添加一个参数--sample，用于指定是否进行条件采样（1）或者无条件采样（0）
    parser.add_argument('--mode', type=str, default='infill',
                        help='infill or predict.')  # 添加一个参数--mode，用于指定任务类型（填充或预测）
    parser.add_argument('--milestone', type=int, default=10)  # 添加一个参数--milestone，用于指定加载的模型检查点

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')  # 添加一个参数--missing_ratio，用于指定缺失值比例
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')  # 添加一个参数--pred_len，用于指定预测长度

    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)  # 添加一个参数--opts，用于从命令行修改配置

    args = parser.parse_args()  # 解析命令行参数
    args.save_dir = os.path.join(args.output, f'{args.name}')  # 构建保存目录路径

    return args  # 返回解析后的参数

def main():
    """主函数"""
    args = parse_args()  # 解析命令行参数

    if args.seed is not None:  # 如果指定了随机种子
        seed_everything(args.seed)  # 设置随机种子

    if args.gpu is not None:  # 如果指定了GPU
        torch.cuda.set_device(args.gpu)  # 设置使用的GPU

    config = load_yaml_config(args.config_file)  # 加载YAML配置文件
    config = merge_opts_to_config(config, args.opts)  # 合并命令行参数到配置文件

    logger = Logger(args)  # 创建一个Logger对象，用于记录日志
    logger.save_config(config)  # 保存配置文件到日志

    model = instantiate_from_config(config['model']).cuda()  # 从配置实例化模型并将其移动到GPU
    if args.sample == 1 and args.mode in ['infill', 'predict']:  # 如果进行条件采样且任务是填充或预测
        test_dataloader_info = build_dataloader_cond(config, args)  # 构建用于测试的条件数据加载器
    dataloader_info = build_dataloader(config, args)  # 构建数据加载器
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)  # 创建一个Trainer对象，用于训练或采样

    if args.train:  # 如果指定了训练模式
        trainer.train()  # 开始训练
    elif args.sample == 1 and args.mode in ['infill', 'predict']:  # 如果进行条件采样且任务是填充或预测
        trainer.load(args.milestone)  # 加载指定milestone的模型
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']  # 获取测试数据集和dataloader
        coef = config['dataloader']['test_dataset']['coefficient']  # 获取系数
        stepsize = config['dataloader']['test_dataset']['step_size'] # 获取步长
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps'] # 获取采样步数
        samples, *_ = trainer.restore(dataloader, [dataset.window, dataset.var_num], coef, stepsize, sampling_steps)  # 从dataloader恢复数据
        if dataset.auto_norm: # 如果数据集进行了自动标准化
            samples = unnormalize_to_zero_to_one(samples) # 将数据反归一化到0-1之间
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape) # (可选) 如果dataset使用scaler，则反scaler
        np.save(os.path.join(args.save_dir, f'ddpm_{args.mode}_{args.name}.npy'), samples)  # 保存采样结果到文件
    else:  # 如果不进行训练且不进行条件采样
        trainer.load(args.milestone)  # 加载指定milestone的模型
        dataset = dataloader_info['dataset']  # 获取数据集
        samples = trainer.sample(num=len(dataset), size_every=2001, shape=[dataset.window, dataset.var_num]) # 从模型中采样数据
        if dataset.auto_norm: # 如果数据集进行了自动标准化
            samples = unnormalize_to_zero_to_one(samples) # 将数据反归一化到0-1之间
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape) # (可选) 如果dataset使用scaler，则反scaler
        np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.-npy'), samples)  # 保存采样结果到文件

if __name__ == '__main__':
    main()  # 如果是主程序，则运行main函数