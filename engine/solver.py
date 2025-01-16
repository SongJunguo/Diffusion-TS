import os
import sys
import time
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Utils.metric_utils import calculate_metrics

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        # 新增：训练模式开关，'steps' 或 'epochs'
        self.train_mode = config['solver'].get('train_mode', 'steps')
        # 新增：如果是按epochs训练，需要指定epochs数量
        self.train_num_epochs = config['solver'].get('num_epochs', 50)

        if self.train_mode == 'epochs':
            self.train_num_steps = self.train_num_epochs * len(dataloader['dataloader'])
            self.save_cycle = len(dataloader['dataloader']) #如果是epoch训练，就每个epoch保存checkpoint
        else:
            self.train_num_steps = config['solver']['max_epochs']  # config['solver']['max_epochs']中原有的含义是最大步数
            self.save_cycle = config['solver']['save_cycle']# 按照iter迭代训练，按照设定的iter间隔保存checkpoint
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']

        self.dl = cycle(dataloader['dataloader'])
        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger

        # self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        tmp_save_dir = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        self.results_folder = Path(os.path.join(Path(config['save_dir']), tmp_save_dir))  # 构建保存目录路径
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 10

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info(
                'Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        # torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))
        # 修改这里，先将self.results_folder转换为Path对象
        save_path = Path(self.results_folder) / f'checkpoint-{milestone}.pt'
        torch.save(data, str(save_path))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    def train(self):
        device = self.device
        # step = 0  # 删除此行，因为self.step已经在load函数中设置
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        # with tqdm(initial=step, total=self.train_num_steps) as pbar: # 修改此行
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            # while step < self.train_num_steps: # 修改此行
            while self.step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    loss = self.model(data, target=data)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                # step += 1 # 删除此行，因为self.step已经更新
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                        # self.logger.log_info('saved in {}'.format(str(self.results_folder / f'checkpoint-{self.milestone}.pt')))

                    if self.logger is not None and self.step % self.log_frequency == 0:
                        # info = '{}: train'.format(self.args.name)
                        # info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
                        # info += ' ||'
                        # info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
                        # info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
                        # info += ' | Total Loss: {:.6f}'.format(total_loss)
                        # self.logger.log_info(info)
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)
                        # 将 loss 信息直接写入 log 文件
                        self.logger.log_info(f'Step: {self.step}, Loss: {total_loss:.6f}')

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))


    def sample(self, num, size_every, shape=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):
            sample = self.ema.ema_model.generate_mts(batch_size=size_every)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:  # 如果提供了日志记录器
            tic = time.time()  # 记录开始时间
            self.logger.log_info('开始数据恢复...')  # 记录日志

        # 设置模型的参数
        model_kwargs = {}  # 用于传递给模型的额外参数
        model_kwargs['coef'] = coef  # 设置系数，用于控制恢复过程中的一些参数
        model_kwargs['learning_rate'] = stepsize  # 设置学习率

        # 初始化用于存储样本、真实数据和掩码的空数组
        samples = np.empty([0, shape[0], shape[1]])  # 用于存储生成的样本
        reals = np.empty([0, shape[0], shape[1]])  # 用于存储原始数据
        masks = np.empty([0, shape[0], shape[1]])  # 用于存储掩码

        # 遍历数据加载器中的每个批次
        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)  # 将数据和掩码移到指定设备上

            # 根据sampling_steps的不同选择不同的恢复方法
            if sampling_steps == self.model.num_timesteps:  # 如果使用完整的扩散步骤
                # 使用完整的采样过程进行数据恢复
                sample = self.ema.ema_model.sample_infill(shape=x.shape, target=x * t_m, partial_mask=t_m,
                                                          model_kwargs=model_kwargs)
            else:  # 否则使用快速采样
                # 使用快速采样过程进行数据恢复
                sample = self.ema.ema_model.fast_sample_infill(shape=x.shape, target=x * t_m, partial_mask=t_m,
                                                               model_kwargs=model_kwargs,
                                                               sampling_timesteps=sampling_steps)

            # 将恢复后的样本、原始数据和掩码添加到相应的数组中
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])  # 堆叠样本数据
            reals = np.row_stack([reals, x.detach().cpu().numpy()])  # 堆叠原始数据
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])  # 堆叠掩码数据

        if self.logger is not None:  # 如果提供了日志记录器
            self.logger.log_info('数据恢复完成，耗时: {:.2f}秒'.format(time.time() - tic))  # 记录完成时间和耗时

        # 返回恢复后的样本、原始数据和掩码
        return samples, reals, masks
        # 注释掉的代码行表明原本可能只返回样本数据
        # return samples

# 下面的代码是原来的，没有增加epoch和iter切换功能
''' 
import os
import sys
import time
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader'])
        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    def train(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    loss = self.model(data, target=data)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                        # self.logger.log_info('saved in {}'.format(str(self.results_folder / f'checkpoint-{self.milestone}.pt')))
                    
                    if self.logger is not None and self.step % self.log_frequency == 0:
                        # info = '{}: train'.format(self.args.name)
                        # info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
                        # info += ' ||'
                        # info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
                        # info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
                        # info += ' | Total Loss: {:.6f}'.format(total_loss)
                        # self.logger.log_info(info)
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def sample(self, num, size_every, shape=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):
            sample = self.ema.ema_model.generate_mts(batch_size=size_every)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:  # 如果提供了日志记录器
            tic = time.time()  # 记录开始时间
            self.logger.log_info('开始数据恢复...')  # 记录日志

        # 设置模型的参数
        model_kwargs = {}  # 用于传递给模型的额外参数
        model_kwargs['coef'] = coef  # 设置系数，用于控制恢复过程中的一些参数
        model_kwargs['learning_rate'] = stepsize  # 设置学习率

        # 初始化用于存储样本、真实数据和掩码的空数组
        samples = np.empty([0, shape[0], shape[1]])  # 用于存储生成的样本
        reals = np.empty([0, shape[0], shape[1]])  # 用于存储原始数据
        masks = np.empty([0, shape[0], shape[1]])  # 用于存储掩码

        # 遍历数据加载器中的每个批次
        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)  # 将数据和掩码移到指定设备上

            # 根据sampling_steps的不同选择不同的恢复方法
            if sampling_steps == self.model.num_timesteps:  # 如果使用完整的扩散步骤
                # 使用完整的采样过程进行数据恢复
                sample = self.ema.ema_model.sample_infill(shape=x.shape, target=x * t_m, partial_mask=t_m,
                                                          model_kwargs=model_kwargs)
            else:  # 否则使用快速采样
                # 使用快速采样过程进行数据恢复
                sample = self.ema.ema_model.fast_sample_infill(shape=x.shape, target=x * t_m, partial_mask=t_m,
                                                               model_kwargs=model_kwargs,
                                                               sampling_timesteps=sampling_steps)

            # 将恢复后的样本、原始数据和掩码添加到相应的数组中
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])  # 堆叠样本数据
            reals = np.row_stack([reals, x.detach().cpu().numpy()])  # 堆叠原始数据
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])  # 堆叠掩码数据

        if self.logger is not None:  # 如果提供了日志记录器
            self.logger.log_info('数据恢复完成，耗时: {:.2f}秒'.format(time.time() - tic))  # 记录完成时间和耗时

        # 返回恢复后的样本、原始数据和掩码
        return samples, reals, masks
        # 注释掉的代码行表明原本可能只返回样本数据
        # return samples
'''
