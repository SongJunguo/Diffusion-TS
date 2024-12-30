import os  # 导入os模块，用于处理文件路径和目录
import torch  # 导入PyTorch库，用于深度学习
import numpy as np  # 导入numpy库，用于数值计算
import pandas as pd  # 导入pandas库，用于数据处理

from scipy import io  # 导入scipy.io模块，用于读取.mat文件
from sklearn.preprocessing import MinMaxScaler  # 导入sklearn的MinMaxScaler类，用于数据归一化
from torch.utils.data import Dataset, ConcatDataset # 导入PyTorch的Dataset类，用于创建自定义数据集
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, \
    unnormalize_to_zero_to_one  # 导入自定义的数据归一化和反归一化函数
from Utils.masking_utils import noise_mask  # 导入自定义的噪声掩码生成函数
from glob import glob
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed

class CustomDataset(Dataset):  # 定义一个继承自torch.utils.data.Dataset的自定义数据集类
    def __init__(  # 定义类的初始化方法
            self,
            name,  # 数据集名称
            data_root,  # 数据文件路径
            window=64,  # 时间窗口大小
            proportion=0.8,  # 训练集比例
            save2npy=True,  # 是否将数据保存为.npy文件
            neg_one_to_one=True,  # 是否将数据归一化到[-1, 1]之间，否则使用最小最大归一化
            seed=123,  # 随机种子
            period='train',  # 数据集类型，'train'或'test'
            output_dir='./OUTPUT',  # 输出目录
            predict_length=None,  # 预测长度, 仅在测试集使用
            missing_ratio=None,  # 缺失值比例, 仅在测试集使用
            style='separate',  # 缺失值的风格，可选'separate', 'mask'等
            distribution='geometric',  # 掩码的分布，可选'geometric', 'random'等
            mean_mask_length=3  # 掩码的平均长度，当distribution为'geometric'时起作用
    ):
        super(CustomDataset, self).__init__()  # 调用父类的初始化方法
        assert period in ['train', 'test'], 'period must be train or test.'  # 断言period必须是'train'或'test'
        if period == 'train':  # 如果是训练集，则断言predict_length和missing_ratio为None
            assert ~(
                        predict_length is not None or missing_ratio is not None), 'Train dataset does not support missing values or predict length'
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio  # 设置数据集名称，预测长度，缺失值比例
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length  # 设置mask的风格，分布，平均长度
        self.rawdata, self.scaler = self.read_data(data_root, self.name)  # 读取数据和scaler
        self.dir = os.path.join(output_dir, 'samples')  # 设置保存数据的目录
        os.makedirs(self.dir, exist_ok=True)  # 创建保存数据的目录，如果目录已存在则不报错

        self.window, self.period = window, period  # 设置时间窗口和数据集类型
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]  # 获取原始数据的长度和变量数
        self.save2npy = save2npy  # 设置是否保存数据
        self.auto_norm = neg_one_to_one  # 设置是否进行[-1,1]归一化

        self.data = self.__normalize(self.rawdata)  # 对原始数据进行归一化
        train, inference = self.__getsamples(self.data, proportion, seed)  # 获取训练集和测试集

        self.samples = train if period == 'train' else inference  # 如果是训练集，则使用训练集数据，否则使用测试集数据
        if period == 'test':  # 如果是测试集
            if missing_ratio is not None:  # 如果missing_ratio不为None，则生成缺失值mask
                self.masking = self.mask_data(seed)  # 使用mask_data函数生成mask
            elif predict_length is not None:  # 如果predict_length不为None, 则生成预测mask
                masks = np.ones(self.samples.shape)  # 生成全1的mask
                masks[:, -predict_length:, :] = 0  # 将末尾predict_length的时间步mask设置为0
                self.masking = masks.astype(bool)  # 将mask类型设置为bool
            else:
                raise NotImplementedError(
                    "Missing ratio or predict length must be set when test.")  # 如果测试集既没有缺失值，也没有预测长度，则报错
        self.sample_num = self.samples.shape[0]  # 设置样本数量

    def __getsamples(self, data, proportion, seed):
        """
        将数据划分成窗口，并分为训练集和测试集。
        """
        sample_num_total = max(self.len - self.window + 1, 0)  # 计算样本总数
        x = np.zeros((sample_num_total, self.window, self.var_num))  # 创建一个存储窗口数据的numpy数组
        for i in range(sample_num_total):  # 遍历所有窗口
            start = i  # 窗口起始位置
            end = i + self.window  # 窗口结束位置
            x[i, :, :] = data[start:end, :]  # 将数据添加到x中

        train_data, test_data = self.divide(x, proportion, seed)  # 将数据划分成训练集和测试集

        if self.save2npy:  # 如果save2npy为True，则保存数据到npy文件
            if 1 - proportion > 0:  # 如果测试集比例大于0，则保存测试集数据
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"),
                        self.unnormalize(test_data))  # 保存反归一化后的测试集数据
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"),
                    self.unnormalize(train_data))  # 保存反归一化后的训练集数据
            if self.auto_norm:  # 如果使用[-1, 1]归一化，则保存归一化后的数据
                if 1 - proportion > 0:  # 如果测试集比例大于0，则保存测试集数据
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"),
                            unnormalize_to_zero_to_one(test_data))  # 保存反归一化后的测试集数据
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"),
                        unnormalize_to_zero_to_one(train_data))  # 保存反归一化后的训练集数据
            else:  # 如果不使用[-1,1]归一化
                if 1 - proportion > 0:  # 如果测试集比例大于0，则保存测试集数据
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"),
                            test_data)  # 保存测试集数据
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"),
                        train_data)  # 保存训练集数据

        return train_data, test_data  # 返回训练集和测试集数据

    def normalize(self, sq):  # 定义数据归一化方法
        """
        数据归一化，输出的形状和输入相同。
        """
        d = sq.reshape(-1, self.var_num)  # 将数据reshape为二维，方便归一化
        d = self.scaler.transform(d)  # 使用scaler进行归一化
        if self.auto_norm:  # 如果进行[-1,1]归一化
            d = normalize_to_neg_one_to_one(d)  # 归一化到[-1,1]
        return d.reshape(-1, self.window, self.var_num)  # 将数据reshape回原始形状

    def unnormalize(self, sq):  # 定义数据反归一化方法
        """
        数据反归一化，输出的形状和输入相同。
        """
        d = self.__unnormalize(sq.reshape(-1, self.var_num))  # 将数据reshape为二维，方便反归一化
        return d.reshape(-1, self.window, self.var_num)  # 将数据reshape回原始形状

    def __normalize(self, rawdata):  # 定义数据归一化方法
        """
        使用scaler进行归一化，并且可以可选的进行[-1, 1]归一化。
        """
        data = self.scaler.transform(rawdata)  # 使用scaler进行归一化
        if self.auto_norm:  # 如果进行[-1,1]归一化
            data = normalize_to_neg_one_to_one(data)  # 归一化到[-1,1]
        return data  # 返回归一化后的数据

    def __unnormalize(self, data):  # 定义数据反归一化方法
        """
        使用scaler进行反归一化，并且可以可选的进行[-1, 1]反归一化。
        """
        if self.auto_norm:  # 如果进行[-1,1]归一化
            data = unnormalize_to_zero_to_one(data)  # 反归一化到[0,1]
        x = data  # 如果不进行[-1,1]归一化，则x为传入的data
        return self.scaler.inverse_transform(x)  # 返回反归一化后的数据

    @staticmethod
    def divide(data, ratio, seed=2023):
        """
        将数据划分为训练集和测试集。
        """
        size = data.shape[0]  # 获取数据长度
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()  # 保存当前numpy随机数生成器的状态
        np.random.seed(seed)  # 设置随机种子

        regular_train_num = int(np.ceil(size * ratio))  # 计算训练集数量
        # id_rdm = np.random.permutation(size) #  生成随机打乱的索引, 如果需要打乱顺序可以使用这个
        id_rdm = np.arange(size)  # 生成按顺序的索引
        regular_train_id = id_rdm[:regular_train_num]  # 获取训练集索引
        irregular_train_id = id_rdm[regular_train_num:]  # 获取测试集索引

        regular_data = data[regular_train_id, :]  # 获取训练集数据
        irregular_data = data[irregular_train_id, :]  # 获取测试集数据

        # Restore RNG.
        np.random.set_state(st0)  # 恢复随机数生成器的状态
        return regular_data, irregular_data  # 返回训练集和测试集数据

    @staticmethod
    def read_data(filepath, name=''):  # 定义静态方法，用于读取数据
        """
        读取csv数据。
        """
        df = pd.read_csv(filepath, header=0)  # 读取csv文件
        if name in ['etth', 'weather']:  # 如果数据集名称是'etth'或者'weather'，则删除第一列
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values  # 获取数据值
        scaler = MinMaxScaler()  # 创建一个MinMaxScaler对象
        scaler = scaler.fit(data)  # 使用数据拟合scaler
        return data, scaler  # 返回数据和scaler

    def mask_data(self, seed=2023):
        """
        生成缺失值mask。
        """
        masks = np.ones_like(self.samples)  # 创建全1的掩码，形状和样本相同
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()  # 保存numpy随机数生成器的状态
        np.random.seed(seed)  # 设置随机种子

        for idx in range(self.samples.shape[0]):  # 遍历所有样本
            x = self.samples[idx, :, :]  # 获取当前样本 (seq_length, feat_dim)
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # 调用noise_mask函数生成mask (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask  # 将生成的mask添加到masks中

        if self.save2npy:  # 如果save2npy为True, 则保存mask
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)  # 保存mask到npy文件

        # Restore RNG.
        np.random.set_state(st0)  # 恢复numpy随机数生成器的状态
        return masks.astype(bool)  # 返回mask，类型是bool

    def __getitem__(self, ind):  # 定义__getitem__方法，用于获取数据
        """
        根据索引获取数据。
        """
        if self.period == 'test':  # 如果是测试集
            x = self.samples[ind, :, :]  # 获取样本数据 (seq_length, feat_dim)
            m = self.masking[ind, :, :]  # 获取mask数据 (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)  # 返回样本数据和mask，类型都是torch.Tensor，并转换为float类型
        x = self.samples[ind, :, :]  # 获取样本数据 (seq_length, feat_dim)
        return torch.from_numpy(x).float()  # 返回样本数据，类型是torch.Tensor，并转换为float类型

    def __len__(self):  # 定义__len__方法，用于获取数据集长度
        """
        返回数据集长度。
        """
        return self.sample_num  # 返回样本数量


class fMRIDataset(CustomDataset):  # 定义一个继承自CustomDataset的fMRIDataset类
    def __init__(  # 定义初始化方法
            self,
            proportion=1.,  # 设置训练集比例，默认值为1
            **kwargs  # 接受其他参数
    ):
        super().__init__(proportion=proportion, **kwargs)  # 调用父类的初始化方法

    @staticmethod
    def read_data(filepath, name=''):  # 定义静态方法，用于读取数据
        """
        读取mat文件数据。
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']  # 读取mat文件，并获取'ts'变量的数据
        scaler = MinMaxScaler()  # 创建一个MinMaxScaler对象
        scaler = scaler.fit(data)  # 使用数据拟合scaler
        return data, scaler  # 返回数据和scaler



class AircraftTrajectoryDataLoaders(Dataset):
    def __init__(  # 定义类的初始化方法
            self,
            name,  # 数据集名称
            data_root,  # 数据文件路径
            window=64,  # 时间窗口大小
            proportion=0.8,  # 训练集比例
            save2npy=True,  # 是否将数据保存为.npy文件
            neg_one_to_one=True,  # 是否将数据归一化到[-1, 1]之间，否则使用最小最大归一化
            seed=123,  # 随机种子
            period='train',  # 数据集类型，'train'或'test'
            output_dir='./OUTPUT',  # 输出目录
            predict_length=None,  # 预测长度, 仅在测试集使用
            missing_ratio=None,  # 缺失值比例, 仅在测试集使用
            style='separate',  # 缺失值的风格，可选'separate', 'mask'等
            distribution='geometric',  # 掩码的分布，可选'geometric', 'random'等
            mean_mask_length=3  # 掩码的平均长度，当distribution为'geometric'时起作用
    ):
        super(CustomDataset, self).__init__()  # 调用父类的初始化方法
        assert period in ['train', 'test'], 'period must be train or test.'  # 断言period必须是'train'或'test'
        if period == 'train':  # 如果是训练集，则断言predict_length和missing_ratio为None
            assert ~(
                        predict_length is not None or missing_ratio is not None), 'Train dataset does not support missing values or predict length'
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio  # 设置数据集名称，预测长度，缺失值比例
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length  # 设置mask的风格，分布，平均长度
        self.rawdata, self.scaler = self.read_data(self,data_root, self.name)  # 读取数据和scaler
        self.dir = os.path.join(output_dir, 'samples')  # 设置保存数据的目录
        os.makedirs(self.dir, exist_ok=True)  # 创建保存数据的目录，如果目录已存在则不报错

        self.window, self.period = window, period  # 设置时间窗口和数据集类型
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]  # 获取原始数据的长度和变量数
        self.save2npy = save2npy  # 设置是否保存数据
        self.auto_norm = neg_one_to_one  # 设置是否进行[-1,1]归一化

        self.data = self.__normalize(self.rawdata)  # 对原始数据进行归一化
        train, inference = self.__getsamples(self.data, proportion, seed)  # 获取训练集和测试集

        self.samples = train if period == 'train' else inference  # 如果是训练集，则使用训练集数据，否则使用测试集数据
        if period == 'test':  # 如果是测试集
            if missing_ratio is not None:  # 如果missing_ratio不为None，则生成缺失值mask
                self.masking = self.mask_data(seed)  # 使用mask_data函数生成mask
            elif predict_length is not None:  # 如果predict_length不为None, 则生成预测mask
                masks = np.ones(self.samples.shape)  # 生成全1的mask
                masks[:, -predict_length:, :] = 0  # 将末尾predict_length的时间步mask设置为0
                self.masking = masks.astype(bool)  # 将mask类型设置为bool
            else:
                raise NotImplementedError(
                    "Missing ratio or predict length must be set when test.")  # 如果测试集既没有缺失值，也没有预测长度，则报错
        self.sample_num = self.samples.shape[0]  # 设置样本数量

    def __getsamples(self, data, proportion, seed):
        """
        将数据划分成窗口，并分为训练集和测试集。
        """
        sample_num_total = max(self.len - self.window + 1, 0)  # 计算样本总数
        x = np.zeros((sample_num_total, self.window, self.var_num))  # 创建一个存储窗口数据的numpy数组
        for i in range(sample_num_total):  # 遍历所有窗口
            start = i  # 窗口起始位置
            end = i + self.window  # 窗口结束位置
            x[i, :, :] = data[start:end, :]  # 将数据添加到x中

        train_data, test_data = self.divide(x, proportion, seed)  # 将数据划分成训练集和测试集

        if self.save2npy:  # 如果save2npy为True，则保存数据到npy文件
            if 1 - proportion > 0:  # 如果测试集比例大于0，则保存测试集数据
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"),
                        self.unnormalize(test_data))  # 保存反归一化后的测试集数据
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"),
                    self.unnormalize(train_data))  # 保存反归一化后的训练集数据
            if self.auto_norm:  # 如果使用[-1, 1]归一化，则保存归一化后的数据
                if 1 - proportion > 0:  # 如果测试集比例大于0，则保存测试集数据
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"),
                            unnormalize_to_zero_to_one(test_data))  # 保存反归一化后的测试集数据
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"),
                        unnormalize_to_zero_to_one(train_data))  # 保存反归一化后的训练集数据
            else:  # 如果不使用[-1,1]归一化
                if 1 - proportion > 0:  # 如果测试集比例大于0，则保存测试集数据
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"),
                            test_data)  # 保存测试集数据
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"),
                        train_data)  # 保存训练集数据

        return train_data, test_data  # 返回训练集和测试集数据

    def normalize(self, sq):  # 定义数据归一化方法
        """
        数据归一化，输出的形状和输入相同。
        """
        d = sq.reshape(-1, self.var_num)  # 将数据reshape为二维，方便归一化
        d = self.scaler.transform(d)  # 使用scaler进行归一化
        if self.auto_norm:  # 如果进行[-1,1]归一化
            d = normalize_to_neg_one_to_one(d)  # 归一化到[-1,1]
        return d.reshape(-1, self.window, self.var_num)  # 将数据reshape回原始形状

    def unnormalize(self, sq):  # 定义数据反归一化方法
        """
        数据反归一化，输出的形状和输入相同。
        """
        d = self.__unnormalize(sq.reshape(-1, self.var_num))  # 将数据reshape为二维，方便反归一化
        return d.reshape(-1, self.window, self.var_num)  # 将数据reshape回原始形状

    def __normalize(self, rawdata):  # 定义数据归一化方法
        """
        使用scaler进行归一化，并且可以可选的进行[-1, 1]归一化。
        """
        data = self.scaler.transform(rawdata)  # 使用scaler进行归一化
        if self.auto_norm:  # 如果进行[-1,1]归一化
            data = normalize_to_neg_one_to_one(data)  # 归一化到[-1,1]
        return data  # 返回归一化后的数据

    def __unnormalize(self, data):  # 定义数据反归一化方法
        """
        使用scaler进行反归一化，并且可以可选的进行[-1, 1]反归一化。
        """
        if self.auto_norm:  # 如果进行[-1,1]归一化
            data = unnormalize_to_zero_to_one(data)  # 反归一化到[0,1]
        x = data  # 如果不进行[-1,1]归一化，则x为传入的data
        return self.scaler.inverse_transform(x)  # 返回反归一化后的数据

    @staticmethod
    def divide(data, ratio, seed=2023):
        """
        将数据划分为训练集和测试集。
        """
        size = data.shape[0]  # 获取数据长度
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()  # 保存当前numpy随机数生成器的状态
        np.random.seed(seed)  # 设置随机种子

        regular_train_num = int(np.ceil(size * ratio))  # 计算训练集数量
        # id_rdm = np.random.permutation(size) #  生成随机打乱的索引, 如果需要打乱顺序可以使用这个
        id_rdm = np.arange(size)  # 生成按顺序的索引
        regular_train_id = id_rdm[:regular_train_num]  # 获取训练集索引
        irregular_train_id = id_rdm[regular_train_num:]  # 获取测试集索引

        regular_data = data[regular_train_id, :]  # 获取训练集数据
        irregular_data = data[irregular_train_id, :]  # 获取测试集数据

        # Restore RNG.
        np.random.set_state(st0)  # 恢复随机数生成器的状态
        return regular_data, irregular_data  # 返回训练集和测试集数据

    @staticmethod
    def read_data(self,filepath, name=''):  # 定义静态方法，用于读取数据
        """
        读取csv数据。
        """
        csv_files = sorted(glob(os.path.join(filepath, "*.csv")))
        feature_columns = ["Longitude", "Latitude","Altitude","TAS" ]

        # Fit scaler if it hasn't been fitted yet
        # if self.scaler and not hasattr(self.scaler, 'scale_'):
        #     all_data = pd.concat([pd.read_csv(file)[feature_columns] for file in csv_files]).astype(np.float32)
        #     self.scaler.fit(all_data)

        datasets = []
        for file in csv_files:
            df = pd.read_csv(file)
            df['Id'] = df['Id'].astype(str)
            trajectories, time_data = self.process_file(df, feature_columns)

        # df = pd.read_csv(filepath, header=0)  # 读取csv文件
        # if name == 'etth':  # 如果是etth数据集，则删除第一列
        #     df.drop(df.columns[0], axis=1, inplace=True)
        # data = df.values  # 获取数据值
        scaler = MinMaxScaler()  # 创建一个MinMaxScaler对象
        scaler = scaler.fit(data)  # 使用数据拟合scaler
        return data, scaler  # 返回数据和scaler

    def process_file(self, df, feature_columns):
        """Process each file to extract trajectories and time features."""
        iso_times = df["ISO time"].values
        df_stamp = pd.DataFrame({'date': pd.to_datetime(iso_times)})

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq).transpose(1, 0)

        # Scale features
        all_data = df[feature_columns].values.astype(np.float32)
        if self.scaler:
            all_data = self.scaler.transform(all_data)
            all_data[:, :3] *= 100
            all_data[:, 6:] *= 100

        # Split data based on flag
        num_trajectories = len(df['Id'].unique())
        train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2
        if self.flag == 'train':
            indices = list(range(int(num_trajectories * train_ratio)))
        elif self.flag == 'val':
            start = int(num_trajectories * train_ratio)
            end = start + int(num_trajectories * val_ratio)
            indices = list(range(start, end))
        else:  # 'test'
            start = int(num_trajectories * (train_ratio + val_ratio))
            indices = list(range(start, num_trajectories))

        trajectories = [all_data[df["Id"] == id_val] for id_val in df['Id'].unique() if
                        df['Id'].unique().tolist().index(id_val) in indices]
        time_data = [data_stamp[df["Id"] == id_val] for id_val in df['Id'].unique() if
                     df['Id'].unique().tolist().index(id_val) in indices]

        return trajectories, time_data

    def mask_data(self, seed=2023):
        """
        生成缺失值mask。
        """
        masks = np.ones_like(self.samples)  # 创建全1的掩码，形状和样本相同
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()  # 保存numpy随机数生成器的状态
        np.random.seed(seed)  # 设置随机种子

        for idx in range(self.samples.shape[0]):  # 遍历所有样本
            x = self.samples[idx, :, :]  # 获取当前样本 (seq_length, feat_dim)
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # 调用noise_mask函数生成mask (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask  # 将生成的mask添加到masks中

        if self.save2npy:  # 如果save2npy为True, 则保存mask
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)  # 保存mask到npy文件

        # Restore RNG.
        np.random.set_state(st0)  # 恢复numpy随机数生成器的状态
        return masks.astype(bool)  # 返回mask，类型是bool

    def __getitem__(self, ind):  # 定义__getitem__方法，用于获取数据
        """
        根据索引获取数据。
        """
        if self.period == 'test':  # 如果是测试集
            x = self.samples[ind, :, :]  # 获取样本数据 (seq_length, feat_dim)
            m = self.masking[ind, :, :]  # 获取mask数据 (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)  # 返回样本数据和mask，类型都是torch.Tensor，并转换为float类型
        x = self.samples[ind, :, :]  # 获取样本数据 (seq_length, feat_dim)
        return torch.from_numpy(x).float()  # 返回样本数据，类型是torch.Tensor，并转换为float类型

    def __len__(self):  # 定义__len__方法，用于获取数据集长度
        """
        返回数据集长度。
        """
        return self.sample_num  # 返回样本数量


''' 
class CustomTrajectoryDataset(Dataset):  # 自定义数据集类，继承自torch.utils.data.Dataset
    def __init__(
            self,
            name,  # 数据集名称
            data_root,  # 数据文件路径
            window=64,  # 时间窗口大小
            proportion=0.8,  # 训练集比例
            save2npy=True,  # 是否将数据保存为.npy文件
            neg_one_to_one=True,  # 是否使用[-1, 1]归一化
            seed=123,  # 随机种子
            period='train',  # 数据集类型，'train'或'test'
            output_dir='./OUTPUT',  # 输出目录
            predict_length=None,  # 预测长度，仅在测试集使用
            missing_ratio=None,  # 缺失值比例，仅在测试集使用
            style='separate',  # 缺失值的风格
            distribution='geometric',  # 掩码的分布
            mean_mask_length=3,  # 掩码的平均长度
            **kwargs  # 允许传递额外的参数 用于选取轨迹feature_columns和dtype
    ):
        super(CustomTrajectoryDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(
                    predict_length is not None or missing_ratio is not None), 'Train dataset does not support missing values or predict length'
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.data_columns_dtype = kwargs.get('data_columns_dtype', {})
        self.data_columns = list(self.data_columns_dtype.keys())  # 从YAML配置中获取列名
        self.rawdata, self.scaler = self.read_data(data_root)  # 读取所有csv数据，并进行归一化
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]  # 获取数据的长度和变量维度
        self.sample_num_total = max(self.len - self.window + 1, 0)  # 计算滑动窗口的总数
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.data = self.__normalize(self.rawdata)  # 对读取的数据进行归一化
        train, inference = self.__getsamples(self.data, proportion, seed)  # 获取训练集和测试集

        self.samples = train if period == 'train' else inference  # 如果是训练集，则使用训练集数据，否则使用测试集数据
        if period == 'test':  # 如果是测试集
            if missing_ratio is not None:  # 如果missing_ratio不为None，则生成缺失值mask
                self.masking = self.mask_data(seed)  # 使用mask_data函数生成mask
            elif predict_length is not None:  # 如果predict_length不为None, 则生成预测mask
                masks = np.ones(self.samples.shape)  # 生成全1的mask
                masks[:, -predict_length:, :] = 0  # 将末尾predict_length的时间步mask设置为0
                self.masking = masks.astype(bool)  # 将mask类型设置为bool
            else:
                raise NotImplementedError(
                    "Missing ratio or predict length must be set when test.")  # 如果测试集既没有缺失值，也没有预测长度，则报错
        self.sample_num = self.samples.shape[0]  # 设置样本数量

    def __getsamples(self, data, proportion, seed):
        """
        将数据划分成窗口，并分为训练集和测试集。
        """
        x = np.zeros((self.sample_num_total, self.window, self.var_num))  # 创建一个存储窗口数据的numpy数组
        for i in range(self.sample_num_total):  # 遍历所有窗口
            start = i  # 窗口起始位置
            end = i + self.window  # 窗口结束位置
            x[i, :, :] = data[start:end, :]  # 将数据添加到x中

        train_data, test_data = self.divide(x, proportion, seed)  # 将数据划分成训练集和测试集

        if self.save2npy:  # 如果save2npy为True，则保存数据到npy文件
            if 1 - proportion > 0:  # 如果测试集比例大于0，则保存测试集数据
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"),
                        self.unnormalize(test_data))  # 保存反归一化后的测试集数据
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"),
                    self.unnormalize(train_data))  # 保存反归一化后的训练集数据
            if self.auto_norm:  # 如果使用[-1, 1]归一化，则保存归一化后的数据
                if 1 - proportion > 0:  # 如果测试集比例大于0，则保存测试集数据
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"),
                            unnormalize_to_zero_to_one(test_data))  # 保存反归一化后的测试集数据
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"),
                        unnormalize_to_zero_to_one(train_data))  # 保存反归一化后的训练集数据
            else:  # 如果不使用[-1,1]归一化
                if 1 - proportion > 0:  # 如果测试集比例大于0，则保存测试集数据
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"),
                            test_data)  # 保存测试集数据
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"),
                        train_data)  # 保存训练集数据

        return train_data, test_data  # 返回训练集和测试集数据

    def normalize(self, sq):
        """数据归一化，输出的形状和输入相同。"""
        d = sq.reshape(-1, self.var_num)  # 将数据reshape为二维，方便归一化
        d = self.scaler.transform(d)  # 使用scaler进行归一化
        if self.auto_norm:  # 如果进行[-1,1]归一化
            d = normalize_to_neg_one_to_one(d)  # 归一化到[-1,1]
        return d.reshape(-1, self.window, self.var_num)  # 将数据reshape回原始形状

    def unnormalize(self, sq):
        """数据反归一化，输出的形状和输入相同。"""
        d = self.__unnormalize(sq.reshape(-1, self.var_num))  # 将数据reshape为二维，方便反归一化
        return d.reshape(-1, self.window, self.var_num)  # 将数据reshape回原始形状

    def __normalize(self, rawdata):
        """使用scaler进行归一化，并且可以可选的进行[-1, 1]归一化。"""
        data = self.scaler.transform(rawdata)  # 使用scaler进行归一化
        if self.auto_norm:  # 如果进行[-1,1]归一化
            data = normalize_to_neg_one_to_one(data)  # 归一化到[-1,1]
        return data

    def __unnormalize(self, data):
        """使用scaler进行反归一化，并且可以可选的进行[-1, 1]反归一化。"""
        if self.auto_norm:  # 如果进行[-1,1]归一化
            data = unnormalize_to_zero_to_one(data)  # 反归一化到[0,1]
        x = data  # 如果不进行[-1,1]归一化，则x为传入的data
        return self.scaler.inverse_transform(x)  # 返回反归一化后的数据

    @staticmethod
    def divide(data, ratio, seed=2023):
        """将数据划分为训练集和测试集。"""
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        # id_rdm = np.random.permutation(size)
        id_rdm = np.arange(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    def read_data(self, filepath):
        """读取所有csv文件中的数据, 并合并成一个numpy数组"""
        csv_files = sorted(glob(os.path.join(filepath, "*.csv")))  # 获取所有csv文件路径
        # feature_columns = ["Longitude", "Latitude", "Altitude", "TAS"]  # 定义使用的特征列
        all_data = []
        for file in csv_files:  # 遍历所有csv文件
            df = pd.read_csv(file, usecols= self.data_columns, dtype=self.data_columns_dtype, engine='c')  # 读取当前csv文件
            df['Id'] = df['Id'].astype(str)  # 将id列转化为str类型
            feature_columns = self.data_columns.copy()  # 创建副本
            if 'Id' in feature_columns:
                feature_columns.remove('Id')
            trajectories = self.process_file(df, feature_columns)  # 处理当前csv文件
            all_data.extend(trajectories)  # 将当前文件的轨迹数据添加到all_data中

        # all_data = np.concatenate(all_data, axis=0)  # 将所有轨迹数据合并成一个numpy数组
        all_data = np.concatenate(all_data)
        scaler = MinMaxScaler()  # 创建一个MinMaxScaler对象
        scaler = scaler.fit(all_data)  # 使用数据拟合scaler
        return all_data, scaler  # 返回数据和scaler

    def process_file(self, df, feature_columns):
        """Process each file to extract trajectories."""
        all_data = df[feature_columns].values.astype(np.float32)
        trajectories = []

        # 确定经度和纬度在 feature_columns 中的索引
        longitude_index = feature_columns.index('Longitude') if 'Longitude' in feature_columns else None
        latitude_index = feature_columns.index('Latitude') if 'Latitude' in feature_columns else None

        for id_val in df['Id'].unique():
            traj_data = all_data[df["Id"] == id_val]

            # 仅对经度和纬度进行处理
            if longitude_index is not None:
                traj_data[:, longitude_index] = traj_data[:, longitude_index] - traj_data[0, longitude_index]
            if latitude_index is not None:
                traj_data[:, latitude_index] = traj_data[:, latitude_index] - traj_data[0, latitude_index]

            trajectories.append(traj_data)

        return trajectories

    # def read_and_process_file(self, file, feature_columns):
    #     """读取并处理单个CSV文件"""
    #     try:
    #         # 读取时根据YAML配置的dtype来设置数据类型
    #         dtypes = {k: v for k, v in self.data_columns_dtype.items() if k in feature_columns}
    #         df = pd.read_csv(file, usecols=feature_columns, dtype=dtypes)
    #         df['Id'] = df['Id'].astype(str)
    #         print(f"Data types after reading {file}:")
    #         df.info()  # 打印列的数据类型，方便调试
    #
    #         # 获取每个轨迹的起始索引
    #         _, start_indices = np.unique(df['Id'].values, return_index=True)
    #         start_indices.sort()
    #
    #         end_indices = np.append(start_indices[1:], df.shape[0])
    #
    #         # 从feature_columns中移除'Id'
    #         feature_columns_without_id = [col for col in feature_columns if col != 'Id']
    #
    #         # 使用feature_columns_without_id来创建trajectories
    #         trajectories = np.zeros((len(start_indices), max(end_indices - start_indices), len(feature_columns_without_id)),
    #                                 dtype=np.float32)
    #
    #         for i, (start, end) in enumerate(zip(start_indices, end_indices)):
    #             # 只选择需要的数值列进行处理
    #             traj_data = df.iloc[start:end][feature_columns_without_id].values.astype(np.float32)
    #             traj_data -= traj_data[0]
    #             trajectories[i, :len(traj_data)] = traj_data
    #
    #         # 移除多余的零填充
    #         trajectories = [traj[:len(traj[np.any(traj != 0, axis=1)])] for traj in trajectories]
    #
    #         return trajectories
    #     except Exception as e:
    #         print(f"Error reading or processing {file}: {e}")
    #         return None
    #
    # def read_data(self, filepath):
    #     """读取所有csv文件中的数据, 并合并成一个numpy数组"""
    #     csv_files = sorted(glob(os.path.join(filepath, "*.csv")))  # 获取所有csv文件路径
    #
    #     with ThreadPoolExecutor() as executor:
    #         future_to_file = {executor.submit(self.read_and_process_file, file, self.feature_columns): file for file in
    #                           csv_files}
    #         all_trajectories = []
    #         for future in as_completed(future_to_file):
    #             data = future.result()
    #             if data is not None:
    #                 all_trajectories.extend(data)
    #
    #     if not all_trajectories:
    #         raise ValueError("No data was successfully read from any CSV file.")
    #
    #     all_data = np.concatenate(all_trajectories, axis=0)
    #
    #     scaler = MinMaxScaler()
    #     scaler = scaler.fit(all_data)
    #
    #     return all_data, scaler

    def mask_data(self, seed=2023):
        """生成缺失值mask。"""
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # 调用noise_mask函数生成mask (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        """根据索引获取数据。"""
        if self.period == 'test':
            x = self.samples[ind, :, :]
            m = self.masking[ind, :, :]
            return torch.from_numpy(x).float(), torch.from_numpy(m)  # 返回样本数据和mask，类型都是torch.Tensor，并转换为float类型
        x = self.samples[ind, :, :]
        return torch.from_numpy(x).float()  # 返回样本数据，类型是torch.Tensor，并转换为float类型

    def __len__(self):
        """返回数据集长度。"""
        return self.sample_num
'''

class CustomTrajectoryDataset(Dataset):  # 自定义数据集类，继承自torch.utils.data.Dataset
    def __init__(
            self,
            name,  # 数据集名称
            data_root,  # 数据文件路径
            window=64,  # 时间窗口大小
            proportion=0.8,  # 训练集比例
            save2npy=True,  # 是否将数据保存为.npy文件
            neg_one_to_one=True,  # 是否使用[-1, 1]归一化
            seed=123,  # 随机种子
            period='train',  # 数据集类型，'train'或'test'
            output_dir='./OUTPUT',  # 输出目录
            predict_length=None,  # 预测长度，仅在测试集使用
            missing_ratio=None,  # 缺失值比例，仅在测试集使用
            style='separate',  # 缺失值的风格
            distribution='geometric',  # 掩码的分布
            mean_mask_length=3,  # 掩码的平均长度
            **kwargs  # 允许传递额外的参数 用于选取轨迹feature_columns和dtype
    ):
        super(CustomTrajectoryDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(
                    predict_length is not None or missing_ratio is not None), 'Train dataset does not support missing values or predict length'
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.data_columns_dtype = kwargs.get('data_columns_dtype', {})
        self.data_columns = list(self.data_columns_dtype.keys())  # 从YAML配置中获取列名
        self.all_data, self.scaler, self.trajectory_lengths = self.read_data(data_root)  # 读取所有csv数据，并进行归一化
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        # self.len, self.var_num = self.all_data.shape[0], self.all_data.shape[-1]  # 获取数据的长度和变量维度
        self.var_num = len(self.data_columns) - 1 if 'Id' in self.data_columns else len(self.data_columns)

        # Modify calculation of sample_num_total
        self.sample_num_total = sum(max(traj_len - self.window + 1, 0) for traj_len in self.trajectory_lengths)

        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        # Normalize each trajectory individually
        self.data = [self.__normalize(traj) for traj in self.all_data]

        # Concatenate all samples after applying window
        # 获取'split_mode'参数，如果未提供则默认设为 'trajectory'
        self.split_mode = kwargs.get('split_mode', 'trajectory')
        train, inference = self.__getsamples(self.data, proportion, seed, split_mode = self.split_mode )
        self.samples = np.concatenate(train) if period == 'train' else np.concatenate(inference)

        if period == 'test':  # 如果是测试集
            if missing_ratio is not None:  # 如果missing_ratio不为None，则生成缺失值mask
                self.masking = self.mask_data(seed)  # 使用mask_data函数生成mask
            elif predict_length is not None:  # 如果predict_length不为None, 则生成预测mask
                masks = np.ones(self.samples.shape)  # 生成全1的mask
                masks[:, -predict_length:, :] = 0  # 将末尾predict_length的时间步mask设置为0
                self.masking = masks.astype(bool)  # 将mask类型设置为bool
            else:
                raise NotImplementedError(
                    "Missing ratio or predict length must be set when test.")  # 如果测试集既没有缺失值，也没有预测长度，则报错
        self.sample_num = self.samples.shape[0]  # 设置样本数量

    def __getsamples(self, data, proportion, seed, split_mode='trajectory'):
        """
        将数据划分成窗口，并分为训练集和测试集。

        Args:
            data: 包含所有轨迹的列表。
            proportion: 训练集比例。
            seed: 随机种子。
            split_mode: 分割方式，'trajectory' 表示整轨分割，'in-trajectory' 表示轨迹内分割。
        """
        train_data_list = []
        test_data_list = []

        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        if split_mode == 'in-trajectory':
            for traj in data:
                size = traj.shape[0]

                # Sliding window within each trajectory
                x = np.zeros((max(size - self.window + 1, 0), self.window, self.var_num))
                for i in range(max(size - self.window + 1, 0)):
                    start = i
                    end = i + self.window
                    x[i, :, :] = traj[start:end, :]

                # Split trajectory into train and test based on proportion
                # 随机划分
                all_indices = np.arange(max(size - self.window + 1, 0))  # 获取所有滑窗样本的索引
                np.random.shuffle(all_indices)  # 打乱索引

                regular_train_num = int(np.ceil(len(all_indices) * proportion))  # 重新计算训练集的样本数量

                regular_train_id = all_indices[:regular_train_num]
                irregular_train_id = all_indices[regular_train_num:]

                if len(regular_train_id) > 0:
                    train_data_list.append(x[regular_train_id])
                if len(irregular_train_id) > 0:
                    test_data_list.append(x[irregular_train_id])

        elif split_mode == 'trajectory':
            # Shuffle the order of trajectories
            traj_indices = np.arange(len(data))
            np.random.shuffle(traj_indices)

            # Split trajectories into train and test sets
            num_train_traj = int(len(data) * proportion)
            train_traj_indices = traj_indices[:num_train_traj]
            test_traj_indices = traj_indices[num_train_traj:]

            # Apply sliding window to each trajectory in train and test sets
            for idx in train_traj_indices:
                traj = data[idx]
                size = traj.shape[0]
                x = np.zeros((max(size - self.window + 1, 0), self.window, self.var_num))
                for i in range(max(size - self.window + 1, 0)):
                    start = i
                    end = i + self.window
                    x[i, :, :] = traj[start:end, :]
                if len(x) > 0:
                    train_data_list.append(x)

            for idx in test_traj_indices:
                traj = data[idx]
                size = traj.shape[0]
                x = np.zeros((max(size - self.window + 1, 0), self.window, self.var_num))
                for i in range(max(size - self.window + 1, 0)):
                    start = i
                    end = i + self.window
                    x[i, :, :] = traj[start:end, :]
                if len(x) > 0:
                    test_data_list.append(x)

        else:
            raise ValueError(f"Invalid split_mode: {split_mode}")

        # Restore RNG.
        np.random.set_state(st0)

        return train_data_list, test_data_list

    def normalize(self, sq):
        """数据归一化，输出的形状和输入相同。"""
        d = sq.reshape(-1, self.var_num)  # 将数据reshape为二维，方便归一化
        d = self.scaler.transform(d)  # 使用scaler进行归一化
        if self.auto_norm:  # 如果进行[-1,1]归一化
            d = normalize_to_neg_one_to_one(d)  # 归一化到[-1,1]
        return d.reshape(-1, self.window, self.var_num)  # 将数据reshape回原始形状

    def unnormalize(self, sq):
        """数据反归一化，输出的形状和输入相同。"""
        d = self.__unnormalize(sq.reshape(-1, self.var_num))  # 将数据reshape为二维，方便反归一化
        return d.reshape(-1, self.window, self.var_num)  # 将数据reshape回原始形状

    def __normalize(self, rawdata):
        """使用scaler进行归一化，并且可以可选的进行[-1, 1]归一化。"""
        data = self.scaler.transform(rawdata)  # 使用scaler进行归一化
        if self.auto_norm:  # 如果进行[-1,1]归一化
            data = normalize_to_neg_one_to_one(data)  # 归一化到[-1,1]
        return data

    def __unnormalize(self, data):
        """使用scaler进行反归一化，并且可以可选的进行[-1, 1]反归一化。"""
        if self.auto_norm:  # 如果进行[-1,1]归一化
            data = unnormalize_to_zero_to_one(data)  # 反归一化到[0,1]
        x = data  # 如果不进行[-1,1]归一化，则x为传入的data
        return self.scaler.inverse_transform(x)  # 返回反归一化后的数据

    @staticmethod
    def divide(data, ratio, seed=2023):
        """将数据划分为训练集和测试集。"""
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        # id_rdm = np.random.permutation(size)
        id_rdm = np.arange(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    def read_data(self, filepath):
        """读取所有csv文件中的数据, 并合并成一个numpy数组"""
        csv_files = sorted(glob(os.path.join(filepath, "*.csv")))  # 获取所有csv文件路径
        all_data = []
        trajectory_lengths = []
        for file in csv_files:  # 遍历所有csv文件
            df = pd.read_csv(file, usecols= self.data_columns, dtype=self.data_columns_dtype, engine='c')  # 读取当前csv文件
            df['Id'] = df['Id'].astype(str)  # 将id列转化为str类型
            feature_columns = self.data_columns.copy()  # 创建副本
            if 'Id' in feature_columns:
                feature_columns.remove('Id')
            trajectories = self.process_file(df, feature_columns)  # 处理当前csv文件
            all_data.extend(trajectories)  # 将当前文件的轨迹数据添加到all_data中
            trajectory_lengths.extend([len(traj) for traj in trajectories])

        # Calculate scaler based on all data
        scaler = MinMaxScaler()
        scaler.fit(np.concatenate(all_data))

        return all_data, scaler, trajectory_lengths

    def process_file(self, df, feature_columns):
        """Process each file to extract trajectories."""
        all_data = df[feature_columns].values.astype(np.float32)
        trajectories = []

        # 确定经度和纬度在 feature_columns 中的索引
        longitude_index = feature_columns.index('Longitude') if 'Longitude' in feature_columns else None
        latitude_index = feature_columns.index('Latitude') if 'Latitude' in feature_columns else None

        for id_val in df['Id'].unique():
            traj_data = all_data[df["Id"] == id_val]

            # 仅对经度和纬度进行处理
            if longitude_index is not None:
                traj_data[:, longitude_index] = traj_data[:, longitude_index] - traj_data[0, longitude_index]
            if latitude_index is not None:
                traj_data[:, latitude_index] = traj_data[:, latitude_index] - traj_data[0, latitude_index]

            trajectories.append(traj_data)

        return trajectories

    def mask_data(self, seed=2023):
        """生成缺失值mask。"""
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # 调用noise_mask函数生成mask (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        """根据索引获取数据。"""
        if self.period == 'test':
            x = self.samples[ind, :, :]
            m = self.masking[ind, :, :]
            return torch.from_numpy(x).float(), torch.from_numpy(m)  # 返回样本数据和mask，类型都是torch.Tensor，并转换为float类型
        x = self.samples[ind, :, :]
        return torch.from_numpy(x).float()  # 返回样本数据，类型是torch.Tensor，并转换为float类型

    def __len__(self):
        """返回数据集长度。"""
        return self.sample_num