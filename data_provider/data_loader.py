from pathlib import Path
from gluonts.dataset.jsonl import JsonLinesWriter
from gluonts.dataset.repository import get_dataset
import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
# from data_provider.m4 import M4Dataset, M4Meta # removed due to 
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len,
                    12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 +
                    4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(
                df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 *
                    30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 *
                    30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(
                df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len,
                    len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(
                df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# Removed due to the LICENSE file constraints of m4.py
# class Dataset_M4(Dataset):
#     def __init__(self, root_path, flag='pred', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
#                  seasonal_patterns='Yearly'):
#         # size [seq_len, label_len, pred_len]
#         # init
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.inverse = inverse
#         self.timeenc = timeenc
#         self.root_path = root_path

#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]

#         self.seasonal_patterns = seasonal_patterns
#         self.history_size = M4Meta.history_size[seasonal_patterns]
#         self.window_sampling_limit = int(self.history_size * self.pred_len)
#         self.flag = flag

#         self.__read_data__()

#     def __read_data__(self):
#         # M4Dataset.initialize()
#         if self.flag == 'train':
#             dataset = M4Dataset.load(
#                 training=True, dataset_file=self.root_path)
#         else:
#             dataset = M4Dataset.load(
#                 training=False, dataset_file=self.root_path)
#         training_values = np.array(
#             [v[~np.isnan(v)] for v in
#              dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
#         self.ids = np.array(
#             [i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
#         self.timeseries = [ts for ts in training_values]

#     def __getitem__(self, index):
#         insample = np.zeros((self.seq_len, 1))
#         insample_mask = np.zeros((self.seq_len, 1))
#         outsample = np.zeros((self.pred_len + self.label_len, 1))
#         outsample_mask = np.zeros(
#             (self.pred_len + self.label_len, 1))  # m4 dataset

#         sampled_timeseries = self.timeseries[index]
#         cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
#                                       high=len(sampled_timeseries),
#                                       size=1)[0]

#         insample_window = sampled_timeseries[max(
#             0, cut_point - self.seq_len):cut_point]
#         insample[-len(insample_window):, 0] = insample_window
#         insample_mask[-len(insample_window):, 0] = 1.0
#         outsample_window = sampled_timeseries[
#             cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
#         outsample[:len(outsample_window), 0] = outsample_window
#         outsample_mask[:len(outsample_window), 0] = 1.0
#         return insample, outsample, insample_mask, outsample_mask

#     def __len__(self):
#         return len(self.timeseries)

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

#     def last_insample_window(self):
#         """
#         The last window of insample size of all timeseries.
#         This function does not support batching and does not reshuffle timeseries.

#         :return: Last insample window of all timeseries. Shape "timeseries, insample size"
#         """
#         insample = np.zeros((len(self.timeseries), self.seq_len))
#         insample_mask = np.zeros((len(self.timeseries), self.seq_len))
#         for i, ts in enumerate(self.timeseries):
#             ts_last_window = ts[-self.seq_len:]
#             insample[i, -len(ts):] = ts_last_window
#             insample_mask[i, -len(ts):] = 1.0
#         return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(
            root_path, 'test_label.csv')).values[:, 1:]

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(
            os.path.join(root_path, "MSL_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(
            root_path, "SMAP_test_label.npy"))

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(
            os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        # print("test:", self.test.shape)
        # print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_path, file_list=None, limit_size=None, flag=None):
        self.root_path = root_path
        self.all_df, self.labels_df = self.load_all(
            root_path, file_list=file_list, flag=flag)
        # all sample IDs (integer indices 0 ... num_samples-1)
        self.all_IDs = self.all_df.index.unique()

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        # print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(
                root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(
                os.path.join(root_path, '*')))
        if flag is not None:
            # fix the train TRAIN bug.
            pattern = re.compile(flag, re.IGNORECASE)
            data_paths = list(filter(lambda x: pattern.search(x), data_paths))
            # data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(
            p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern = '*.ts'
            raise Exception(
                "No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(
            input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                   replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        # special process for numerical stability
        if self.root_path.count('EthanolConcentration') > 0:
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(
                torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
            torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

    def __len__(self):
        return len(self.all_IDs)


default_dataset_writer = JsonLinesWriter()


class GLUONTSDataset(Dataset):
    """
    NOTE: added flags for splits, multivariate timeseries, and normalization

    Copied from GLUONTS:

    Get a repository dataset.

    The datasets that can be obtained through this function have been used
    with different processing over time by several papers (e.g., [SFG17]_,
    [LCY+18]_, and [YRD15]_) or are obtained through the `Monash Time Series
    Forecasting Repository <https://forecastingdata.org/>`_.

    Parameters
    ----------
    dataset_name
        Name of the dataset, for instance "m4_hourly".
    regenerate
        Whether to regenerate the dataset even if a local file is present.
        If this flag is False and the file is present, the dataset will not
        be downloaded again.
    path
        Where the dataset should be saved.
    prediction_length
        The prediction length to be used for the dataset. If None, the default
        prediction length will be used. If the dataset is already materialized,
        setting this option to a different value does not have an effect.
        Make sure to set `regenerate=True` in this case. Note that some
        datasets from the Monash Time Series Forecasting Repository do not
        actually have a default prediction length -- the default then depends
        on the frequency of the data:
        - Minutely data --> prediction length of 60 (one hour)
        - Hourly data --> prediction length of 48 (two days)
        - Daily data --> prediction length of 30 (one month)
        - Weekly data --> prediction length of 8 (two months)
        - Monthly data --> prediction length of 12 (one year)
        - Yearly data --> prediction length of 4 (four years)

    Returns
    -------
        Dataset obtained by either downloading or reloading from local file.
    """

    default_pred_lens = {
        "exchange_rate": 30,
        "solar-energy": 24,
        "electricity": 24,
        "traffic": 24,
        "exchange_rate_nips": 30,
        "electricity_nips": 24,
        "traffic_nips": 24,
        "solar_nips": 24,
        "wiki2000_nips": 30,
        "wiki-rolling_nips": 30,
        "taxi_30min": 24,
        "kaggle_web_traffic_without_missing": 59,
        "kaggle_web_traffic_weekly": 8,
        "m1_yearly": 10,
        "m1_quarterly": 8,
        "m1_monthly": 18,
        "nn5_daily_without_missing": 56,
        "nn5_weekly": 8,
        "tourism_monthly": 24,
        "tourism_quarterly": 8,
        "tourism_yearly": 4,
        "cif_2016": 12,
        "wind_farms_without_missing": 60,
        "car_parts_without_missing": 12,
        "dominick": 8,
        "fred_md": 12,
        "pedestrian_counts": 48,
        "hospital": 12,
        "covid_deaths": 30,
        "kdd_cup_2018_without_missing": 48,
        "weather": 30,
        "m3_monthly": 18,
        "m3_quarterly": 8,
        "m3_yearly": 6,
        "m3_other": 8,
        "m4_hourly": 48,
        "m4_daily": 14,
        "m4_weekly": 13,
        "m4_monthly": 18,
        "m4_quarterly": 8,
        "m4_yearly": 6,
        "uber_tlc_daily": 7,
        "uber_tlc_hourly": 24,
        "airpassengers": 12,
        "australian_electricity_demand": 60,
        "electricity_hourly": 48,
        "electricity_weekly": 8,
        "rideshare_without_missing": 48,
        "saugeenday": 30,
        "solar_10_minutes": 60,
        "solar_weekly": 5,
        "sunspot_without_missing": 30,
        "temperature_rain_without_missing": 30,
        "vehicle_trips_without_missing": 30,
    }

    def __init__(self,
                 dataset_name: str,
                 size: tuple,
                 path="../dataset/gluonts",
                 dataset_writer=default_dataset_writer,
                 features="S",
                 flag="train",
                 scale=True,
                 ):

        path = Path(path)

        assert dataset_name in self.default_pred_lens.keys(
        ), "{} dataset not recognized".format(dataset_name)

        if size is None:  # Hardcoded behavior, we can change via setting size
            self.seq_len = default_pred_lens[dataset_name] * 2
            self.label_len = 0
            self.pred_len = default_pred_lens[dataset_name]
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        try:  # Tries first to not regenerate, but does it if needed
            self.gluonts_dataset = get_dataset(
                dataset_name=dataset_name,
                path=path,
                regenerate=False,
                dataset_writer=dataset_writer
            )
        except:
            print('Regenerating {}...'.format(dataset_name))
            self.gluonts_dataset = get_dataset(
                dataset_name=dataset_name,
                path=path,
                regenerate=True,
                dataset_writer=dataset_writer
            )

        self.scale = scale
        self.features = features  # "S" - singlevariate or "M" - mulitvariate

        # Will need to do splitting internally or externally after the below steps

        # Getting test gives you all the data. .train is just downsampled version
        x = []
        times = []
        for inp_dict in self.gluonts_dataset.test:
            x.append(inp_dict['target'])
            times.append(inp_dict['start'])
            # May need to look into quicker method if the for loop turns out to be slow

        if self.features == "M":
            # multivariate, dataset becomes just one series, where each series is a new variable
            x = np.stack(x, axis=1)

            # start and end borders for train, val, and test splits
            # start_borders = [0, int(x.shape[0] * 0.8 - self.seq_len), int(x.shape[0] * 0.9 - self.seq_len)]
            # end_borders = [int(x.shape[0] * 0.8), int(x.shape[0] * 0.9), x.shape[0]]
            start_borders = [
                0, int(x.shape[0] * 0.8 - self.seq_len), int(x.shape[0] * 0.8 - self.seq_len)]
            end_borders = [int(x.shape[0] * 0.8), x.shape[0], x.shape[0]]
            set_type = {"train": 0, "val": 1, "test": 2}[flag]
            start = start_borders[set_type]
            end = end_borders[set_type]

            if scale:
                self.scaler = StandardScaler()
                train_start, train_end = start_borders[0], end_borders[0]
                x_train = x[train_start:train_end]
                self.scaler.fit(x_train)
                x = self.scaler.transform(x)

            self.data_x = x[start:end]
            self.data_x = torch.from_numpy(self.data_x)

        elif self.features == "S":
            raise NotImplementedError(
                "need to implement train/test/val split and scaling for single variable")
            self.Xtmp = x
            x_pad, pad_mask = pad_and_stack(x)
            self.data_x = context_based_split(
                x_pad, pad_mask, context_len=self.seq_len + self.pred_len)
            self.data_x = torch.from_numpy(self.data_x)
            if len(self.data_x.shape) == 2:
                # Make multivariate with one sensor
                self.data_x = self.data_x.unsqueeze(-1)

        self.scaler = StandardScaler()

        self.times = times  # Need to transform ------ Not used for now

    def __getitem__(self, ind):
        if self.features == "S":
            # Another thing: there are multple samples, thus we can't just index from data_x like below and in other datasets
            s_begin = 0  # Always 0 bc we collapse down into individual samples
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = self.data_x[ind, s_begin:s_end, :]
            # y is drawn from x - they're the same sequence
            seq_y = self.data_x[ind, r_begin:r_end, :]
            seq_x_mark = torch.zeros_like(seq_x)
            seq_y_mark = torch.zeros_like(seq_y)

        elif self.features == "M":
            s_begin = ind
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_x[r_begin:r_end]
            seq_x_mark = torch.zeros_like(seq_x)
            seq_y_mark = torch.zeros_like(seq_y)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.features == "S":
            return self.data_x.shape[0]
        elif self.features == "M":
            return len(self.data_x) - self.seq_len - self.pred_len + 1


def pad_and_stack(arrays):
    """
    Pads and stacks a list of numpy arrays of varying lengths and creates a boolean array 
    indicating padded elements.

    Args:
    arrays (list of np.array): List of one-dimensional numpy arrays.

    Returns:
    np.array: A two-dimensional numpy array where each original array is padded with zeros
              to match the length of the longest array in the list.
    np.array: A two-dimensional boolean array where True indicates a padded element and 
              False indicates an original element.
    """
    # Find the maximum length among all arrays
    max_len = max(len(a) for a in arrays)

    # Initialize lists to hold padded arrays and boolean arrays
    padded_arrays = []
    boolean_arrays = []

    for a in arrays:
        # Amount of padding needed
        padding = max_len - len(a)

        # Pad the array and add it to the list
        padded_arrays.append(np.pad(a, (0, padding), mode='constant'))

        # Create a boolean array (False for original elements, True for padding)
        boolean_array = np.array([False] * len(a) + [True] * padding)
        boolean_arrays.append(boolean_array)

    # Stack the padded arrays and boolean arrays vertically
    stacked_array = np.vstack(padded_arrays)
    boolean_stacked = np.vstack(boolean_arrays)

    return stacked_array, boolean_stacked


def context_based_split(X, is_pad, context_len: int):
    split_inds = np.arange(start=0, stop=X.shape[1], step=context_len)

    X_collapse = []
    is_pad_collapse = []

    for i in range(1, len(split_inds)):
        X_collapse.append(X[:, split_inds[i-1]:split_inds[i]])
        is_pad_collapse.append(is_pad[:, split_inds[i-1]:split_inds[i]])

    Xnew = np.concatenate(X_collapse, axis=0)
    pad_new = np.concatenate(is_pad_collapse, axis=0)

    pad_by_sample = np.any(pad_new, axis=1)

    return Xnew[~pad_by_sample, :]
