import datetime
import os
import random
from abc import ABC
from typing import List

import numpy as np
from pandas import DataFrame

from classifier import env_helpers
from classifier.cnn.text_ci import TextClassifierInformation

current_year = datetime.datetime.now().year


class BaseDataFrame(ABC):
    raw_data: List
    __data_x: np.ndarray
    data_y: np.ndarray

    def __init__(self, ci: TextClassifierInformation, name, restore):
        self.ci = ci
        self.name = name
        np.random.seed(10)

        if restore:
            self.load_dataset()
        else:
            self.init_df()

    @property
    def data_x(self):
        if self.__data_x is None:
            raise Exception('Call vocalprocessor before accessing data_x')

    def init_df(self):
        env_helpers.create_directory(self.ci.df_path)
        self.raw_data = []
        self.__data_x = None
        self.data_y = np.empty([0, self.ci.y_len])

    def get_pickle_filename(self):
        fn_raw = os.path.join(self.ci.df_path, f"cname={self.ci.name},name={self.name}-raw.npy")
        fn_x = os.path.join(self.ci.df_path, f"cname={self.ci.name},name={self.name}-x.npy")
        fn_y = os.path.join(self.ci.df_path, f"cname={self.ci.name},name={self.name}-y.npy")
        return fn_raw, fn_x, fn_y

    def show_summary(self):
        print('')
        print(f'Dataset "{self.name}"')
        print(f"Totally {len(self.raw_data)} samples")
        for c in range(self.ci.y_len):
            print("Class %s: %s samples" % (c, list(self.data_y[:, c]).count(1)))

    def load_dataset(self):
        fn_raw, fn_x, fn_y = self.get_pickle_filename()
        if os.path.isfile(fn_x) and os.path.isfile(fn_y):
            self.raw_data = np.load(fn_raw)
            self.__data_x = np.load(fn_x)
            self.data_y = np.load(fn_y)
        else:
            self.init_df()

    def save_dataset(self):
        fn_raw, fn_x, fn_y = self.get_pickle_filename()
        np.save(fn_raw, self.raw_data)
        np.save(fn_x, self.__data_x)
        np.save(fn_y, self.data_y)
        print(f'Dataset "{self.name}" saved')

    def randomize(self):
        print('Randomize data..')

        combined = list(zip(self.raw_data, self.data_x, self.data_y))
        random.shuffle(combined)

        self.raw_data[:], self.data_x[:], self.data_y[:] = zip(*combined)

    def get_row(self, idx):
        return self.data_x[idx, :], self.data_y[idx, :]

    def has_data(self):
        return self.data_x is not None and self.data_x.shape[0] > 0

    def get_data(self):
        return self.data_x, self.data_y

    def split_train_test(self, train_size_rel):
        num_samples = len(self.data_y)
        train_samples = int(num_samples * train_size_rel)

        train_x = self.data_x[:train_samples]
        train_y = self.data_y[:train_samples]

        test_x = self.data_x[train_samples:]
        test_y = self.data_y[train_samples:]

        df_train = DataFrame(self.ci, 'train', False)
        df_train.data_x = train_x
        df_train.data_y = train_y
        df_train.save_dataset()

        df_test = DataFrame(self.ci, 'test', False)
        df_test.data_x = test_x
        df_test.data_y = test_y
        df_test.save_dataset()

        return df_train, df_test
