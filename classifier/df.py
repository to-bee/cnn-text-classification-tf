import datetime
import os
import random
from abc import ABC
from typing import List

import numpy as np
from sklearn.utils import shuffle

from classifier import env_helpers
from classifier.cnn.text_ci import TextClassifierInformation

current_year = datetime.datetime.now().year


class BaseDataFrame(ABC):
    vocab_processor = None
    vocab_size: int
    raw_data: List
    data_x: np.ndarray
    data_y: np.ndarray

    def __init__(self, ci: TextClassifierInformation, name, restore):
        self.ci = ci
        self.name = name
        np.random.seed(10)

        if restore:
            self.load_dataset()
        else:
            self.init_df()

    def init_df(self):
        env_helpers.create_directory(self.ci.df_path)
        self.vocab_size = None
        self.raw_data = []
        self.data_x = None
        self.data_y = np.empty([0, self.ci.y_len])

    def get_pickle_filename(self):
        vocab_size = os.path.join(self.ci.df_path, f"cname={self.ci.name},name={self.name}-vocab_size.npy")
        fn_raw = os.path.join(self.ci.df_path, f"cname={self.ci.name},name={self.name}-raw.npy")
        fn_x = os.path.join(self.ci.df_path, f"cname={self.ci.name},name={self.name}-x.npy")
        fn_y = os.path.join(self.ci.df_path, f"cname={self.ci.name},name={self.name}-y.npy")
        fn_vocab = os.path.join(self.ci.df_path, f"cname={self.ci.name},name={self.name}-vocab")
        return vocab_size, fn_raw, fn_x, fn_y, fn_vocab

    def show_summary(self):
        print('')
        print(f'Dataset "{self.name}"')
        print(f"Totally {len(self.raw_data)} samples")
        for c in range(self.ci.y_len):
            print("Class %s: %s samples" % (c, list(self.data_y[:, c]).count(1)))

    def load_dataset(self):
        fn_vocab_size, fn_raw, fn_x, fn_y, fn_vocab = self.get_pickle_filename()
        if os.path.isfile(fn_x) and os.path.isfile(fn_y):
            self.vocab_size = int(np.load(fn_vocab_size))
            self.raw_data = np.load(fn_raw)
            self.data_x = np.load(fn_x)
            self.data_y = np.load(fn_y)

            from tensorflow.contrib.learn.python.learn.preprocessing import VocabularyProcessor
            self.vocab_processor = VocabularyProcessor.restore(fn_vocab)
        else:
            self.init_df()

    def save_dataset(self):
        fn_vocab_size, fn_raw, fn_x, fn_y, fn_vocab = self.get_pickle_filename()
        np.save(fn_vocab_size, self.vocab_size)
        np.save(fn_raw, self.raw_data)
        np.save(fn_x, self.data_x)
        np.save(fn_y, self.data_y)

        if self.vocab_processor:
            self.vocab_processor.save(fn_vocab)

        print(f'Dataset "{self.name}" saved')

    def randomize(self):
        print('Randomize data..')

        # Not working with mixed types (List, ndarray)
        # combined = list(zip(self.data_x, self.data_y, self.raw_data))
        # random.shuffle(combined)
        # self.data_x[:], self.data_y[:], self.raw_data[:] = zip(*combined)

        self.data_y, self.data_x, self.raw_data = shuffle(self.data_y, self.data_x, self.raw_data)

    def get_row(self, idx):
        return self.data_x[idx, :], self.data_y[idx, :]

    def has_data(self):
        return self.data_x is not None and self.data_x.shape[0] > 0

    def get_data(self):
        return self.vocab_size, self.raw_data, self.data_x, self.data_y

    def split_train_test(self, train_size_rel, name_df_1, name_df_2):
        num_samples = len(self.data_y)
        train_samples = int(num_samples * train_size_rel)

        train_raw = self.raw_data[:train_samples]
        train_x = self.data_x[:train_samples]
        train_y = self.data_y[:train_samples]

        test_raw = self.raw_data[train_samples:]
        test_x = self.data_x[train_samples:]
        test_y = self.data_y[train_samples:]

        df_train = BaseDataFrame(self.ci, name_df_1, False)
        df_train.vocab_size = self.vocab_size
        df_train.raw_data = train_raw
        df_train.data_x = train_x
        df_train.data_y = train_y
        df_train.vocab_processor = self.vocab_processor
        df_train.save_dataset()

        df_test = BaseDataFrame(self.ci, name_df_2, False)
        df_test.vocab_size = self.vocab_size
        df_test.raw_data = test_raw
        df_test.data_x = test_x
        df_test.data_y = test_y
        df_test.vocab_processor = self.vocab_processor
        df_test.save_dataset()

        return df_train, df_test
