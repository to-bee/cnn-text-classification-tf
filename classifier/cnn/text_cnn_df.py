from typing import List

import numpy as np
from tensorflow.contrib import learn

from classifier.cnn.text_ci import TextClassifierInformation
from classifier.df import BaseDataFrame


class SkipRow(Exception):
    pass


class TextDataFrame(BaseDataFrame):
    def __init__(self, ci: TextClassifierInformation, name, restore):
        super(TextDataFrame, self).__init__(ci, name, restore)

    def add_rows(self, raw_data: List[str], data_y: List):
        print('Appending rows...')
        self.raw_data.extend(raw_data)
        self.data_y = np.vstack([self.data_y, np.array(data_y)])
        print(f'{len(raw_data)} rows appended. Totally {len(self.raw_data)} rows available.')

    def vectorize_data(self):
        # Build vocabulary
        max_document_length = max([len(line.split(" ")) for line in self.raw_data])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_document_length)
        self.data_x = np.array(list(vocab_processor.fit_transform(self.raw_data)))
        self.vocab_size = len(vocab_processor.vocabulary_)

        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

    # def preprocess(self):
    #     # Data Preparation
    #     # ==================================================
    #
    #     # Load data
    #     print("Loading data...")
    #     raw_data, data_x, data_y = self.get_data()
    #
    #
    #
    #     # Randomly shuffle data
    #     np.random.seed(10)
    #     shuffle_indices = np.random.permutation(np.arange(len(data_y)))
    #     x_shuffled = x[shuffle_indices]
    #     y_shuffled = data_y[shuffle_indices]
    #
    #     # Split train/test set
    #     # TODO: This is very crude, should use cross-validation
    #     dev_sample_percentage = 0.1
    #     dev_sample_index = -1 * int(dev_sample_percentage * float(len(data_y)))
    #     x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    #     y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    #
    #     del x, data_y, x_shuffled, y_shuffled
    #
    #     print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    #     print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    #     # return x_train, y_train, vocab_processor, x_dev, y_dev
    #     return vocab_processor