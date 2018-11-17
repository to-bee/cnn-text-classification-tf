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
        self._data_x = np.array(list(vocab_processor.fit_transform(self.raw_data)))

        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
