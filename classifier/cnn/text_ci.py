import os

import numpy as np

from classifier.base_ci import ClassifierInformation
from monitoring.models import LABEL_SPAM, LABEL_HAM, LABEL_UPDATES


class TextClassifierInformation(ClassifierInformation):
    name = 'text_cnn'
    train_steps = 200
    dropout_train = 0.5  # Dropout rate
    l2_reg_lambda = 0.0  # Lambda regularization

    ham_data_file = './data/ham.txt'
    spam_data_file = './data/spam.txt'
    promotions_data_file = './data/promotions.txt'

    def __init__(self):
        data_path = os.path.join(f'/Users/tobi/repo/anpr/data/{self.name}')
        tmp_path = os.path.join(f'/var/anpr/{self.name}')

        super(TextClassifierInformation, self).__init__(data_path=data_path, tmp_path=tmp_path)

        self.train_steps = 12000  # Number of train steps
        self.label_dict = {
            LABEL_SPAM: 0,
            LABEL_HAM: 1,
            LABEL_UPDATES: 2,
        }

        self.labels = self.label_dict.values()

        self.image_shape = (20, 60, 1)
        self.image_height = self.image_shape[0]
        self.image_width = self.image_shape[1]  # Width of one field
        self.image_channels = self.image_shape[2]  # Channels of one field

        self.x_len = np.prod(self.image_shape)
        self.y_len = len(self.labels)

        self.plate_colors = range(200, 255)
        self.text_colors = range(0, 50)

    def label2id(self, key: str):
        return self.label_dict.get(key)

    def id2label(self, id: int):
        result = [key for (key, value) in self.label_dict.items() if value == id]
        if result:
            return result[0]
