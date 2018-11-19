import os

from classifier.base_ci import ClassifierInformation
from monitoring.models import LABEL_SPAM, LABEL_HAM, LABEL_UPDATES


class TextClassifierInformation(ClassifierInformation):
    name = 'text_cnn'
    train_steps = 4000
    dropout_train = 0.5  # Dropout rate
    l2_reg_lambda = 0.0  # Lambda regularization

    ham_data_file = './data/ham.txt'
    spam_data_file = './data/spam.txt'
    promotions_data_file = './data/promotions.txt'

    def __init__(self):
        data_path = os.path.join(f'/var/postfix_monitoring/ml/data/{self.name}')
        tmp_path = os.path.join(f'/var/postfix_monitoring/ml/tmp/{self.name}')

        super(TextClassifierInformation, self).__init__(data_path=data_path, tmp_path=tmp_path)

        self.label_dict = {
            LABEL_SPAM: 0,
            LABEL_HAM: 1,
            LABEL_UPDATES: 2,
        }

        self.labels = self.label_dict.values()
        self.y_len = len(self.labels)

    def label2id(self, key: str):
        return self.label_dict.get(key)

    def id2label(self, id: int):
        result = [key for (key, value) in self.label_dict.items() if value == id]
        if result:
            return result[0]
