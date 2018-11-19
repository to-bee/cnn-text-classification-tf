import re
from typing import List

import numpy as np
from bs4 import BeautifulSoup
from django.db.models import QuerySet
from nltk import sent_tokenize

from classifier.cnn.text_ci import TextClassifierInformation
from classifier.cnn.text_cnn_df import TextDataFrame
from monitoring import models
from monitoring.models import Email, LABEL_SPAM


class DataFactory(object):
    def __init__(self, ci: TextClassifierInformation):
        self.ci = ci
        self.df = TextDataFrame(self.ci, name='df1_train', restore=False)

    def collect_data(self)->TextDataFrame:
        spam_mails_qs: QuerySet = Email.objects.filter(human_annotations__classification=LABEL_SPAM, text_plain__isnull=False)
        spam_mails: List[str] = self.preprocess_samples(qs=spam_mails_qs, label=LABEL_SPAM)
        self.add_samples(raw_data=spam_mails, label_key=models.LABEL_SPAM)

        ham_mails_qs: QuerySet = Email.objects.filter(human_annotations__classification=models.LABEL_HAM, text_plain__isnull=False)
        ham_mails: List[str] = self.preprocess_samples(qs=ham_mails_qs, label=models.LABEL_HAM)
        self.add_samples(raw_data=ham_mails, label_key=models.LABEL_HAM)

        update_mails_qs: QuerySet = Email.objects.filter(human_annotations__classification=models.LABEL_UPDATES, text_plain__isnull=False)
        update_mails: List[str] = self.preprocess_samples(qs=update_mails_qs, label=models.LABEL_UPDATES)
        self.add_samples(raw_data=update_mails, label_key=models.LABEL_UPDATES)

        self.vectorize_data()

        return self.df

    def add_samples(self, raw_data: List[str], label_key: str):
        label_id = self.ci.label2id(key=label_key)
        one_hot_label = self.one_hot(mapping=label_id)
        data_y = [one_hot_label for _ in raw_data]

        self.df.add_rows(raw_data=raw_data, data_y=data_y)

    def preprocess_samples(self, qs: QuerySet, label) -> List[str]:
        text_plain_qs = qs.values_list('text_plain', flat=True)

        textes = list(text_plain_qs)

        # Drop html text
        textes = [text for text in textes if not self.has_html(text)]

        # Replace urls in text
        textes = [self.replace_urls(text) for text in textes]

        # Tokenize text
        lines = []
        for text in textes:
            tokens = sent_tokenize(text)
            if tokens:
                # Remove tokens shorter than 3 characters
                tokens = [token for token in tokens if len(token) > 3]
                lines.extend(tokens)

        lines = [self._clean_str(line) for line in lines]
        lines = [line.strip() for line in lines]

        return lines

    def serialize_result(self):
        self.df.save_dataset()

    def _clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    URL_REGEX = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    def has_url(self, text: str) -> bool:
        return re.match(self.URL_REGEX, text) is not None

    def has_html(self, text: str) -> bool:
        return bool(BeautifulSoup(text, "html.parser").find())
        # has_html2 = '! doctype html public w3c' in text

    # def encode_categories(self, row_values):
    #     label_encoder = LabelEncoder()
    #     integer_encoded = label_encoder.fit_transform(row_values)
    #     return integer_encoded.reshape(len(integer_encoded), 1)

    def one_hot(self, mapping):
        """
        Sets the i'th value to 1
        :param mapping:
        :return:
        """
        y = np.zeros((self.ci.y_len))
        y[mapping] = 1
        return y

    def replace_urls(self, line):
        return re.sub(r'^https?:\/\/.*[\r\n]*', '', line, flags=re.MULTILINE)

    def vectorize_data(self):
        self.df.vectorize_data()
