from typing import List

from django.db.models import QuerySet
from tensorflow.contrib.learn.python.learn.preprocessing import VocabularyProcessor

from classifier.cnn.data_factory.data_factory import DataFactory
from classifier.cnn.text_ci import TextClassifierInformation
from classifier.cnn.text_cnn_df import TextDataFrame
from monitoring import models
from monitoring.models import Email, LABEL_SPAM
import numpy as np

class TrainSetFactory(DataFactory):
    def __init__(self, ci: TextClassifierInformation):
        super(TrainSetFactory, self).__init__(ci)
        self.df = TextDataFrame(self.ci, name='df1_train', restore=False)

    def collect_data(self) -> TextDataFrame:
        spam_mails_qs: QuerySet = Email.objects.filter(human_annotations__classification=LABEL_SPAM, text_plain__isnull=False)
        spam_mails: List[str] = self.preprocess_samples(qs=spam_mails_qs)
        self.add_samples(raw_data=spam_mails, label_key=models.LABEL_SPAM)

        ham_mails_qs: QuerySet = Email.objects.filter(human_annotations__classification=models.LABEL_HAM, text_plain__isnull=False)
        ham_mails: List[str] = self.preprocess_samples(qs=ham_mails_qs)
        self.add_samples(raw_data=ham_mails, label_key=models.LABEL_HAM)

        update_mails_qs: QuerySet = Email.objects.filter(human_annotations__classification=models.LABEL_UPDATES, text_plain__isnull=False)
        update_mails: List[str] = self.preprocess_samples(qs=update_mails_qs)
        self.add_samples(raw_data=update_mails, label_key=models.LABEL_UPDATES)

        self.vectorize_data()

        return self.df

    def preprocess_samples(self, qs: QuerySet) -> List[str]:
        text_plain_qs = qs.values_list('text_plain', flat=True)

        textes = list(text_plain_qs)

        # Drop html text
        textes = [text for text in textes if not self.has_html(text)]

        # Replace urls in text
        textes = [self.replace_urls(text) for text in textes]

        # Tokenize text
        lines = []
        from nltk import sent_tokenize

        for text in textes:
            tokens = sent_tokenize(text)
            if tokens:
                # Remove tokens shorter than 3 characters
                tokens = [token for token in tokens if len(token) > 3]
                lines.extend(tokens)

        lines = [self._clean_str(line) for line in lines]
        lines = [line.strip() for line in lines]

        return lines

    def vectorize_data(self):
        raw_data = self.df.raw_data
        max_document_length = max([len(line.split(" ")) for line in self.df.raw_data])
        vocab_processor = VocabularyProcessor(max_document_length=max_document_length)

        data_x = np.array(list(vocab_processor.fit_transform(raw_data)))
        vocab_size = len(vocab_processor.vocabulary_)

        self.df.vocab_processor = vocab_processor
        self.df.data_x = data_x
        self.df.vocab_size = vocab_size
        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))