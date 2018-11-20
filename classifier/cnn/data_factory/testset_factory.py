import numpy as np

from classifier.cnn.data_factory.data_factory import DataFactory
from classifier.cnn.text_ci import TextClassifierInformation
from classifier.cnn.text_cnn_df import TextDataFrame
from monitoring import models


class TestSetFactory(DataFactory):
    def __init__(self, ci: TextClassifierInformation, vocab_processor):
        super(TestSetFactory, self).__init__(ci)
        self.df = TextDataFrame(self.ci, name='pb_testset', restore=False)
        self.vocab_processor = vocab_processor

    def collect_data(self) -> TextDataFrame:
        textes_spam = [
            'Möchten Sie gerne Viagra Pillen kaufen?',
            'So günstig ist es nie wieder.',
        ]
        self.add_samples(raw_data=textes_spam, label_key=models.LABEL_SPAM)

        textes_ham = [
            'Hallo Peter. Kann ich dich morgen besuchen?',
            'Ich wäre diesen Freitag ab 20.00 Uhr verfügbar.',
        ]
        self.add_samples(raw_data=textes_ham, label_key=models.LABEL_HAM)

        textes_updates = [
            'Der Turnverein Eien-Kleindöttingen schliesst den Wettkampf mit hervorragenden Ergebnissen ab.',
            'Die Turnerin hat konnte sogar den Schweizermeistertitel für sich gewinnen.',
        ]
        self.add_samples(raw_data=textes_updates, label_key=models.LABEL_UPDATES)

        self.vectorize_data()

        return self.df

    def vectorize_data(self):
        raw_data = self.df.raw_data
        max_document_length = max([len(line.split(" ")) for line in self.df.raw_data])

        data_x = np.array(list(self.vocab_processor.transform(raw_data)))

        vocab_size = len(self.vocab_processor.vocabulary_)

        self.df.vocab_processor = self.vocab_processor
        self.df.data_x = data_x
        self.df.vocab_size = vocab_size

        print("Vocabulary Size: {:d}".format(len(self.vocab_processor.vocabulary_)))
