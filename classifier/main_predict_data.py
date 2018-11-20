import os

import django

if not os.getenv('DJANGO_SETTINGS_MODULE'):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'configuration.settings_local')
    django.setup()

from monitoring import models

from classifier.cnn.data_factory.testset_factory import TestSetFactory
from classifier.cnn.text_ci import TextClassifierInformation
from classifier.cnn.text_cnn_df import TextDataFrame
from classifier.pb import Pb
from classifier.predictor import ClassifierPredictor

import numpy as np


def predict_data():
    ci = TextClassifierInformation()

    df_train = TextDataFrame(ci, name='df1_train', restore=True)
    vocab_processor = df_train.vocab_processor

    data_factory = TestSetFactory(ci=TextClassifierInformation(), vocab_processor=vocab_processor)
    df: TextDataFrame = data_factory.collect_data()
    df.randomize()

    df.show_summary()
    df.save_dataset()

    data_x = df.data_x

    predictor = ClassifierPredictor(ci=ci, pb=Pb(ci=ci))
    probs = predictor.fetch_probs(x_batch=np.matrix(data_x))
    for prob_row, raw_text in zip(probs, df.raw_data):
        prediction_idx = int(np.argmax(prob_row))
        prediction = ci.id2label(prediction_idx)
        prob_row_str = ', '.join(['{0:1.3f}'.format(prob) for prob in prob_row])
        print(f'Text: {raw_text}')
        print(f'Predicted label: {prediction}')
        print(f'Probabilities for ({models.LABEL_SPAM},{models.LABEL_HAM},{models.LABEL_UPDATES}): {prob_row_str}')
        print('')


if __name__ == "__main__":
    predict_data()
    print()
