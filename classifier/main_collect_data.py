import os

import django

if not os.getenv('DJANGO_SETTINGS_MODULE'):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'configuration.settings_local')
    django.setup()

from classifier.cnn.data_factory import DataFactory
from classifier.cnn.text_ci import TextClassifierInformation

if __name__ == '__main__':
    data_factory = DataFactory(ci=TextClassifierInformation())
    data_factory.collect_train_data()
    data_factory.serialize_result()
