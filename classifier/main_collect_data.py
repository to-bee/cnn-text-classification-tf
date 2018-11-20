import os

import django

if not os.getenv('DJANGO_SETTINGS_MODULE'):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'configuration.settings_local')
    django.setup()
from classifier.cnn.data_factory.trainset_factory import TrainSetFactory

from classifier.cnn.text_cnn_df import TextDataFrame

from classifier.cnn.text_ci import TextClassifierInformation

if __name__ == '__main__':
    data_factory = TrainSetFactory(ci=TextClassifierInformation())
    df: TextDataFrame = data_factory.collect_data()
    df.randomize()

    df_train, df_test = df.split_train_test(0.9, name_df_1='df1_train', name_df_2='df1_test')

    df_train.show_summary()
    df_train.save_dataset()

    df_test.show_summary()
    df_test.save_dataset()
