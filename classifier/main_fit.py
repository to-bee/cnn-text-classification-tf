import os

import django

if not os.getenv('DJANGO_SETTINGS_MODULE'):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'configuration.settings_local')
    django.setup()

from classifier.cnn.text_ci import TextClassifierInformation
from classifier.cnn.text_cnn_df import TextDataFrame
from classifier.cnn.text_cnn_trainer import TextCnnClassifierTrainer
from classifier.pb import Pb

if __name__ == '__main__':
    ci = TextClassifierInformation()

    restore_df = False
    path = ci.resource_train_path
    df_train = TextDataFrame(ci, name='df1_train', restore=True)
    df_train.show_summary()

    df_test = TextDataFrame(ci, name='df1_test', restore=True)
    df_test.show_summary()

    pb = Pb(ci=ci)
    cnn = TextCnnClassifierTrainer(df_train=df_train, df_test=df_test, ci=ci, pb=pb, restore_model=False)
    (global_step, ckpt, final_test_accuracy, final_test_cost) = cnn.fit()
    pb.save(ckpt, final_test_accuracy, final_test_cost)
