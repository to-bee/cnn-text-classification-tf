import os

import env_helpers


class ClassifierInformation(object):
    labels = None
    pb_path = None
    pb_graph_path = None
    pb_information_path = None
    pb_labels_path = None
    name = None
    train_steps = None

    def __init__(self, data_path, tmp_path):
        self.data_path = data_path  # Path where data can be found
        self.tmp_path = tmp_path

        resource_path = os.path.join(data_path, 'resources')
        self.resource_path = resource_path
        self.resource_train_path = os.path.join(resource_path, 'train')
        self.resource_test_path = os.path.join(resource_path, 'test')

        self.df_path = os.path.join(self.tmp_path, 'df')
        self.tf_path = os.path.join(self.tmp_path, 'tf')

        checkpoint_path = os.path.join(self.tf_path, 'checkpoints')
        self.checkpoint_path = checkpoint_path  # checkpoint_path
        self.checkpoint_file = os.path.join(checkpoint_path, 'model')  # checkpoint_file

        self.loss_hist_path = os.path.join(self.tf_path, 'loss_history')  # loss_hist_path
        self.train_summary_path = os.path.join(self.tf_path, 'summaries', 'train')  # train_summary_path
        self.cv_summary_path = os.path.join(self.tf_path, 'summaries', 'cv')  # cv_summary_path
        self.test_summary_path = os.path.join(self.tf_path, 'summaries', 'test')  # test_summary_path

        pb_path = os.path.join(data_path, 'pb')
        self.pb_path = pb_path
        self.pb_file = os.path.join(pb_path, env_helpers.get_export_filename(name='weights', ending='pb'))  #
        self.pb_graph_path = os.path.join(pb_path, env_helpers.get_export_filename(name='graph_def', ending='txt'))  #
        self.pb_information_path = os.path.join(pb_path, env_helpers.get_export_filename(name='information', ending='txt'))  #
        self.pb_labels_path = os.path.join(pb_path, env_helpers.get_export_filename(name='labels', ending='txt'))  #

        self.auto_ckpt_interval = 50  # Save checkpoint after number of intervals
        self.auto_snapshot_interval = 500  # Save histogram snapshot after number of intervals

        self.dropout_train = 0.9  # Dropout rate
        self.test_size_rel = 0.1  # Size of the cross validation set in percent
        self.cv_size_rel = 0.1  # Size of the cross validation set in percent
        self.learning_rate = 1e-4  # Learning rate

        self.batch_size = 50  # Batch size

    def get_export_filename(self, ending, name):
        return '%s.%s' % (name, ending)

    @property
    def pb_modified(self):
        return os.path.getmtime(self.pb_path) if os.path.isfile(self.pb_path) else 0

    @property
    def pb_graph_modified(self):
        return os.path.getmtime(self.pb_graph_path) if os.path.isfile(self.pb_graph_path) else 0

    @property
    def pb_information_modified(self):
        return os.path.getmtime(self.pb_information_path) if os.path.isfile(self.pb_information_path) else 0

    @property
    def pb_labels_modified(self):
        return os.path.getmtime(self.pb_labels_path) if os.path.isfile(self.pb_labels_path) else 0


