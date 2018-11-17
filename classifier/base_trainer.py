import os
from abc import abstractmethod
from random import randint

import numpy as np
import tensorflow as tf

from classifier import env_helpers
from classifier.cnn.text_ci import TextClassifierInformation
from classifier.pb import Pb
from classifier.cnn.text_cnn_model import TextCNN


class ClassifierTrainer:
    props: TextClassifierInformation = None

    def __init__(self, props: TextClassifierInformation, pb: Pb, restore_model, restore_ckpt=None):
        tf.reset_default_graph()

        self.props = props
        self.pb = pb

        if not restore_model:
            self.clear_train_data()
            self.delete_loss_hist()

        env_helpers.create_directory(self.props.tf_path)
        env_helpers.create_directory(self.props.loss_hist_path)
        env_helpers.create_directory(self.props.train_summary_path)
        env_helpers.create_directory(self.props.train_summary_path)
        env_helpers.create_directory(self.props.test_summary_path)

        env_helpers.create_directory(self.props.pb_path)
        env_helpers.create_directory(self.props.checkpoint_path)
        env_helpers.create_directory(self.props.loss_hist_path)
        env_helpers.create_directory(self.props.train_summary_path)
        env_helpers.create_directory(self.props.cv_summary_path)
        env_helpers.create_directory(self.props.test_summary_path)

        self.model = self.load_model()
        self.saver = tf.train.Saver(max_to_keep=100)
        self.sess = self.init_model(saver=self.saver, restore=restore_model, ckpt=restore_ckpt)

        self.train_summary_writer = tf.summary.FileWriter(self.props.train_summary_path, self.sess.graph)
        self.cv_summary_writer = tf.summary.FileWriter(self.props.cv_summary_path, self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(self.props.test_summary_path, self.sess.graph)

        self.pb.save_graph_def(self.sess)
        self.load_loss_hists(restore=restore_model)

    @abstractmethod
    def load_model(self) -> TextCNN:
        pass

    def random_batch(self, data_y, data_x):
        # shuffle_y, shuffle_x = sklearn.utils.shuffle(data_y, data_x, n_samples=self.ci.batch_size)

        batch_size = self.props.batch_size
        (rows, cols) = data_x.shape
        batch_from = randint(0, rows)
        batch_to = batch_from + batch_size

        if (batch_from <= batch_to):
            y_data = data_y[batch_from:batch_to, :]
            x_data = data_x[batch_from:batch_to, :]
        else:
            y_data1 = data_y[batch_from:rows, :]
            y_data2 = data_y[0:batch_to, :]
            y_data = np.vstack([y_data1, y_data2])
            x_data1 = data_x[batch_from:rows, :]
            x_data2 = data_x[0:batch_to, :]
            x_data = np.vstack([x_data1, x_data2])

        return (y_data, x_data)

    def next_batch(self, data_y, data_x, step):
        batch_size = self.props.batch_size
        (rows, cols) = data_x.shape
        if (rows < batch_size):
            return (data_y, data_x)

        # np.random.shuffle(data)

        batch_from = (step * batch_size) % rows
        batch_to = ((step + 1) * batch_size) % rows
        # if(batch_to > rows):
        #     batch_to = rows

        # When i*size reachs the end - take one part at the end and another part from zero position
        if (batch_from <= batch_to):
            y_data = data_y[batch_from:batch_to, :]
            x_data = data_x[batch_from:batch_to, :]
        else:
            y_data1 = data_y[batch_from:rows, :]
            y_data2 = data_y[0:batch_to, :]
            y_data = np.vstack([y_data1, y_data2])
            x_data1 = data_x[batch_from:rows, :]
            x_data2 = data_x[0:batch_to, :]
            x_data = np.vstack([x_data1, x_data2])

        return (y_data, x_data)

    def save_model(self, sess, saver, global_step=None, force=False):
        if global_step % self.props.auto_ckpt_interval == 0 or force is True:
            saver.save(sess, self.props.checkpoint_file)
            # print("model saved")
        if global_step % self.props.auto_snapshot_interval == 0 or force is True:
            save_path = saver.save(sess, self.props.checkpoint_file, global_step=global_step)
            print("model saved to: %s" % save_path)

    def evaluate_best_checkpoint(self):
        has_hist = len(self.cv_cost_hist) > 0 and len(self.test_cost_hist) > 0
        if has_hist:
            # evaluate best result
            cv_hist_np = np.array(self.cv_cost_hist)
            test_hist_np = np.array(self.test_cost_hist)
            min_cv_cost_idx = np.argmin(cv_hist_np)
            min_cv_cost = np.min(cv_hist_np)
            min_test_cost_idx = np.argmin(test_hist_np)
            min_test_cost = np.min(test_hist_np)
            # if min_cv_cost_idx < min_test_cost_idx:
            #     min_cost_idx = min_cv_cost_idx
            #     min_loss = min_cv_cost
            # else:
            #     min_cost_idx = min_test_cost_idx
            #     min_loss = min_test_cost

            min_cost_idx = int(min_test_cost_idx)
            min_loss = min_test_cost

            best_step = self.global_step_hist[min_cost_idx]
            print('best checkpoint found at step %s, index %s with test loss %s' % (best_step, min_cost_idx + 1, min_loss))
            return best_step

        else:
            print('not enough checkpoints available to evaluate the best')
            return None

    def predict(self, x_batch, dropout):
        """
        Old prediction without loading pb file
        :param x_batch:
        :param dropout:
        :return:
        """
        # (train_x, cv_x, train_y, cv_y) = df.split_train_cv(0.1)
        probabilities = self.sess.run(self.y_pred, feed_dict={self.x: x_batch, self.keep_prob: dropout})
        return probabilities

    def load_loss_hists(self, restore=True):
        self.global_step_hist = self.load_loss_hist(list_name='global_step_hist', restore=restore)
        self.cv_cost_hist = self.load_loss_hist(list_name='cv_cost_hist', restore=restore)
        self.test_cost_hist = self.load_loss_hist(list_name='test_cost_hist', restore=restore)

    def load_loss_hist(self, list_name, restore):
        path = self.get_pickle_filename(list_name=list_name) + '.npy'
        if restore is True and os.path.isfile(path):
            return np.load(path).tolist()
        else:
            return []

    def save_loss_hist(self, list_name, list):
        path = self.get_pickle_filename(list_name=list_name)
        np.save(path, list)

    def save_loss_hists(self):
        self.save_loss_hist(list_name='global_step_hist', list=self.global_step_hist)
        self.save_loss_hist(list_name='cv_cost_hist', list=self.cv_cost_hist)
        self.save_loss_hist(list_name='test_cost_hist', list=self.test_cost_hist)

    def get_pickle_filename(self, list_name):
        return os.path.join(self.props.loss_hist_path, '%s' % (list_name))

    def clear_train_data(self):
        print("model folder truncated")
        env_helpers.clear_folder(dir=self.props.tf_path)

    def delete_loss_hist(self):
        env_helpers.clear_folder(dir=self.props.loss_hist_path)

    def init_model(self, saver, restore, ckpt=None):
        # tf.reset_default_graph()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        try:
            if restore:
                restore_latest = ckpt is None
                if ckpt is not None and ckpt % self.props.auto_ckpt_interval != 0:
                    print('checkpoint snapshots are made all %d steps. %d cannot be restored. Restoring latest checkpoint...' % (self.props.auto_ckpt_interval, ckpt))
                    restore_latest = True

                if restore_latest:
                    saver.restore(sess, tf.train.latest_checkpoint(self.props.checkpoint_path))
                    print("latest checkpoint restored")
                else:
                    path = '%s-%d' % (self.props.checkpoint_file, ckpt)
                    tf.reset_default_graph()
                    saver.restore(sess, path)
                    print('checkpoint %d restored' % ckpt)
            else:
                sess.run(tf.global_variables_initializer())
        except:
            print('cannot restore checkpoint')
            sess.run(tf.global_variables_initializer())

        return sess

    def restore(self, sess):
        snapshot_file = os.path.join(self.props.tf_path, 'variables.ckpt')
        self.saver.restore(sess, snapshot_file)
        print("Model restored.")

    def save(self, sess):
        snapshot_file = os.path.join(self.props.tf_path, 'variables.ckpt')
        save_path = self.saver.save(sess, snapshot_file)
        print("Model saved in file: %s" % save_path)
