import tensorflow as tf

from classifier.base_trainer import ClassifierTrainer
from classifier.df import TextDataFrame
from classifier.cnn.text_ci import TextClassifierInformation
from classifier.cnn.text_cnn_model import TextCNN
from classifier.pb import Pb
from eval import vocab_processor


class PlateCnnClassifierTrainer(ClassifierTrainer):
    def __init__(self, props: TextClassifierInformation, pb: Pb, restore_model, restore_ckpt=None):
        super(PlateCnnClassifierTrainer, self).__init__(props, pb, restore_model, restore_ckpt)

    def fit(self, df_train: TextDataFrame, df_test):
        dropout_train = self.props.dropout_train
        train_steps = self.props.train_steps

        print("\n-------------- TRAIN --------------")
        print('dropout: %f, train_steps: %d' % (dropout_train, train_steps))

        df_train.randomize()

        df_train, df_cv = df_train.split_train_test(train_size_rel=1 - self.props.test_size_rel)

        train_x, train_y = df_train.get_data()
        cv_x, cv_y = df_cv.get_data()
        test_x, test_y = df_test.get_data()

        cv_cost = None
        test_cost = None
        global_step = self.sess.run(self.model.global_step)
        while global_step < train_steps:
            self.save_model(self.sess, self.saver, global_step)

            (batch_y, batch_x) = self.random_batch(train_y, train_x)
            # (batch_y, batch_x) = self.next_batch(data_y=train_y, data_x=train_x, step=global_step)
            feed_dict_train = {self.model.x: batch_x, self.model.y_true: batch_y, self.model.keep_prob: dropout_train}
            [global_step, _, _] = self.sess.run([self.model.global_step, self.model.train_step, self.model.summ], feed_dict=feed_dict_train)

            if global_step % 100 == 0:
                print('step: %d' % global_step)

                train_accuracy, train_cost, train_precision, train_recall, train_f1, train_summ = self.get_summary(batch_x, batch_y, dropout_train)
                cv_accuracy, cv_cost, cv_precision, cv_recall, cv_f1, cv_summ = self.get_summary(cv_x, cv_y, 1.0)
                test_accuracy, test_cost, test_precision, test_recall, test_f1, test_summ = self.get_summary(test_x, test_y, 1.0)

                self.train_summary_writer.add_summary(train_summ, global_step)
                self.cv_summary_writer.add_summary(cv_summ, global_step)
                self.test_summary_writer.add_summary(test_summ, global_step)

                print('')
                print('train accuracy %g' % (train_accuracy))
                print('train precision %g' % (train_precision))
                print('train recall %g' % (train_recall))
                print('train f1 %g' % (train_f1))

                print('')
                print('cv accuracy %g' % (cv_accuracy))
                print('cv precision %g' % (cv_precision))
                print('cv recall %g' % (cv_recall))
                print('cv f1 %g' % (cv_f1))

                print('')
                print('testset accuracy %g' % (test_accuracy))
                print('testset precision %g' % (test_precision))
                print('testset recall %g' % (test_recall))
                print('testset f1 %g' % (test_f1))

            if global_step % self.props.auto_snapshot_interval == 0:
                self.global_step_hist.append(global_step)
                self.cv_cost_hist.append(cv_cost)
                self.test_cost_hist.append(test_cost)
                self.save_loss_hists()

        self.save_model(self.sess, self.saver, global_step, force=True)
        tf.reset_default_graph()

        test_accuracy, test_cost, test_precision, test_recall, test_f1, test_summ = self.get_summary(test_x, test_y, 1.0)
        print('training ended - final test accuracy: %f, final test cost: %f' % (test_accuracy, test_cost))

        best_ckpt = self.evaluate_best_checkpoint() if global_step >= train_steps else None
        return (global_step, best_ckpt, test_accuracy, test_cost)

    def get_summary(self, x, y, dropout):
        # self.sess.run(tf.local_variables_initializer())

        return self.sess.run([
            self.model.accuracy_op,
            self.model.cost,
            self.model.recall_op,
            self.model.precision_op,
            self.model.f1_score_op,
            self.model.summ,
        ], feed_dict={self.model.x: x, self.model.y_true: y, self.model.keep_prob: dropout})

    def load_model(self)->TextCNN:
        print('loading model')
        return TextCNN(
            sequence_length=self.props.x_len,
            num_classes=self.props.y_len,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=128,
            filter_sizes=list(map(int, [3, 4, 5])),
            num_filters=128,
            l2_reg_lambda=self.props.l2_reg_lambda,
            learning_rate=self.props.learning_rate
        )