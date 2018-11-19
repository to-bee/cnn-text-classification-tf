import tensorflow as tf

from classifier.base_trainer import ClassifierTrainer
from classifier.cnn.text_ci import TextClassifierInformation
from classifier.cnn.text_cnn_df import TextDataFrame
from classifier.cnn.text_cnn_model import TextCNN
from classifier.pb import Pb


class TextCnnClassifierTrainer(ClassifierTrainer):
    def __init__(self, df_train: TextDataFrame, df_test: TextDataFrame, ci: TextClassifierInformation, pb: Pb, restore_model, restore_ckpt=None):
        self.df_train = df_train
        self.df_test = df_test

        super(TextCnnClassifierTrainer, self).__init__(ci=ci, pb=pb, restore_model=restore_model, restore_ckpt=restore_ckpt)

    def fit(self):

        dropout_train = self.ci.dropout_train
        train_steps = self.ci.train_steps

        print("\n-------------- TRAIN --------------")
        print('dropout: %f, train_steps: %d' % (dropout_train, train_steps))

        self.df_train.randomize()

        df_train, df_cv = self.df_train.split_train_test(train_size_rel=1 - self.ci.test_size_rel, name_df_1='train', name_df_2='cv')

        train_vocab_size, train_raw_data, train_x, train_y = df_train.get_data()
        cv_vocab_size, cv_raw_data, cv_x, cv_y = df_cv.get_data()
        test_vocab_size, test_raw_data, test_x, test_y = self.df_test.get_data()

        cv_cost = None
        test_cost = None
        global_step = self.sess.run(self.model.global_step)
        while global_step < train_steps:
            self.save_model(self.sess, self.saver, global_step)

            (batch_y, batch_x) = self.random_batch(train_y, train_x)
            # (batch_y, batch_x) = self.next_batch(data_y=train_y, data_x=train_x, step=global_step)
            feed_dict_train = {self.model.input_x: batch_x, self.model.y_true: batch_y, self.model.dropout_keep_prob: dropout_train}
            [global_step, _, _] = self.sess.run([self.model.global_step, self.model.train_step, self.model.summ], feed_dict=feed_dict_train)

            if global_step % 20 == 0:
                print('step: %d' % global_step)

                train_accuracy, train_cost, train_precision, train_recall, train_f1, train_summ = self.get_summary(batch_x, batch_y, dropout_train)
                cv_accuracy, cv_cost, cv_precision, cv_recall, cv_f1, cv_summ = self.get_summary(cv_x, cv_y, 1.0)

                self.train_summary_writer.add_summary(train_summ, global_step)
                self.cv_summary_writer.add_summary(cv_summ, global_step)

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

            if global_step % 200 == 0:
                test_accuracy, test_cost, test_precision, test_recall, test_f1, test_summ = self.get_summary(test_x, test_y, 1.0)
                self.test_summary_writer.add_summary(test_summ, global_step)

                print('')
                print('testset accuracy %g' % (test_accuracy))
                print('testset precision %g' % (test_precision))
                print('testset recall %g' % (test_recall))
                print('testset f1 %g' % (test_f1))

            if global_step % self.ci.auto_snapshot_interval == 0:
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
        ], feed_dict={self.model.input_x: x, self.model.y_true: y, self.model.dropout_keep_prob: dropout})

    def load_model(self) -> TextCNN:
        print('loading model')

        # vocab_path = os.path.join(self.ci.checkpoint_path, "..", "vocab")
        # vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

        return TextCNN(
            sequence_length=self.df_train.data_x.shape[1],
            num_classes=self.ci.y_len,
            vocab_size=self.df_train.vocab_size,
            embedding_size=128,
            filter_sizes=list(map(int, [3, 4, 5])),
            num_filters=128,
            l2_reg_lambda=self.ci.l2_reg_lambda,
            learning_rate=self.ci.learning_rate
        )
