import tensorflow as tf

from classifier.cnn.text_ci import TextClassifierInformation
from classifier.pb import Pb


class ClassifierPredictor():
    """
    This works for both models
    model code: https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
    """

    def __init__(self, ci: TextClassifierInformation, pb: Pb):
        self.props = ci
        self.pb = pb
        self.graph = pb.load()
        self.sess = tf.Session(graph=self.graph)

    def fetch_probs(self, x_batch):
        if self.graph is not None:
            # We access the input and output nodes
            # x = self.graph.get_tensor_by_name('prefix/input:0')
            x = self.graph.get_tensor_by_name('prefix/input_x:0')
            dropout_keep_prob = self.graph.get_tensor_by_name('prefix/dropout_keep_prob:0')
            y_pred = self.graph.get_tensor_by_name('prefix/output/logits:0')
            # y_pred = self.graph.get_operation_by_name("output/logits").outputs[0]

            probabilities = self.sess.run(y_pred, feed_dict={x: x_batch, dropout_keep_prob: 1.0})
            # print(probabilities)
            return probabilities

    def print_ops(self):
        # We can verify that we can access the list of operations in the graph
        for op in self.graph.get_operations():
            print(op.name)
            # prefix/Placeholder/inputs_placeholder
            # ...
            # prefix/Accuracy/predictions
