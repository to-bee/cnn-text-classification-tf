import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda, learning_rate):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.y_true = tf.placeholder(tf.float32, [None, num_classes], name="y_true")
        y_true = tf.argmax(self.y_true, axis=1)

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.layer_fc_dropout = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and logits
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.layer_fc_dropout, W, b, name="scores")
            self.logits = tf.argmax(self.scores, axis=1, name="logits")

        # Calculate mean cross-entropy loss
        # with tf.name_scope("loss"):
        #     losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y_true)
        #     self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("Basic-Metrics"):
            correct_prediction = tf.equal(self.logits, y_true)
            self.accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('Accuracy', self.accuracy_op)

            # Computes the mean of elements across dimensions of a tensor.
            # so in this case across output probabilties
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.y_true)
            # self.cost = tf.reduce_mean(cross_entropy, name="cost") # old one without l2 regularization
            self.cost = tf.reduce_mean(cross_entropy, name="cost") + l2_reg_lambda * l2_loss  # from cnn-text-classification-tf
            # save that single number
            tf.summary.scalar("xent", self.cost)

        with tf.name_scope("Extended-Metrics"):
            # from: https://gist.github.com/Mistobaan/337222ac3acbfc00bdac
            predictions = self.logits
            # predicted = tf.round(tf.nn.sigmoid(self.y_pred))
            actuals = tf.argmax(self.y_true, axis=1)

            ones_like_actuals = tf.ones_like(actuals)
            zeros_like_actuals = tf.zeros_like(actuals)
            ones_like_predictions = tf.ones_like(predictions)
            zeros_like_predictions = tf.zeros_like(predictions)

            tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals), tf.equal(predictions, ones_like_predictions)), "float"))
            tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals), tf.equal(predictions, zeros_like_predictions)), "float"))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals), tf.equal(predictions, ones_like_predictions)), "float"))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals), tf.equal(predictions, zeros_like_predictions)), "float"))

            # same as above but slower
            # self.accuracy_op = (tp + tn) / (tp + fp + fn + tn)

            self.precision_op = tp / (tp + fp)
            self.recall_op = tp / (tp + fn)
            self.f1_score_op = (2 * self.precision_op * self.recall_op) / (self.precision_op + self.recall_op)

            # Add metrics to TensorBoard.
            tf.summary.scalar('Precision', self.precision_op)
            tf.summary.scalar('Recall', self.recall_op)
            tf.summary.scalar('F1', self.f1_score_op)

        with tf.name_scope("train"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost, global_step=self.global_step)

        # merge them all so one write to disk, more comp efficient
        self.summ = tf.summary.merge_all()
        print('model loaded')
