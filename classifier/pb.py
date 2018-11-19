import ntpath
import os

import tensorflow as tf


class Pb():
    """
    Protobuf wrapper
    """

    def __init__(self, ci):
        self.ci = ci

    def save_graph_def(self, sess):
        logdir, name = ntpath.split(self.ci.pb_graph_path)
        tf.train.write_graph(sess.graph_def, logdir=logdir, name=name, as_text=True)

    def save(self, ckpt, final_test_accuracy=None, final_test_cost=None):
        """

        :param ckpt: If None - no best result could be evaluated
        :param final_test_accuracy:
        :param final_test_cost:
        :return:
        """
        if ckpt is not None:
            file = self.ci.pb_file
            print('saving pb file to: %s ' % file)
            tf.reset_default_graph()

            if ckpt is None:
                checkpoint_file = tf.train.latest_checkpoint(self.ci.checkpoint_path)
            else:
                checkpoint_file = '%s-%d' % (self.ci.checkpoint_file, ckpt)

            # input_node_names = ['input', 'keep_prob']
            output_node_name = 'output'
            from tensorflow.python.tools.freeze_graph import freeze_graph  # import takes very long
            freeze_graph(input_graph=self.ci.pb_graph_path,
                         input_saver=None,
                         input_binary=False,
                         input_checkpoint=checkpoint_file,
                         output_node_names=output_node_name,
                         restore_op_name="save/restore_all",
                         filename_tensor_name="save/Const:0",
                         output_graph=file,
                         clear_devices=True,
                         initializer_nodes="")

            # saves graphdef in binary graph structure file
            input_graph_def = tf.GraphDef()
            with tf.gfile.Open(file, "rb") as f:
                input_graph_def.ParseFromString(f.read())

            # output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            #     input_graph_def, input_node_names, [output_node_name],
            #     tf.float32.as_datatype_enum)

            # bazel-bin/tensorflow/examples/label_image/label_image --output_layer = final_result --labels = /tf_files/retrained_labels.txt --image=/tf_files/flower_photos/daisy/5547758_eea9edfd54_n.jpg --graph = /tf_files /rounded_graph.pb

            print("graph for classifier %s saved!" % self.ci.name)

            print('export labels')
            with tf.gfile.Open(self.ci.pb_labels_path, 'w') as f:
                for label in range(self.ci.y_len):
                    f.write('%s\n' % label)

            print('export label information')
            with tf.gfile.Open(self.ci.pb_information_path, 'w') as f:
                f.write('classifier name: %s\n' % self.ci.name)
                f.write('checkpoint: %s\n' % ckpt)

                if final_test_accuracy is not None:
                    f.write('final testset accuracy %s\n' % (final_test_accuracy))
                if final_test_cost is not None:
                    f.write('final testset loss %s\n' % (final_test_cost))

    def load(self):
        if os.path.isfile(self.ci.pb_file):
            print('loading pb file from: %s' % self.ci.pb_file)

            with tf.gfile.FastGFile(self.ci.pb_file, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                # tf.import_graph_def(graph_def)

            with tf.Graph().as_default() as graph:
                tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    return_elements=None,
                    name="prefix",
                    # op_dict=None, # Deprecated
                    producer_op_list=None
                )

                return graph
        else:
            raise Exception('the given pb file %s doesn\'t exist' % self.ci.pb_file)
