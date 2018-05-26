import tensorflow as tf

from common_utils import read_embedding
from tf_biaffine_mst.layers import keep_prob_when_training
import numpy as np


class Embedding(object):
    def __init__(self, dictionary, size, name="Embedding", dtype=tf.float32):
        self.name = name
        with tf.variable_scope(name):
            self.embeddings_matrix = tf.get_variable(
                name, shape=(len(dictionary), size), dtype=dtype)

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            return tf.nn.embedding_lookup(self.embeddings_matrix, inputs)


class BiLSTMLayer(object):
    def __init__(self, lstm_size, n_layers, input_keep_prob=1, state_keep_prob=1):
        self.lstm_size = lstm_size
        self.n_layers = n_layers
        self.input_keep_prob = input_keep_prob
        self.state_keep_prob = state_keep_prob
        self.lstm_cells = []
        self.lstm_cells_bw = []
        for i in range(n_layers):
            self.lstm_cells.append(tf.nn.rnn_cell.LSTMCell(lstm_size))
            self.lstm_cells_bw.append(tf.nn.rnn_cell.LSTMCell(lstm_size))

    def __call__(self, input_3d, is_training, lengths_1d=None, reuse=None, return_seq=True):
        # batch_size, seq_length, dims
        output_3d = input_3d
        assert self.n_layers > 0
        for i in range(self.n_layers):
            with tf.variable_scope('RNN%d' % i, reuse=reuse):
                if not isinstance(is_training, bool) or is_training:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(
                        self.lstm_cells[i],
                        input_keep_prob=keep_prob_when_training(self.input_keep_prob, is_training),
                        state_keep_prob=keep_prob_when_training(self.state_keep_prob, is_training)
                    )
                    cell_bw = tf.contrib.rnn.DropoutWrapper(
                        self.lstm_cells_bw[i],
                        input_keep_prob=keep_prob_when_training(self.input_keep_prob, is_training),
                        state_keep_prob=keep_prob_when_training(self.state_keep_prob, is_training)
                    )
                else:
                    cell_fw = self.lstm_cells[i]
                    cell_bw = self.lstm_cells_bw[i]

                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output_3d, lengths_1d,
                                                                            dtype=tf.float32)
                output_3d = tf.concat([output_fw, output_bw], 2)
        if return_seq:
            return output_3d
        else:
            # batch_size, embedding_dims, lstm_size
            # noinspection PyUnboundLocalVariable
            output_bw_last_2d = output_bw[:, 0, :]
            batch_size = tf.shape(input_3d)[0]
            fw_indices = tf.stack([
                tf.range(batch_size),
                tf.nn.relu(lengths_1d - 1)
            ], axis=1)
            # batch_size, embedding_dims, lstm_size
            # noinspection PyUnboundLocalVariable
            output_fw_last_2d = tf.gather_nd(output_fw, fw_indices)
            # batch_size, embedding_dims, lstm_size * 2
            output_2d = tf.concat([output_fw_last_2d, output_bw_last_2d], axis=1)
            return output_2d


class ExternalEmbeddingLoader(object):
    def __init__(self, embedding_filename, encoding="utf-8", dtype=np.float32):
        words_and_vectors = read_embedding(embedding_filename, encoding)
        self.dim = len(words_and_vectors[0][1])
        words_and_vectors.insert(0, ("*UNK*", np.array([0] * self.dim)))

        words, vectors = zip(*words_and_vectors)
        self.lookup = {word: idx for idx, word in enumerate(words)}
        self.vectors = np.array(vectors, dtype=dtype)


class ExternalEmbedding(object):
    def __init__(self, loader, name="PretrainedEmbedding", dtype=tf.float32):
        self.name = name
        with tf.variable_scope(name):
            self.embeddings_matrix = tf.get_variable(
                name, initializer=loader.vectors, trainable=False,
                dtype=dtype)

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            return tf.nn.embedding_lookup(self.embeddings_matrix, inputs)


def get_batch_sequence_lengths(words_2d):
    return tf.reduce_sum(tf.to_int32(tf.greater(words_2d, 0)), axis=1)


def tile_in_new_dim(x, k, axis):
    expanded = tf.expand_dims(x, axis=axis)
    multiples = [1] * len(expanded.get_shape())
    multiples[axis] *= k
    return tf.tile(expanded, multiples)
