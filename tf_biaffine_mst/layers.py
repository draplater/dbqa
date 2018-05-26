import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.layers.utils import smart_cond


class Bilinear(object):
    def __init__(self, inputs1_size, inputs2_size, output_size, add_bias1=True, add_bias2=False,
                 add_bias=False, initializer=None, scope=None):
        self.output_size = output_size
        self.add_bias1 = add_bias1
        self.add_bias2 = add_bias2
        self.add_bias = add_bias
        self.initializer = initializer
        if self.initializer is None:
            initializer = tf.orthogonal_initializer
        self.scope = scope
        with tf.variable_scope(self.scope or 'Bilinear', reuse=False):
            self.weights = tf.get_variable(
                'Weights', [inputs1_size + add_bias1, output_size, inputs2_size + add_bias2],
                initializer=initializer)
            tf.add_to_collection('Weights', self.weights)
            if add_bias:
                self.bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer)

    def calc(self, inputs1, inputs2, keep_prob=None):
        with tf.variable_scope(self.scope or 'Bilinear'):
            # Reformat the inputs
            ndims = len(inputs1.get_shape().as_list())
            inputs1_shape = tf.shape(inputs1)
            batch_size = inputs1_shape[0]
            inputs1_bucket_size = inputs1_shape[ndims - 2]
            inputs1_size = inputs1.get_shape().as_list()[-1]

            inputs2_shape = tf.shape(inputs2)
            inputs2_bucket_size = inputs2_shape[ndims - 2]
            inputs2_size = inputs2.get_shape().as_list()[-1]
            output_shape = []
            batch_size = 1
            for i in range(ndims - 2):
                batch_size *= inputs1_shape[i]
                output_shape.append(inputs1_shape[i])

            output_shape.append(inputs1_bucket_size)
            output_shape.append(self.output_size)
            output_shape.append(inputs2_bucket_size)
            output_shape = tf.stack(output_shape)

            if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
                noise_shape = tf.stack([batch_size, 1, inputs1_size])
                inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape)
                inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape)

            inputs1 = tf.reshape(inputs1, tf.stack([batch_size, inputs1_bucket_size, inputs1_size]))
            inputs2 = tf.reshape(inputs2, tf.stack([batch_size, inputs2_bucket_size, inputs2_size]))
            if self.add_bias1:
                inputs1 = tf.concat([inputs1, tf.ones(tf.stack([batch_size, inputs1_bucket_size, 1]))], 2)
            if self.add_bias2:
                inputs2 = tf.concat([inputs2, tf.ones(tf.stack([batch_size, inputs2_bucket_size, 1]))], 2)

            # Do the multiplications
            # (bn x d) (d x rd) -> (bn x rd)
            lin = tf.matmul(tf.reshape(inputs1, [-1, inputs1_size + self.add_bias1]),
                            tf.reshape(self.weights, [inputs1_size + self.add_bias1, -1]))
            # (b x nr x d) (b x n x d)T -> (b x nr x n)
            bilin = tf.matmul(
                tf.reshape(lin, tf.stack([batch_size, inputs1_bucket_size * self.output_size,
                                          inputs2_size + self.add_bias2])),
                inputs2, adjoint_b=True)
            # (bn x r x n)
            bilin = tf.reshape(bilin, tf.stack([-1, self.output_size, inputs2_bucket_size]))
            # (b x n x r x n)
            bilin = tf.reshape(bilin, output_shape)

            # Get the bias
            if self.add_bias:
                bias = tf.get_variable('Biases', [self.output_size], initializer=tf.zeros_initializer)
                bilin += tf.expand_dims(bias, 1)
            return bilin

    def __call__(self, inputs1, inputs2, keep_prob=None):
        bilin = self.calc(inputs1, inputs2, keep_prob)
        bilin = tf.squeeze(bilin, axis=2)
        return bilin
        # return output

    def conditional(self, inputs1, inputs2, probs, keep_prob=None):
        input_shape = tf.shape(inputs1)
        batch_size = input_shape[0]
        bucket_size = input_shape[1]
        input_size = inputs1.get_shape().as_list()[-1]
        n_classes = self.output_size
        input_shape_to_set = [tf.Dimension(None), tf.Dimension(None), input_size + 1]
        output_shape = tf.stack([batch_size, bucket_size, n_classes, bucket_size])
        if not context.in_eager_mode():
            probs_dim_count = len(probs.get_shape().as_list())
        else:
            probs_dim_count = len(probs.shape)
        if probs_dim_count == 2:
            probs = tf.to_float(tf.one_hot(tf.to_int64(probs), bucket_size, 1, 0))
        else:
            probs = tf.stop_gradient(probs)

        inputs1 = tf.concat([inputs1, tf.ones(tf.stack([batch_size, bucket_size, 1]))], 2)
        inputs1.set_shape(input_shape_to_set)
        inputs2 = tf.concat([inputs2, tf.ones(tf.stack([batch_size, bucket_size, 1]))], 2)
        inputs2.set_shape(input_shape_to_set)

        bilin = self.calc(inputs1, inputs2, keep_prob)
        weighted_bilin = tf.matmul(bilin, tf.expand_dims(probs, 3))
        return weighted_bilin, bilin


def keep_prob_when_training(keep_prob, is_training):
    return smart_cond(is_training, lambda: keep_prob, lambda: 1.0)


class ConvLayer(object):
    def __init__(self, filter_sizes_and_counts, stride=1):
        self.conv_layers = {size: tf.layers.Conv2D(count, size, (1, stride), 'valid', use_bias=True)
                            for size, count in filter_sizes_and_counts.items()}

    def __call__(self, input_tensor):
        # (batch_size, bucket_size, lstm_output_size, 1)
        input_tensor = tf.expand_dims(input_tensor, 3)
        bucket_size = tf.shape(input_tensor)[1]
        outputs = []
        for size, conv_layer in self.conv_layers.items():
            pad_front = (size[0] - 1) // 2
            pad_back = size[0] - 1 - pad_front
            padded = tf.pad(input_tensor, [[0, 0], [pad_front, pad_back], [0, 0], [0, 0]])
            # (batch_size, bucket_size, lstm_output_size - kernel_size + 1, 1)
            output = conv_layer(padded)
            dim = tf.shape(input_tensor)[2]
            # output = tf.nn.max_pool(output, (1, 1, bucket_size, 1), (1, 1, 1, 1), "SAME")
            # output = tf.reduce_sum(output, axis=2) / tf.to_float(dim)
            output = tf.reduce_max(output, axis=2)
            outputs.append(output)
        return tf.nn.leaky_relu(tf.concat(outputs, axis=2), 0.1)


class CharacterConvLayer(object):
    def __init__(self, options, filter_sizes_and_counts, stride=1):
        self.options = options
        self.conv_layers = {size: tf.layers.Conv2D(count, size, (1, stride), 'valid', use_bias=True)
                            for size, count in filter_sizes_and_counts.items()}

    def __call__(self, characters_3d, char_embeded_4d, reuse=False):
        # (batch_size, bucket_size, lstm_output_size, 1)
        outputs = []
        batch_size = tf.shape(char_embeded_4d)[0]
        bucket_size = tf.shape(char_embeded_4d)[1]

        # batch_size * bucket_size, max_characters, embed_size
        char_embeded_3d = tf.reshape(char_embeded_4d, [batch_size * bucket_size,
                                    self.options.max_char, -1])

        char_embeded_3d_expanded = tf.expand_dims(char_embeded_3d, 3)
        for size, conv_layer in self.conv_layers.items():
            # (batch_size * bucket_size, max_char, lstm_output_size - kernel_size + 1, output_channels)
            output = conv_layer(char_embeded_3d_expanded)
            # (batch_size * bucket_size, output_channels)
            output = tf.reduce_max(output, axis=[1, 2])
            outputs.append(output)

        ret_2d = tf.nn.leaky_relu(tf.concat(outputs, axis=1), 0.1)
        ret_3d = tf.reshape(ret_2d, (batch_size, bucket_size, self.options.embed_size))
        return ret_3d


class CharLSTMLayer(object):
    def __init__(self, options, reuse=False):
        self.options = options
        self.char_lstm_cells = []
        self.char_lstm_cells_bw = []
        for i in range(self.options.char_lstm_layers):
            with tf.variable_scope('CharRNN%d' % i, reuse=reuse):
                self.char_lstm_cells.append(
                    tf.nn.rnn_cell.LSTMCell(self.options.embed_size / 2, name="fw"))
                self.char_lstm_cells_bw.append(
                    tf.nn.rnn_cell.LSTMCell(self.options.embed_size / 2, name="bw"))

    def __call__(self, characters_3d, char_embeded_4d, reuse=False):
        """

        :param characters: batch_size, bucket_size, max_characters, embed_size
        :return:
        """
        batch_size = tf.shape(char_embeded_4d)[0]
        bucket_size = tf.shape(char_embeded_4d)[1]

        # batch_size, bucket_size
        word_lengths_2d = tf.reduce_sum(tf.to_int32(tf.greater(characters_3d, 0)), axis=2)
        # batch_size * bucket_size
        word_lengths_1d = tf.reshape(word_lengths_2d, [-1])

        # batch_size * bucket_size, max_characters, embed_size
        char_embeded_3d = tf.reshape(char_embeded_4d, [batch_size * bucket_size,
                                    self.options.max_char, self.options.embed_size])
        char_top_recur = char_embeded_3d

        for i in range(2):
            with tf.variable_scope('embeddings/characters/CharRNN%d' % i, reuse=reuse):
                cell_fw = self.char_lstm_cells[i]
                cell_bw = self.char_lstm_cells_bw[i]
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, char_top_recur,
                                                                            word_lengths_1d,
                                                                            dtype=tf.float32)
                # batch_size * bucket_size, word_length, embedding_dims
                if i == 1:
                    # batch_size * bucket_size, embedding_dims
                    output_bw_last = output_bw[:,0,:]
                    fw_indices = tf.stack([
                        tf.range(batch_size * bucket_size),
                        tf.nn.relu(word_lengths_1d - 1)
                        ], axis=1)
                    output_fw_last = tf.gather_nd(output_fw, fw_indices)
                    # batch_size * bucket_size, embedding_dims * 2
                    lstm_output_2d = tf.concat([output_fw_last, output_bw_last], axis=1)
                    # batch_size, bucket_size, embedding_dims * 2
                    lstm_output_3d = tf.reshape(lstm_output_2d, [batch_size, bucket_size, -1])
                    lstm_output_3d.set_shape([None, None, 100])
                else:
                    char_top_recur = tf.concat([output_fw, output_bw], 2)
        # noinspection PyUnboundLocalVariable
        return lstm_output_3d


loss_funcs = {"softmax": lambda logits, labels: tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits),
              "hinge": tf.losses.hinge_loss}

