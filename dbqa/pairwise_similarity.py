import tensorflow as tf

from tf_biaffine_mst.layers import keep_prob_when_training
from tf_extra.layers import Embedding, ExternalEmbedding, BiLSTMLayer, get_batch_sequence_lengths, \
    ExternalEmbeddingLoader


class CNNMerger(object):
    def __init__(self, options, n_class):
        self.layers_filters = [
            # ("dropout", 0.20),
            ("conv", {1: 500, 2: 500, 3: 500}),
            # ("dropout", 0.55),
            ("globalpooling", None),
            ("relu", None),
            # ("relu", None),
            # ("conv", {2: 128, 3: 128, 4: 128}),
            # ("relu", None),
            # ("pooling", 2),
            # ("dropout", 0.20),
            # ("conv", {2: 128, 3: 128}),
            # ("relu", None),
            # ("pooling", 3),
            # ("conv", {1: n_class})
        ]
        self.options = options
        self.layers_objs = []
        for layer_type, params in self.layers_filters:
            objs = []
            self.layers_objs.append(objs)
            if layer_type == "conv":
                for size, count in params.items():
                    objs.append(tf.layers.Conv1D(count, size, padding="same"))
            elif layer_type == "pooling":
                objs.append(tf.layers.MaxPooling1D(params, params))
            elif layer_type == "dropout":
                objs.append(
                    (lambda keep_rate: lambda x, y:
                    tf.nn.dropout(x, keep_prob_when_training(keep_rate, y)))(params))
            elif layer_type == "relu":
                objs.append(tf.nn.relu)
            elif layer_type == "globalpooling":
                objs.append(lambda x:tf.reduce_max(x, axis=1))
            elif layer_type == "tanh":
                objs.append(tf.nn.tanh)
            else:
                raise Exception("Invalid layer type " + layer_type)

    def __call__(self, word_embeddings_3d, lengths, is_training):
        conv_output = word_embeddings_3d
        for objs in self.layers_objs:
            output_3d_list = []
            for obj in objs:
                try:
                    result = obj(conv_output)
                except TypeError:
                    result = obj(conv_output, is_training)
                output_3d_list.append(result)
            conv_output = tf.concat(output_3d_list, 2) \
                if len(output_3d_list) > 1 \
                else output_3d_list[0]
        return conv_output


class PairwiseSimilarity(object):
    def __init__(self, options, statistics):
        self.options = options
        self.word_embeddings = Embedding(statistics.words, options.embed_size, "Words")
        self.bigram_embeddings = Embedding(statistics.bigrams, options.embed_size, "Bigrams")
        if options.external_embedding is not None:
            embedding_loader = ExternalEmbeddingLoader(options.external_embedding)
            self.pretrained_embeddings = ExternalEmbedding(embedding_loader)

        if options.merger_type == "rnn":
            with tf.variable_scope("Recurrent", reuse=False):
                self.rnn = BiLSTMLayer(
                    options.lstm_size, options.n_recur,
                    options.input_keep_prob, options.recurrent_keep_prob)
        else:
            self.cnn = CNNMerger(self.options, options.lstm_size)

    def encode_sentence(self, words_2d, bigrams_2d, is_training):
        length = get_batch_sequence_lengths(words_2d)
        words_embed_3d = self.word_embeddings(words_2d)
        if self.options.external_embedding is not None:
            embed_pretrained_3d = self.word_embeddings(words_2d)
            words_embed_total_3d = words_embed_3d + embed_pretrained_3d
        else:
            words_embed_total_3d = words_embed_3d
        if self.options.use_bigram:
            bigrams_embed_3d = self.bigram_embeddings(bigrams_2d)
            embed_total_3d = tf.concat([words_embed_total_3d, bigrams_embed_3d], axis=1)
        else:
            embed_total_3d = words_embed_total_3d
        if self.options.merger_type == "rnn":
            encoded_2d = self.rnn(embed_total_3d, is_training, length, return_seq=False)
        else:
            encoded_2d = self.cnn(embed_total_3d, length, is_training)
        encoded_norm_2d = tf.nn.l2_normalize(encoded_2d, 1)
        return encoded_norm_2d

    @staticmethod
    def cos_similarity(a, b):
        return tf.reduce_sum(tf.multiply(a, b), axis=1)

    def get_loss(self,
                 question_2d,
                 question_bigram_2d,
                 answer_2d,
                 answer_bigram_2d,
                 wrong_answer_2d,
                 wrong_answer_bigram_2d,
                 ):
        batch_size = tf.to_float(tf.shape(question_2d)[0])
        question_encoded_2d = self.encode_sentence(question_2d, question_bigram_2d, True)
        answer_encoded_2d = self.encode_sentence(answer_2d, answer_bigram_2d, True)
        wrong_answer_encoded_2d = self.encode_sentence(wrong_answer_2d, wrong_answer_bigram_2d, True)
        correct_sim = self.cos_similarity(question_encoded_2d, answer_encoded_2d)
        wrong_sim = self.cos_similarity(question_encoded_2d, wrong_answer_encoded_2d)
        diff = tf.maximum(0.0, 0.5 + wrong_sim - correct_sim)
        loss = tf.reduce_sum(diff) / batch_size
        accuracy = tf.reduce_sum(tf.to_float(tf.equal(diff, 0.0))) / batch_size
        return loss, accuracy

    def get_similarity(self,
                       question_2d,
                       question_bigram_2d,
                       answer_2d,
                       answer_bigram_2d,
                       ):
        question_encoded_2d = self.encode_sentence(question_2d, question_bigram_2d, False)
        answer_encoded_2d = self.encode_sentence(answer_2d, answer_bigram_2d, False)
        return self.cos_similarity(question_encoded_2d, answer_encoded_2d)
