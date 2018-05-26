import os
from random import Random

import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer

from dbqa.data_reader import NLPCC16DBQA, DBQAStatistics, generate_train_batches, generate_predict_batches, \
    NLPCC16DBQACharacterBased
from dbqa.pairwise_similarity import PairwiseSimilarity
from logger import logger
from parser_base import DependencyParserBase
from tf_extra.layers import ExternalEmbeddingLoader


class DBQA(DependencyParserBase):
    available_data_formats = {"word-based": NLPCC16DBQA,
                              "character-based": NLPCC16DBQACharacterBased}
    default_data_format_name = "word-based"

    @classmethod
    def add_parser_arguments(cls, arg_parser):
        super(DBQA, cls).add_parser_arguments(arg_parser)
        group = arg_parser.add_argument_group(DBQA.__name__)
        group.add_argument("--external-embedding")
        group.add_argument("--batch-size", type=int, default=4096)
        group.add_argument("--embed-size", type=int, default=100)
        group.add_argument("--lstm-size", type=int, default=256)
        group.add_argument("--n-recur", type=int, default=2)
        group.add_argument("--use-bigram", type=int, default=1)
        group.add_argument("--input-keep-prob", type=int, default=1)
        group.add_argument("--recurrent-keep-prob", type=int, default=1)
        group.add_argument("--seed", type=int, default=42)
        group.add_argument("--steps", type=int, default=50000)
        group.add_argument("--merger-type", choices=["rnn", "cnn"], default="rnn")

    def __init__(self, options, data_train, session=None):
        self.statistics = DBQAStatistics.from_data(data_train)
        self.options = options

        self.optimizer = AdamOptimizer()
        self.global_step = tf.train.get_or_create_global_step()

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        self.question_2d_pl = tf.placeholder(tf.int32, (None, None))
        self.question_bigram_2d_pl = tf.placeholder(tf.int32, (None, None))
        self.answer_2d_pl = tf.placeholder(tf.int32, (None, None))
        self.answer_bigram_2d_pl = tf.placeholder(tf.int32, (None, None))
        self.wrong_answer_2d_pl = tf.placeholder(tf.int32, (None, None))
        self.wrong_answer_bigram_2d_pl = tf.placeholder(tf.int32, (None, None))

        self.network = PairwiseSimilarity(
            options, self.statistics)
        self.loss, self.accuracy = self.network.get_loss(
            self.question_2d_pl,
            self.question_bigram_2d_pl,
            self.answer_2d_pl,
            self.answer_bigram_2d_pl,
            self.wrong_answer_2d_pl,
            self.wrong_answer_bigram_2d_pl,
        )

        self.similarity = self.network.get_similarity(
            self.question_2d_pl,
            self.question_bigram_2d_pl,
            self.answer_2d_pl,
            self.answer_bigram_2d_pl
        )

        self.optimize_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        if session is None:
            self.session = self.create_session()
            self.session.run(tf.global_variables_initializer())
        else:
            self.session = session
        self.random = Random(42)

    def create_session(self):
        config_proto = tf.ConfigProto()
        # config_proto.gpu_options.per_process_gpu_memory_fraction = self.options.per_process_gpu_memory_fraction
        return tf.Session(config=config_proto)

    def train(self, data_train):
        for questions_np, questions_bigram_np, \
            corrects_np, corrects_bigram_np, \
            wrongs_np, wrongs_bigram_np in generate_train_batches(
            data_train, self.options.batch_size, self.random
        ):
            step, loss, accuracy, _ = self.session.run([self.global_step, self.loss, self.accuracy, self.optimize_op],
                                                       {self.question_2d_pl: questions_np,
                                                        self.question_bigram_2d_pl: questions_bigram_np,
                                                        self.answer_2d_pl: corrects_np,
                                                        self.answer_bigram_2d_pl: corrects_bigram_np,
                                                        self.wrong_answer_2d_pl: wrongs_np,
                                                        self.wrong_answer_bigram_2d_pl: wrongs_bigram_np
                                                        })
            logger.info("Train: Step {}, loss {}, accuracy {}".format(
                step, loss, accuracy))

    @classmethod
    def repeat_train_and_validate(cls, data_train, data_devs, data_test, options):
        tf.set_random_seed(options.seed)
        parser = cls(options, data_train)
        for question in data_train:
            question.fill_ids(parser.statistics)
        for file_name, data_dev in data_devs.items():
            for question in data_dev:
                question.fill_ids(parser.statistics)
        while True:
            step = parser.session.run(parser.global_step)
            if step > options.steps:
                break
            parser.random.shuffle(data_train)
            parser.train(data_train)
            for file_name, data_dev in data_devs.items():
                try:
                    prefix, suffix = os.path.basename(file_name).rsplit(".", 1)
                except ValueError:
                    prefix = os.path.basename(file_name)
                    suffix = ""
                dev_output = os.path.join(options.output, '{}_step_{}.{}'.format(prefix, step, suffix))
                scores = list(parser.predict(data_dev))
                with open(dev_output, "w") as f_output:
                    for score in scores:
                        f_output.write("{}\n".format(score))

    @classmethod
    def load(cls, prefix, new_options=None):
        pass

    def predict(self, data_dev):
        for questions_np, questions_bigram_np,\
            answer_np, answer_bigram_np in generate_predict_batches(
                data_dev, self.options.batch_size
        ):
            similarities = self.session.run(
                self.similarity,
                {self.question_2d_pl: questions_np,
                 self.question_bigram_2d_pl: questions_bigram_np,
                 self.answer_2d_pl: answer_np,
                 self.answer_bigram_2d_pl: answer_bigram_np
                 })
            for similarity in similarities:
                yield similarity

    def save(self, prefix):
        pass
