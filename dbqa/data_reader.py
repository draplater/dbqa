import pickle
from typing import List, Tuple

import attr
import os

import numpy as np

from logger import logger
from vocab_utils import Dictionary


def preprocess(input_file, output_file, segged=True):
    data = []
    with open(input_file) as f:
        last_question = ""
        question_item = None
        for line_idx, line in enumerate(f):
            if line_idx % 100 == 0:
                logger.info("{} preprocessed.".format(line_idx))
            line_s = line.strip()
            if not line_s:
                continue
            question, text, answer = line_s.split("\t")
            if question != last_question:
                last_question = question
                print(question_item)
                if question_item is not None:
                    data.append(question_item)
                if segged:
                    question_stripped = question.strip().split("  ")
                else:
                    question_stripped = list(question.strip())
                question_item = NLPCC16DBQA(question_stripped, [], [])
            answer = bool(int(answer))
            if segged:
                text_stripped = text.strip().split("  ")
            else:
                text_stripped = list(text.strip())
            if answer:
                question_item.correct.append(text_stripped)
                question_item.origin_idx.append(
                    ("correct", len(question_item.correct) - 1))
            else:
                question_item.wrong.append(text_stripped)
                question_item.origin_idx.append(
                    ("wrong", len(question_item.wrong) - 1))
        data.append(question_item)
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    return data


def to_bigram(sentence: List[str]) -> List[str]:
    shifted = ["<s>"] + sentence[:-1]
    ret = []
    for prev, this in zip(shifted, sentence):
        ret.append(prev + this)
    return ret


@attr.s
class NLPCC16DBQA(object):
    question = attr.ib(type=List[str])
    correct = attr.ib(type=List[List[str]])
    wrong = attr.ib(type=List[List[str]])
    origin_idx = attr.ib(type=List[Tuple[str, int]],
                         default=attr.Factory(lambda: []))
    question_id = attr.ib(type=List[int], default=None)
    correct_ids = attr.ib(type=List[List[int]], default=None)
    wrong_ids = attr.ib(type=List[List[int]], default=None)
    question_bigram_id = attr.ib(type=List[int], default=None)
    correct_bigram_ids = attr.ib(type=List[List[int]], default=None)
    wrong_bigram_ids = attr.ib(type=List[List[int]], default=None)

    @classmethod
    def from_file(cls, input_file, no_answer=True, segged=True):
        preprocessed_file = input_file + ".preprocessed.{}.5.pickle".format(segged)
        if os.path.exists(preprocessed_file):
            with open(preprocessed_file, "rb") as f:
                return pickle.load(f)
        else:
            return preprocess(input_file, preprocessed_file, segged)

    def fill_ids(self, statistics):
        self.question_id = [statistics.words.word_to_int.get(token, 1)
                            for token in self.question]
        self.correct_ids = [[statistics.words.word_to_int.get(token, 1)
                             for token in answer]
                            for answer in self.correct]
        self.wrong_ids = [[statistics.words.word_to_int.get(token, 1)
                           for token in answer]
                          for answer in self.wrong]
        self.question_bigram_id = [statistics.bigrams.word_to_int.get(token, 1)
                                   for token in to_bigram(self.question)]
        self.correct_bigram_ids = [[statistics.bigrams.word_to_int.get(token, 1)
                                    for token in to_bigram(answer)]
                                   for answer in self.correct]
        self.wrong_bigram_ids = [[statistics.bigrams.word_to_int.get(token, 1)
                                  for token in to_bigram(answer)]
                                 for answer in self.wrong]


class NLPCC16DBQACharacterBased(NLPCC16DBQA):
    @classmethod
    def from_file(cls, input_file, no_answer=True, segged=False):
        return super(NLPCC16DBQACharacterBased, cls).from_file(
            input_file, no_answer, segged)


def pad_as_input(pending):
    question_ids, question_bigram_ids, correct_ids, correct_bigram_ids, \
    wrong_ids, wrong_bigram_ids = zip(*pending)
    max_question_length = max(10, max(len(i) for i in question_ids))
    max_correct_length = max(10, max(len(i) for i in correct_ids))
    max_wrong_length = max(10, max(len(i) for i in wrong_ids))
    batch_size = len(pending)
    ret_questions = np.zeros((batch_size, max_question_length))
    ret_questions_bigram = np.zeros((batch_size, max_question_length))
    ret_corrects = np.zeros((batch_size, max_correct_length))
    ret_corrects_bigram = np.zeros((batch_size, max_correct_length))
    ret_wrongs = np.zeros((batch_size, max_wrong_length))
    ret_wrongs_bigram = np.zeros((batch_size, max_wrong_length))
    for idx, (question_id, question_bigram_id, correct_id, correct_bigram_id,
              wrong_id, wrong_bigram_id) in enumerate(pending):
        ret_questions[idx, :len(question_id)] = question_id
        ret_questions_bigram[idx, :len(question_id)] = question_bigram_id
        ret_corrects[idx, :len(correct_id)] = correct_id
        ret_corrects_bigram[idx, :len(correct_id)] = correct_bigram_id
        ret_wrongs[idx, :len(wrong_id)] = wrong_id
        ret_wrongs_bigram[idx, :len(wrong_id)] = wrong_bigram_id
    return ret_questions, ret_questions_bigram,\
           ret_corrects, ret_corrects_bigram,\
           ret_wrongs, ret_wrongs_bigram


def generate_train_batches(question_list: List[NLPCC16DBQA],
                           batch_size: int,
                           random_obj):
    processed_tokens = 0
    pending = []
    for question in question_list:
        for correct_id, correct_bigram_id in zip(question.correct_ids, question.correct_bigram_ids):
            for wrong_id, wrong_bigram_id in zip(question.wrong_ids, question.wrong_bigram_ids):
                pending.append((
                    question.question_id, question.question_bigram_id,
                    correct_id, correct_bigram_id, wrong_id, wrong_bigram_id))
                processed_tokens += len(question.question_id) + len(correct_id) + len(wrong_id)
                if processed_tokens > batch_size:
                    random_obj.shuffle(pending)
                    yield pad_as_input(pending)
                    processed_tokens = 0
                    pending = []
    if pending:
        random_obj.shuffle(pending)
        yield pad_as_input(pending)


def pad_predict_as_input(pending):
    question_ids, question_bigram_ids, answer_ids, answer_bigram_ids = zip(*pending)
    max_question_length = max(10, max(len(i) for i in question_ids))
    max_answer_length = max(10, max(len(i) for i in answer_ids))
    batch_size = len(pending)
    ret_questions = np.zeros((batch_size, max_question_length))
    ret_questions_bigram = np.zeros((batch_size, max_question_length))
    ret_answer = np.zeros((batch_size, max_answer_length))
    ret_answer_bigram = np.zeros((batch_size, max_answer_length))
    for idx, (question_id, question_bigram_id,
              answer_id, answer_bigram_id) in enumerate(pending):
        ret_questions[idx, :len(question_id)] = question_id
        ret_questions_bigram[idx, :len(question_id)] = question_bigram_id
        ret_answer[idx, :len(answer_id)] = answer_id
        ret_answer_bigram[idx, :len(answer_id)] = answer_bigram_id
    return ret_questions, ret_questions_bigram, ret_answer, ret_answer_bigram


def generate_predict_batches(question_list: List[NLPCC16DBQA], batch_size: int):
    processed_tokens = 0
    pending = []
    for question in question_list:
        for ans_type, ans_idx in question.origin_idx:
            answer = (question.correct_ids if ans_type == "correct" else question.wrong_ids)[ans_idx]
            answer_bigram = (question.correct_bigram_ids if ans_type == "correct" else question.wrong_bigram_ids)[ans_idx]
            pending.append((question.question_id, question.question_bigram_id, answer, answer_bigram))
            processed_tokens += len(question.question_id) + len(answer)
            if processed_tokens > batch_size:
                yield pad_predict_as_input(pending)
                processed_tokens = 0
                pending = []
    if pending:
        yield pad_predict_as_input(pending)


@attr.s
class DBQAStatistics(object):
    words = attr.ib(type=Dictionary)
    bigrams = attr.ib(type=Dictionary)

    @classmethod
    def from_data(cls, data):
        words = Dictionary()
        bigrams = Dictionary()
        for question in data:
            words.update(question.question)
            bigrams.update(to_bigram(question.question))
            for answer in question.correct:
                words.update(answer)
                bigrams.update(to_bigram(answer))
            for answer in question.wrong:
                words.update(answer)
                bigrams.update(to_bigram(answer))
        words = words.strip_low_freq(
            min_count=10,
            ensure=("___PAD___", "___UNKNOWN___"))
        bigrams = bigrams.strip_low_freq(
            min_count=10,
            ensure=("___PAD___", "___UNKNOWN___"))
        # noinspection PyArgumentList
        return cls(words, bigrams)


if __name__ == '__main__':
    home = os.path.expanduser("~")
    data_dir = f"{home}/Development/large-data/"
    train_data = NLPCC16DBQA.from_file(data_dir + "nlpcc-iccpol-2016.dbqa.training-data")
    test_data = NLPCC16DBQA.from_file(data_dir + "nlpcc-iccpol-2016.dbqa.testing-data")
    statistics = DBQAStatistics.from_data(train_data)
    # NLPCC16DBQA.from_file(data_dir + "dbqa-small")
    pass
