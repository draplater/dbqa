from collections import Counter, namedtuple

import numpy as np
from namedlist import namedlist


class Dictionary(Counter):
    def __init__(self, initial=("___PAD___", "___UNKNOWN___")):
        super(Dictionary, self).__init__()
        self.int_to_word = []
        self.word_to_int = {}
        self.update(initial)

    def __setitem__(self, key, value):
        if not key in self:
            self.word_to_int[key] = len(self.int_to_word)
            self.int_to_word.append(key)
        super(Dictionary, self).__setitem__(key, value)

    def lookup(self, sentences, bucket, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)

        def lookup_by_keys(obj, keys, default=1):
            result = None
            for key in keys:
                result = self.word_to_int.get(getattr(obj, key))
                if result is not None:
                    break
            if result is None:
                result = default
            return result

        result = np.zeros((len(sentences), bucket), dtype=np.int32)
        for sent_idx, sentence in enumerate(sentences):
            result[sent_idx, :len(sentence)] = [lookup_by_keys(i, keys) for i in sentence]
        return result

    def use_top_k(self, k, ensure=()):
        ret = Dictionary(initial=())
        for ensure_item in ensure:
            ret[ensure_item] = 0
        for word, count in self.most_common(k):
            ret[word] = count
        ret.int_to_word = list(ret.keys())
        ret.word_to_int = {word: idx for idx, word in enumerate(ret.int_to_word)}
        return ret

    def strip_low_freq(self, min_count=1, ensure=()):
        ret = Dictionary(initial=())
        for ensure_item in ensure:
            ret[ensure_item] = 1
        for word, count in self.items():
            if count >= min_count:
                ret[word] = count
        ret.int_to_word = list(ret.keys())
        ret.word_to_int = {word: idx for idx, word in enumerate(ret.int_to_word)}
        return ret

    def __getstate__(self):
        return dict(self), self.int_to_word, self.word_to_int

    def __setstate__(self, state):
        data, self.int_to_word, self.word_to_int = state
        self.update(data)

    def __reduce__(self):
        return Dictionary, ((),), self.__getstate__()


