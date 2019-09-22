# -*- coding: utf-8 -*-
import json
import os
import re

import MeCab
import neologdn

re_delimiter = re.compile("[。,．!\?]")
NEGATION = ('ない', 'ず', 'ぬ')
DICT_DIR = os.path.join(os.path.dirname(__file__), 'dic')


class Analyzer(object):

    def __init__(self, mecab_args=''):
        self.word_dict = json.load(open(os.path.join(DICT_DIR, 'pn_noun.json')))
        self.wago_dict = json.load(open(os.path.join(DICT_DIR, 'pn_wago.json')))
        self.tagger = MeCab.Tagger(mecab_args)
        self.tagger.parse('')  # for avoiding bug

    def _split_per_sentence(self, text):
        for sentence in re_delimiter.split(text):
            if sentence and not re_delimiter.match(sentence):
                yield sentence

    def _lookup_wago(self, lemma, lemmas):
        if lemma in self.wago_dict:
            return 1 if self.wago_dict[lemma].startswith('ポジ') else -1
        for i in range(10, 0, -1):
            wago = ' '.join(lemmas[-i:]) + ' ' + lemma
            if wago in self.wago_dict:
                return 1 if self.wago_dict[wago].startswith('ポジ') else -1
        return None

    def _calc_sentiment_polarity(self, sentence):
        polarities = []
        lemmas = []
        polarity_apeared = False
        node = self.tagger.parseToNode(sentence)
        while node:
            if 'BOS/EOS' not in node.feature:
                surface = node.surface
                feature = node.feature.split(',')
                lemma = feature[6] if feature[6] != '*' else node.surface
                if lemma in self.word_dict:
                    polarity = 1 if self.word_dict[lemma] == 'p' else -1
                    polarities.append(polarity)
                    polarity_apeared = True
                else:
                    polarity = self._lookup_wago(lemma, lemmas)
                    if polarity is not None:
                        polarities.append(polarity)
                        polarity_apeared = True
                    elif polarity_apeared and surface in NEGATION:
                        polarities[-1] *= -1
                lemmas.append(lemma)
            node = node.next
        if not polarities:
            return (0, 0)
        return (sum(polarities), len(polarities))

    def analyze(self, text, raw_score=False, target='sent'):
        """Calculate sentiment polarity scores per sentence or overall document
        Arg:
            text (str)
        Return:
            scores
        """
        text = neologdn.normalize(text)
        if target == 'sent':
            scores = []
            for sentence in self._split_per_sentence(text):
                score = self._calc_sentiment_polarity(sentence)
                if raw_score:
                    scores.append(score)
                else:
                    scores.append(score[0] / score[1] if score[1] != 0 else 0.0)
            return scores
        elif target == 'doc':
            scores = 0
            counts = 0
            for sentence in self._split_per_sentence(text):
                score = self._calc_sentiment_polarity(sentence)
                scores += score[0]
                counts += score[1]
            if raw_score:
                ret = (scores, counts)
            else:
                ret = scores / counts if counts != 0 else 0.0
            return ret
        else:
            raise ValueError('`target` is allowed only \'sent\' or \'doc\'.')
