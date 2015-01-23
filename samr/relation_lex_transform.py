from collections import defaultdict

import cPickle as pickle
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

from samr.transformations import StatelessTransform
from samr.data import *


class BuildSubPhrase(StatelessTransform):
    def __init__(self, prior_phrase=[]):
        assert (isinstance(prior_phrase, list))
        self.phrase_set = set(prior_phrase)

    def transform(self, X):
        """
        `X` is expected to be a list of `str` instances.
        Return value is Phrasepair (lphrase rphrase matched)
        """
        hdict = {}

        phrase_by_len = defaultdict(list)
        for x in X:
            phrase_by_len[len(x)].append(x.phrase)

        for phlen in sorted(phrase_by_len.keys()):
            for phrase in phrase_by_len[phlen]:
                lphrase, rphrase, match = self._find_subphrase(phrase)
                hdict[phrase] = Phrasepair(lphrase, rphrase, match)

        return [hdict[x.phrase] for x in X]

    def _find_subphrase(self, phrase):
        phrases = phrase.split()

        if len(phrases) == 1:
            return phrases[0], '', 'LEFT|SINGLE'

        for i in range(len(phrases)):
            lphrase, rphrase = ' '.join(phrases[:i + 1]), ' '.join(phrases[i + 1:])
            if lphrase in self.phrase_set and rphrase in self.phrase_set:
                return lphrase, rphrase, 'BOTH'

        for i in range(1, len(phrases) - 1):
            lphrase, rphrase = ' '.join(phrases[:i + 1]), ' '.join(phrases[i + 1:])
            if lphrase in self.phrase_set:
                return lphrase, rphrase, 'LEFT'
            elif rphrase in self.phrase_set:
                return lphrase, rphrase, 'RIGHT'

        return phrase, '', 'LEFT|UNKNOWN'


class BuildSubPhrase2():
    def __init__(self, prior_dict={}, default_sentiment=2, verbose=False):
        self.phrase_by_len = defaultdict(list)

        for text in prior_dict.keys():
            self.phrase_by_len[len(text.split())].append(text)

        self.default_sentiment = default_sentiment
        self.verbose = verbose

    def fit(self, X, y):
        self.sentiment_bank = {}
        for (datapoint, sentiment) in zip(X, y):
            self.sentiment_bank[datapoint.phrase] = sentiment

        return self

    def _find_sentiment(self, text):
        assert (isinstance(text, str))

        if text in self.sentiment_bank:
            return self.sentiment_bank[text]
        return self.default_sentiment

    def transform(self, X):
        for datapoint in X:
            text = datapoint.phrase
            self.phrase_by_len[len(text.split())].append(text)

        hdict = {}

        for i in sorted(self.phrase_by_len.keys()):
            for text in self.phrase_by_len[i]:
                lphrase, rphrase, matched = self._find_subphrase(text)
                hdict[text] = Sentimentpoint(lphrase, rphrase, self._find_sentiment(lphrase),
                                             self._find_sentiment(rphrase))

        return [hdict[x.phrase] for x in X]

    def _find_subphrase(self, text):
        phrases = text.split()

        if len(phrases) == 1:
            return phrases[0], '', 'LEFT|SINGLE'

        for i in range(len(phrases)):
            lphrase, rphrase = phrases[:i + 1], phrases[i + 1:]
            if ' '.join(lphrase) in self.phrase_by_len[len(lphrase)] and ' '.join(rphrase) in self.phrase_by_len[
                len(rphrase)]:
                return ' '.join(lphrase), ' '.join(rphrase), 'BOTH'

        for i in range(1, len(phrases)):
            lphrase = phrases[:i + 1]
            if ' '.join(lphrase) in self.phrase_by_len[len(lphrase)]:
                return ' '.join(lphrase), '', 'LEFT'

        for i in range(len(phrases) - 1):
            rphrase = phrases[i + 1:]
            if ' '.join(rphrase) in self.phrase_by_len[len(rphrase)]:
                return '', ' '.join(rphrase), 'RIGHT'

        return text, '', 'LEFT|UNKNOWN'


class ExtractPhraseSide(StatelessTransform):
    def __init__(self, side):
        if side.lower() == 'left':
            self.part = 'left'
        elif side.lower() == 'right':
            self.part = 'right'
        else:
            raise Exception('side should either be left or right')

    def transform(self, X):
        """
        `X` is expected to be a list of `Phrasepair` (lphrase rphrase matched) instances.
        Return value is the lphrase/rphrase of the instances, with type `str`
        """
        if self.part == 'left':
            return [x.lphrase for x in X]
        else:
            return [x.rphrase for x in X]


class SubPhraseMatched(StatelessTransform):
    def transform(self, X):
        """
        `X` is expected to be a list of `Pairphrase` instances.
        Return value is the length of the instance text
        """
        return [{'matched': x.matched} for x in X]


class PhraseLengthFeature(StatelessTransform):
    def transform(self, X):
        """
        `X` is expected to be a list of `str` instances.
        Return value is the length of the instance text
        """
        return [(len(x.split()),) for x in X]


class PhraseSentimentFeature(StatelessTransform):
    def __init__(self, default_sentiment=2, prior_sentiment_dict={}):
        self.sentiment_dict = prior_sentiment_dict
        self.default_sentiment = default_sentiment

    def _get_sentiment(self, x):
        if x in self.sentiment_dict:
            return self.sentiment_dict[x]
        return self.default_sentiment

    def transform(self, X):
        """
        `X` is expected to be a list of `str` instances.
        Return value is the sentiment of the instance
        """
        return [(self._get_sentiment(x),) for x in X]


class BuildPhrasePOS(StatelessTransform):
    def transform(self, X):
        """
        `X` is expected to be a list of `str` instances.
        Return value is a list of part-of-speech tag of the instance
        """
        print 'POS Tagging...'
        return [pos_tag(word_tokenize(x)) for x in X]


class LazyPhrasePOS(StatelessTransform):
    def __init__(self):
        with open('/Users/ecsark/Projects/samr/postag.p', 'rb') as f:
            self.postag = pickle.load(f)

    def transform(self, X, y=None):
        print 'POS Tagging...'

        results = []
        for x in X:
            if x in self.postag:
                tag = self.postag[x]
            else:
                tag = pos_tag(word_tokenize(x))
            results.append(tag)
        return results


class PhraseAllPOSFeature(StatelessTransform):
    def transform(self, X, y=None):
        print 'Extracting POS Tag...'
        return [[tag for (_, tag) in x] for x in X]


class PhraseEdgePosTag(StatelessTransform):
    def __init__(self, pos):
        self.pos = pos

    def transform(self, X, y=None):
        print 'Extracting Edge POS Tag...'
        results = []
        for x in X:
            results.append({'pos': x[self.pos] if len(x) > self.pos and len(x) > 0 else ''})
        return results

class PhraseEdgeWord(StatelessTransform):
    def __init__(self, pos):
        self.pos = pos

    def transform(self, X, y=None):
        print 'Extracting Edge Word...'
        results = []
        for x in X:
            results.append({'word': x[self.pos] if len(x) > self.pos and len(x) > 0 else ''})
        return results