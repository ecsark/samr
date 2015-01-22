"""
This module implements several scikit-learn compatible transformers, see
scikit-learn documentation for the convension fit/transform convensions.
"""

import numpy
import re

from collections import defaultdict
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import fit_ovo
import nltk

from samr.data import Lexpoint, Splitpoint


class StatelessTransform:
    """
    Base class for all transformations that do not depend on training (ie, are
    stateless).
    """
    def fit(self, X, y=None):
        return self



class ExtractText(StatelessTransform):
    """
    This should be the first transformation on a samr pipeline, it extracts
    the phrase text from the richer `Datapoint` class.
    """
    def __init__(self, lowercase=False):
        self.lowercase = lowercase

    def transform(self, X):
        """
        `X` is expected to be a list of `Datapoint` instances.
        Return value is a list of `str` instances in which words were tokenized
        and are separated by a single space " ". Optionally words are also
        lowercased depending on the argument given at __init__.
        """
        it = (" ".join(nltk.word_tokenize(datapoint.phrase)) for datapoint in X)
        if self.lowercase:
            return [x.lower() for x in it]
        return list(it)



class BuildPhraseHierarchy(StatelessTransform):

    def __init__(self, prior_dict={}):
        self.phrase_by_len = defaultdict(list)

        for text in prior_dict.keys():
            self.phrase_by_len[len(text.split())].append(text)


    def transform(self, X):     
        """
        `X` is expected to be a list of `str` instances.
        Return value is also a list of `str` instances with the replacements
        applied.
        """
        for text in X:
            self.phrase_by_len[len(text.split())].append(text)

        hdict = {}

        for i in sorted(self.phrase_by_len.keys()):
            for text in self.phrase_by_len[i]:
                lphrase, rphrase, matched = self._find_subphrase(text)
                hdict[text] = Splitpoint(lphrase, rphrase, matched)

        return [hdict[x] for x in X]


    def _find_subphrase(self, text):
        phrases = text.split()

        if len(phrases) == 1:
            return phrases[0], '', 'LEFT|SINGLE'

        for i in range(len(phrases)):
            lphrase, rphrase = phrases[:i+1], phrases[i+1:]
            if ' '.join(lphrase) in self.phrase_by_len[len(lphrase)] and ' '.join(rphrase) in self.phrase_by_len[len(rphrase)]:
                return ' '.join(lphrase), ' '.join(rphrase), 'BOTH'

        for i in range(1, len(phrases)):
            lphrase = phrases[:i+1]
            if ' '.join(lphrase) in self.phrase_by_len[len(lphrase)]:
                return ' '.join(lphrase), '', 'LEFT'

        for i in range(len(phrases)-1):
            rphrase = phrases[i+1:]
            if ' '.join(rphrase) in self.phrase_by_len[len(rphrase)]:
                return '', ' '.join(rphrase), 'RIGHT'

        return text, '', 'LEFT|UNKNOWN'


class BuildLexHierarchy(StatelessTransform):

    def __init__(self, prior_dict={}):
        self.phrase_by_len = defaultdict(list)

        for text in prior_dict.keys():
            self.phrase_by_len[len(text.split())].append(text)


    def transform(self, X):     

        for text in X:
            self.phrase_by_len[len(text.split())].append(text)

        hdict = {}

        for i in sorted(self.phrase_by_len.keys()):
            for text in self.phrase_by_len[i]:
                word, subphrase, pos = self._find_subphrase(text)
                hdict[text] = Lexpoint(word, subphrase, pos)

        return [hdict[x] for x in X]


    def _find_subphrase(self, text):
        word, subphrase, pos = text.split()[0], ' '.join(text.split()[1:]), 'LEFT'

        if subphrase not in self.phrase_by_len[len(subphrase.split())]:
            word, subphrase, pos = text.split()[-1], ' '.join(text.split()[:-1]), 'RIGHT'

            if subphrase not in self.phrase_by_len[len(subphrase.split())]:
                pos = 'DEFAULT'

        return word, subphrase, pos



class ReplaceText(StatelessTransform):
    def __init__(self, replacements):
        """
        Replacements should be a list of `(from, to)` tuples of strings.
        """
        self.rdict = dict(replacements)
        self.pat = re.compile("|".join(re.escape(origin) for origin, _ in replacements))

    def transform(self, X):
        """
        `X` is expected to be a list of `str` instances.
        Return value is also a list of `str` instances with the replacements
        applied.
        """
        if not self.rdict:
            return X
        return [self.pat.sub(self._repl_fun, x) for x in X]

    def _repl_fun(self, match):
        return self.rdict[match.group()]


class MapToSynsets(StatelessTransform):
    """
    This transformation replaces words in the input with their Wordnet
    synsets[0].
    The intuition behind it is that phrases represented by synset vectors
    should be "closer" to one another (not suffer the curse of dimensionality)
    than the sparser (often poetical) words used for the reviews.

    [0] For example "bank": http://wordnetweb.princeton.edu/perl/webwn?s=bank
    """
    def transform(self, X):
        """
        `X` is expected to be a list of `str` instances.
        It returns a list of `str` instances such that the i-th element
        containins the names of the synsets of all the words in `X[i]`,
        excluding noun synsets.
        `X[i]` is internally tokenized using `str.split`, so it should be
        formatted accordingly.
        """
        return [self._text_to_synsets(x) for x in X]

    def _text_to_synsets(self, text):
        result = []
        for word in text.split():
            ss = nltk.wordnet.wordnet.synsets(word)
            result.extend(str(s) for s in ss if ".n." not in str(s))
        return " ".join(result)


class Densifier(StatelessTransform):
    """
    A transformation that densifies an scipy sparse matrix into a numpy ndarray
    """
    def transform(self, X, y=None):
        """
        `X` is expected to be a scipy sparse matrix.
        It returns `X` in a (dense) numpy ndarray.
        """
        return X.todense()


class ClassifierOvOAsFeatures:
    """
    A transformation that esentially implement a form of dimensionality
    reduction.
    This class uses a fast SGDClassifier configured like a linear SVM to produce
    a vector of decision functions separating target classes in a
    one-versus-rest fashion.
    It's useful to reduce the dimension bag-of-words feature-set into features
    that are richer in information.
    """
    def fit(self, X, y):
        """
        `X` is expected to be an array-like or a sparse matrix.
        `y` is expected to be an array-like containing the classes to learn.
        """
        self.classifiers = fit_ovo(SGDClassifier(), X, numpy.array(y), n_jobs=-1)[0]
        return self

    def transform(self, X, y=None):
        """
        `X` is expected to be an array-like or a sparse matrix.
        It returns a dense matrix of shape (n_samples, m_features) where
            m_features = (n_classes * (n_classes - 1)) / 2
        """
        xs = [clf.decision_function(X).reshape(-1, 1) for clf in self.classifiers]
        return numpy.hstack(xs)
