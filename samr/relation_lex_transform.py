from collections import defaultdict
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

from samr.transformations import StatelessTransform
from samr.data import *


class BuildSubPhrase2():

    def fit(self, X, y=None):
    	"""
    	Input: X: Datapoint (phraseid, sentenceid, phrase, sentiment)
    	"""
        self.phrase_by_len = defaultdict(list)
        self.phrase_to_dp = {}
    	for datapoint in X:
        	text = datapoint.phrase
        	self.phrase_by_len[len(text.split())].append(datapoint)
        	self.phrase_to_dp[text] = datapoint

        return self


    def transform(self, X):
    	"""
    	Input: Datapoint (phraseid, sentenceid, phrase, sentiment)
    	Return: Phrasepair (lphrase, rphrase)
    	"""
    	hdict = {}

        for phlen in sorted(self.phrase_by_len.keys()):
            for datapoint in self.phrase_by_len[phlen]:
                lphrase, rphrase, matched = self._find_subpoint(datapoint)
                hdict[text] = Phrasepair(lphrase, rphrase)

        return [hdict[x.phrase] for x in X]


    def _find_subpoint(self, datapoint):
        phrases = datapoint.phrase.split()

        if len(phrases) == 1:
            return phrases[0], '', 'LEFT|SINGLE'



        for i in range(len(phrases)):
            lphrase, rphrase = phrases[:i+1], phrases[i+1:]
            if _phrase_exist(lphrase) and 
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


class BuildPhraseSentiment():
	def __init__(self, default_sentiment=2):
        self.default_sentiment = default_sentiment

    def fit(self, X, y):
    	self.sentiment_bank = {}
    	for (datapoint, sentiment) in zip(X,y):
    		self.sentiment_bank[datapoint.phrase] = sentiment

    	return self


class BuildSubPhrase():

    def __init__(self, prior_dict={}, default_sentiment=2, verbose=False):
        self.phrase_by_len = defaultdict(list)

        for text in prior_dict.keys():
            self.phrase_by_len[len(text.split())].append(text)

        self.default_sentiment = default_sentiment
        self.verbose = verbose


    def fit(self, X, y):
    	self.sentiment_bank = {}
    	for (datapoint, sentiment) in zip(X,y):
    		self.sentiment_bank[datapoint.phrase] = sentiment

    	return self


    def _find_sentiment(self, text):
    	assert(isinstance(text,str))

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
                hdict[text] = Sentimentpoint(lphrase, rphrase, self._find_sentiment(lphrase), self._find_sentiment(rphrase))

        return [hdict[x.phrase] for x in X]


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



class PhraseLengthFeature(StatelessTransform):
    def transform(self, X, y=None):
    	print 'Extracting Sub-phrase Length...'
    	return [(len(x.lphrase), len(x.rphrase)) for x in X]


class PhraseSentimentFeature(StatelessTransform):
	def transform(self, X, y=None):
		print 'Extracting Sub-phrase Sentiment...'
		return [(x.lsentiment, x.rsentiment) for x in X]


class BuildPhrasePOS(StatelessTransform):
	def transform(self, X, y=None):
		print 'POS Tagging...'
		return [(pos_tag(word_tokenize(x.lphrase)), pos_tag(word_tokenize(x.rphrase))) for x in X]

class LazyPhrasePOS(StatelessTransform):
	def transform(self, X, y=None):
		print 'POS Tagging...'
		import cPickle as pickle
		with open('/Users/ecsark/Projects/samr/postag.p', 'rb') as f:
			postag = pickle.load(f)
		
		results = []
		for x in X:
			if x.lphrase in postag:
				ltag = postag[x.lphrase]
			else:
				ltag = pos_tag(word_tokenize(x.lphrase))
			if x.rphrase in postag:
				rtag = postag[x.rphrase]
			else:
				rtag = pos_tag(word_tokenize(x.rphrase)) 
			results.append((ltag, rtag))
		return results


class PhraseAllPOSFeature(StatelessTransform):
	def transform(self, X, y=None):
		print 'Extracting POS Tag...'
		return [([tag for (_, tag) in x[0]], [tag for (_, tag) in x[1]]) for x in X]


class PhraseEdgeFeature(StatelessTransform):
	def transform(self, X, y=None):
		print 'Extracting Edge POS Tag...'
		results = []
		for (xlpos, xrpos) in X:
			results.append((xlpos[-1] if len(xlpos)>0 else 'NULL', xrpos[0] if len(xrpos)>0 else 'NULL'))
		return results
