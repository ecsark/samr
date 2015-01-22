import argparse
import json
import csv
import sys
import random

from samr.corpus import iter_corpus, iter_test_corpus
from samr.predictor import PhraseSentimentPredictor


#parser = argparse.ArgumentParser(description=__doc__)
#config = json.load(open('/Users/ecsark/Projects/samr/data/model2.json'))

#predictor = PhraseSentimentPredictor(**config)
raw_set = list(iter_corpus())

use_pct = int(0.3 * len(raw_set))

data_set = raw_set[:use_pct]

train_num = int(0.8 * len(data_set))

random.shuffle(data_set)

y = [int(d.sentiment) for d in data_set]
# train_set, dev_set = data_set[:train_num], data_set[train_num:]
#predictor.fit(train_set)
#predictor.score(dev_set)

#####################################
from samr.relation_lex_transform import *
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from samr.transformations import (ExtractText, ReplaceText, MapToSynsets,
                                  Densifier, ClassifierOvOAsFeatures)

pipeline = Pipeline([
	('split_into_subphrases', BuildSubPhrase()),
	('features', FeatureUnion([
		('sub_pos', Pipeline([
			#('pos_tag', BuildPhrasePOS()),
			('pos_tag', LazyPhrasePOS()),
			('extract_pos', PhraseAllPOSFeature()),
			('edge_pos', PhraseEdgeFeature()),
			('tovec', CountVectorizer(binary=True)),
			('dense', ClassifierOvOAsFeatures())

			])),
		('sub_length', PhraseLengthFeature()),
		('sub_sentiment', PhraseSentimentFeature())
		], n_jobs=-1))
	])

Z = pipeline.fit_transform(data_set, y)

train_set, dev_set = Z[:train_num], Z[train_num:]
train_ans, dev_ans = y[:train_num], y[train_num:]

#classifier = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1)
#classifier.fit(train_set, train_ans)
regressor = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, n_jobs=-1)
regressor.fit(train_set, train_ans)

pred = regressor.predict(dev_set)

