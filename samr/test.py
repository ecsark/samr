import random

from sklearn.metrics import mean_absolute_error

from samr.corpus import iter_corpus
from samr.transformations import ClassifierOvOAsFeatures

raw_set = list(iter_corpus())

use_pct = int(0.3 * len(raw_set))

data_set = raw_set[:use_pct]

train_num = int(0.8 * len(data_set))

rand = random.Random()
rand.seed(4721)
rand.shuffle(data_set)

y = [int(d.sentiment) for d in data_set]

#####################################

from samr.relation_lex_transform import *
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

sentiment_bank = {}
for d in data_set:
    sentiment_bank[d.phrase] = d.sentiment

phrase_bank = [x.phrase for x in data_set]

pipeline = Pipeline([
    ('split_phrase', BuildSubPhrase(prior_phrase=phrase_bank)),
    ('features', FeatureUnion([
        ('match_pipeline', Pipeline([
            ('matched', SubPhraseMatched()),
            ('m_to_dict', DictVectorizer(sparse=False)),
            # ('m_pos_ovo', ClassifierOvOAsFeatures())
        ])),

        ('left_pipeline', Pipeline([
            ('left_extract', ExtractPhraseSide('left')),
            ('left_feature', FeatureUnion([
                ('l_length', PhraseLengthFeature()),
                ('l_sentiment', PhraseSentimentFeature(prior_sentiment_dict=sentiment_bank)),
                ('l_pos', Pipeline([
                    ('l_load_pos', LazyPhrasePOS()),
                    ('l_pos_only', PhraseAllPOSFeature()),
                    ('l_edge_pos', PhraseEdgeFeature(-1)),
                    ('l_pos_to_dict', DictVectorizer(sparse=False)),
                    ('l_pos_ovo', ClassifierOvOAsFeatures())
                ]))
            ]))
        ])),
        ('right_pipeline', Pipeline([
            ('right_extract', ExtractPhraseSide('right')),
            ('right_feature', FeatureUnion([
                ('r_length', PhraseLengthFeature()),
                ('r_sentiment', PhraseSentimentFeature(prior_sentiment_dict=sentiment_bank)),
                ('r_pos', Pipeline([
                    ('r_load_pos', LazyPhrasePOS()),
                    ('r_pos_only', PhraseAllPOSFeature()),
                    ('r_edge_pos', PhraseEdgeFeature(0)),
                    ('r_pos_to_dict', DictVectorizer(sparse=False)),
                    ('r_pos_ovo', ClassifierOvOAsFeatures())
                ]))
          ]))
        ]))
    ]))
])

Z = pipeline.fit_transform(data_set, y)

train_set, dev_set = Z[:train_num], Z[train_num:]
train_ans, dev_ans = y[:train_num], y[train_num:]

classifier = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1)
classifier.fit(train_set, train_ans)

# regressor = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, n_jobs=-1)
# regressor.fit(train_set, train_ans)

dev_guess = classifier.predict(dev_set)

print mean_absolute_error(dev_guess, dev_ans)
