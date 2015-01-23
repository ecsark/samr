import random

from sklearn.metrics import mean_absolute_error

from samr.corpus import iter_corpus
from samr.transformations import ClassifierOvOAsFeatures

raw_set = list(iter_corpus())

use_pct = int(0.3 * len(raw_set))

data_set = raw_set[:use_pct]

train_num = int(0.9 * len(data_set))

rand = random.Random()
rand.seed(4721)
rand.shuffle(data_set)

y = [int(d.sentiment) for d in data_set]

#####################################

from samr.relation_lex_transform import *
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

sentiment_bank = {}
for d in data_set:
    sentiment_bank[d.phrase] = d.sentiment

phrase_bank = [x.phrase for x in data_set]
"""
split_phrase = BuildSubPhrase(prior_phrase=phrase_bank).transform(data_set)
matched = DictVectorizer(sparse=False).transform(SubPhraseMatched().transform(split_phrase))

left_extract = ExtractPhraseSide('left').transform(split_phrase)
right_extract = ExtractPhraseSide('right').transform(split_phrase)

phrase_len_feature = PhraseLengthFeature()
left_len = phrase_len_feature.transform(left_extract)
right_len = phrase_len_feature.transform(right_extract)

phrase_sentiment_feature = PhraseSentimentFeature(prior_sentiment_dict=sentiment_bank)
left_sent = phrase_sentiment_feature.transform(left_extract)
right_sent = phrase_sentiment_feature.transform(right_extract)

phrase_pos_tagger = LazyPhrasePOS()
phrase_pos_tag_only = PhraseAllPOSFeature()
left_phrase_pos = phrase_pos_tag_only.transform(phrase_pos_tagger.transform(left_extract))
left_pos_vec = DictVectorizer(sparse=False).fit_transform(PhraseEdgeFeature(-1).transform(left_phrase_pos))
right_phrase_pos = phrase_pos_tag_only.transform(phrase_pos_tagger.transform(right_extract))
right_pos_vec = DictVectorizer(sparse=False).fit_transform(PhraseEdgeFeature(0).transform(right_phrase_pos))
"""


def _build_edge_pos_feature(side):
    if side.lower() == 'left':
        position = -1
    elif side.lower() == 'right':
        position = 0
    else:
        raise Exception('side should either be left or right')
    return make_pipeline(
        LazyPhrasePOS(),
        PhraseAllPOSFeature(),
        PhraseEdgePosTag(position),
        DictVectorizer(),
        ClassifierOvOAsFeatures()
    )


def build_sub_phrase_feature(side):
    if side.lower() not in ['left', 'right']:
        raise Exception('side should either be left or right')
    return make_pipeline(ExtractPhraseSide(side),
                         make_union(PhraseLengthFeature(),
                                    PhraseSentimentFeature(prior_sentiment_dict=sentiment_bank),
                                    _build_edge_pos_feature(side))
    )


pipeline = Pipeline([
    ('split_phrase', BuildSubPhrase(prior_phrase=phrase_bank)),
    ('features', FeatureUnion([
        ('match_pipeline', Pipeline([
            ('matched', SubPhraseMatched()),
            ('m_to_dict', DictVectorizer(sparse=False)),
        ])),
        ('left_features', build_sub_phrase_feature('left')),
        ('right_features', build_sub_phrase_feature('right')),
    ]))
])

Z = pipeline.fit_transform(data_set, y)

Z = [f.tolist() + [int(f[6])-int(f[18]), int(f[6])+int(f[18])] for f in Z]  # sentiment difference, sentiment strength

train_set, dev_set = Z[:train_num], Z[train_num:]
train_ans, dev_ans = y[:train_num], y[train_num:]

classifier = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1)

classifier.fit(train_set, train_ans)

dev_guess = classifier.predict(dev_set)

print mean_absolute_error(dev_guess, dev_ans)
