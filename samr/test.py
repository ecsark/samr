import csv
import random

from sklearn.metrics import mean_absolute_error

from samr.corpus import iter_corpus, iter_test_corpus
from samr.predictor import DuplicatesHandler
from samr.transformations import ClassifierOvOAsFeatures

raw_set = list(iter_corpus())
test_set = list(iter_test_corpus())

#use_pct = int(0.3 * len(raw_set))
use_pct = len(raw_set)

data_set = raw_set[:use_pct]
gold_ans = [int(d.sentiment) for d in data_set]

# train_num = int(0.9 * len(data_set))
train_num = len(data_set)

# rand = random.Random()
# rand.seed(4721)
# rand.shuffle(data_set)




#####################################

from samr.relation_lex_transform import *
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
import numpy
"""
sentiment_bank = {}
for d in data_set:
    sentiment_bank[d.phrase] = d.sentiment
"""
phrase_bank = [x.phrase for x in data_set]
test_phrase_bank = [x.phrase for x in test_set]

point_dict = {}
for d in data_set:
    point_dict[d.phrase] = d


class AugmentedPredictor():
    def __init__(self, duplicates=False):
        self.left_predictor = PhraseSentimentPredictor(classifier="randomforest", map_to_synsets=True,
                                                       map_to_lex=True, duplicates=True,
                                                       classifier_args={"n_estimators": 100, "min_samples_leaf":10, "n_jobs":-1})
        self.right_predictor = PhraseSentimentPredictor(classifier="randomforest", map_to_synsets=True,
                                                        map_to_lex=True, duplicates=True,
                                                        classifier_args={"n_estimators": 100, "min_samples_leaf":10, "n_jobs":-1})
        self.this_predictor = PhraseSentimentPredictor(classifier="randomforest", map_to_synsets=True,
                                                       map_to_lex=True, duplicates=True,
                                                       classifier_args={"n_estimators": 100, "min_samples_leaf":10, "n_jobs":-1})
        self.duplicates = duplicates
        """
        self.pipeline = Pipeline([
            ('split_phrase', BuildSubPhrase(prior_phrase=phrase_bank)),
            ('features', FeatureUnion([
                ('match_pipeline', Pipeline([
                    ('matched', SubPhraseMatched()),
                    ('m_to_dict', DictVectorizer(sparse=False)),
                    ])),
                ('left_features', self.build_sub_phrase_feature('left')),
                ('right_features', self.build_sub_phrase_feature('right')),
                ]))
        ])
        """
        self.classifier = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1)
        self.decider = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1)

    def adapt_phrase_to_point(self, X):
        phrase_to_point = {}
        for x in X:
            phrase_to_point[x.phrase] = x
        return phrase_to_point


    def phrase_to_point(self, X):
        return [Datapoint(None, None, x, None) for x in X]

    def tuple_to_vector(self, X):
        return [(x,) for x in X]

    def fit(self, X, y=None):
        y = [int(x.sentiment) for x in X]

        if self.duplicates:
            self.dupes = DuplicatesHandler()
            self.dupes.fit(X, y)

        self.left_predictor.fit(X)
        self.right_predictor.fit(X)
        self.this_predictor.fit(X)

        P = self.this_predictor.pipeline.fit_transform(X, y)
        P = [list(numpy.array(p).reshape(-1)) for p in P]

        W = BuildSubPhrase(prior_phrase=phrase_bank).transform(X)

        self.matched_ppl = make_pipeline(SubPhraseMatched(), DictVectorizer(sparse=False))
        matched = self.matched_ppl.fit_transform(W, y)

        self.left_extract = ExtractSidePhrase('left')
        self.right_extract = ExtractSidePhrase('right')
        L = self.left_extract.transform(W)
        R = self.right_extract.transform(W)

        left_y = self.left_predictor.predict(self.phrase_to_point(L))
        right_y = self.right_predictor.predict(self.phrase_to_point(R))

        self.left_phr_ppl = self.build_sub_phrase_feature()
        self.right_phr_ppl = self.build_sub_phrase_feature()

        self.left_pos_ppl = self.build_edge_pos_feature('left')
        self.right_pos_ppl = self.build_edge_pos_feature('right')

        left_phr_feat = self.left_phr_ppl.fit_transform(L, left_y)
        right_phr_feat = self.right_phr_ppl.fit_transform(R, right_y)

        left_pos_feat = self.left_pos_ppl.fit_transform(L, y)
        right_pos_feat = self.right_pos_ppl.fit_transform(R, y)

        Z = self.pack_feature([matched, left_phr_feat, left_pos_feat, right_phr_feat, right_pos_feat])

        Z = [f + [int(f[6]) - int(f[18]), int(f[6]) + int(f[18])] for f in Z]  # sentiment difference, sentiment strength
        self.classifier.fit(Z, y)

        U = self.pack_feature([P, Z])
        self.decider.fit(U, y)
        return self

    def predict(self, X):

        P = self.this_predictor.pipeline.transform(X)
        P = [list(numpy.array(p).reshape(-1)) for p in P]

        W = BuildSubPhrase(prior_phrase=test_phrase_bank).transform(X)
        matched = self.matched_ppl.transform(W)

        L = self.left_extract.transform(W)
        R = self.right_extract.transform(W)

        left_y = self.left_predictor.predict(self.phrase_to_point(L))
        right_y = self.right_predictor.predict(self.phrase_to_point(R))

        left_phr_feat = self.left_phr_ppl.fit_transform(L, left_y)
        right_phr_feat = self.right_phr_ppl.fit_transform(R, right_y)

        left_pos_feat = self.left_pos_ppl.transform(L)
        right_pos_feat = self.right_pos_ppl.transform(R)

        Z = self.pack_feature([matched, left_phr_feat, left_pos_feat, right_phr_feat, right_pos_feat])

        Z = [f + [int(f[6]) - int(f[18]), int(f[6]) + int(f[18])] for f in Z]  # sentiment difference, sentiment strength
        # y = self.classifier.predict(Z)

        U = self.pack_feature([P, Z])
        y = self.decider.predict(U)

        if self.duplicates:
            for i, phrase in enumerate(X):
                label = self.dupes.get(phrase)
                if label is not None:
                    y[i] = label
        return y

    def pack_feature(self, features_list):
        packed = []
        for i in range(len(features_list[0])):
            packed.append([x for f in features_list for x in f[i]])
        return packed

    def build_edge_pos_feature(self, side):
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

    def build_sub_phrase_feature(self, prior_sent_dict=None):
        return make_union(PhraseLengthFeature(),
                          PhraseSentimentFeature(prior_sentiment_dict=prior_sent_dict))


train_set, dev_set = data_set[:train_num], data_set[train_num:]
train_ans, dev_ans = gold_ans[:train_num], gold_ans[train_num:]

predictor = AugmentedPredictor()
predictor.fit(train_set)
# prediction = predictor.predict(dev_set)

prediction = predictor.predict(test_set)

#print mean_absolute_error(dev_ans, prediction)

def make_submission(test_set, prediction):
    f = open('/Users/ecsark/Projects/samr/submission/submission.csv', 'wb')
    writer = csv.writer(f)
    writer.writerow(("PhraseId", "Sentiment"))
    for datapoint, sentiment in zip(test_set, prediction):
        writer.writerow((datapoint.phraseid, sentiment))
    f.close()


make_submission(test_set, prediction)

"""
Z = pipeline.fit_transform(data_set, y)

Z = [f.tolist() + [int(f[6]) - int(f[18]), int(f[6]) + int(f[18])] for f in
     Z]  # sentiment difference, sentiment strength

train_set, dev_set = Z[:train_num], Z[train_num:]
train_ans, dev_ans = y[:train_num], y[train_num:]

classifier = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1)

classifier.fit(train_set, train_ans)

dev_guess = classifier.predict(dev_set)

print mean_absolute_error(dev_guess, dev_ans)
"""