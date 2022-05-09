"""
This script will establish some Baseline models for the binary classification task 'Redflag or Not'
For each model it will store the predictions for the validation set  so that `redflag_ranker.py` can make use
of them to produce the ranker metrics.

Baselines:
TFIDF + SVM
TFIDF + Logistic Regression
TFIDF + RandomForest
"""

import numpy as np
import pandas as pd
import joblib
import pytrec_eval
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from data import CSVDatasetFactory

from absl import app
from absl import flags

flags.DEFINE_bool('train', False, 'Train models')
flags.DEFINE_bool('evaluate', False, 'Evaluate the best model on the Validation Dataset')
flags.DEFINE_string('trainset', None, 'CSV File for Training Dataset')
flags.DEFINE_string('valset', None, 'CSV File for Validation Dataset')
flags.DEFINE_string('save_gscv', None, 'Folder to save the GridSearchCV object')
flags.DEFINE_string('save_eval', None, 'JSON file where evaluation results are stored')

FLAGS = flags.FLAGS


# noinspection PyPep8Naming
class ClfSwitcher(BaseEstimator):
    def __init__(self, estimator=LogisticRegression()):
        """
        A Custom BaseEstimator that can switch between classifiers.
        The given classifier must implement the following methods: fit, predict, predict_proba, score
        https://stackoverflow.com/questions/48507651/multiple-classification-models-in-a-scikit-pipeline-python
        :param estimator: sklearn object - The classifier
        """

        self.estimator = estimator

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, e):
        assert all([hasattr(e, x) for x in ['fit', 'predict', 'predict_proba', 'score']])
        self._estimator = e

    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)


def ranker_evaluate_map(est, x, y):
    assert hasattr(est, 'predict_proba')
    if isinstance(y, pd.Series):
        y: pd.Series
        y: np.ndarray = y.to_numpy()

    assert isinstance(y, np.ndarray) and y.ndim == 1
    y: np.ndarray
    qrels = {'redflag': {str(i): int(v) for i, v in enumerate(y)}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map'})

    preds_nd: np.ndarray = est.predict_proba(x)   # shape: (nb_samples, 2) preds_nd[:, 1] -> prob of POSITIVE class
    preds = {'redflag': {str(i): float(v) for i, v in enumerate(preds_nd[:, 1].ravel())}}
    results = evaluator.evaluate(preds)
    return results['redflag']['map']


def main(_):
    if FLAGS.train:
        train_set = CSVDatasetFactory.get_dataset(FLAGS.trainset, 'redflag')

        # Simple pipeline: TFIDF vectorizer + Machine Learning
        # The TFIDF transform function is learnt from the TRAIN data
        # No need for a Scaler after TFIDF
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2, max_features=50000)),
            ('clf', ClfSwitcher())
        ])

        param_grids = [
            {
                'clf__estimator': [SVC(max_iter=1e4, probability=True)],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'clf__estimator__C': np.logspace(-3, 3, 7)
            },
            {
                'clf__estimator': [LogisticRegression(max_iter=1e3)],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'clf__estimator__C': np.logspace(-3, 3, 7)
            },
            {
               'clf__estimator': [RandomForestClassifier()],
               'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
               'clf__estimator__max_depth': [10, 50, 100],
               'clf__estimator__min_samples_leaf': [1, 5, 10],
               'clf__estimator__n_estimators': [100, 500, 1000]
            },
        ]

        gscv = GridSearchCV(pipeline, param_grid=param_grids,
                            scoring={'ranker_map': ranker_evaluate_map, 'classifier_map': 'average_precision'},
                            refit='ranker_map',
                            cv=5, verbose=2, n_jobs=-1)
        gscv.fit(train_set.features, train_set.targets)
        joblib.dump(gscv, FLAGS.save_gscv)

    if FLAGS.evaluate:
        gscv: GridSearchCV = joblib.load(FLAGS.save_gscv)
        val_set = CSVDatasetFactory.get_dataset(FLAGS.valset, 'redflag')

        qrels = {'redflag': {str(i): int(v) for i, v in enumerate(val_set.targets)}}
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, pytrec_eval.supported_measures)

        preds_nd: np.ndarray = gscv.predict_proba(val_set.features)
        preds = {'redflag': {str(i): float(v) for i, v in enumerate(preds_nd[:, 1].ravel())}}
        results = evaluator.evaluate(preds)['redflag']

        results.update({'model_name': repr(gscv.best_estimator_)})
        json.dump(results, open(FLAGS.save_eval, 'w'))


if __name__ == '__main__':
    app.run(main)
