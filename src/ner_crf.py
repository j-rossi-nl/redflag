"""
From https://github.com/davidsbatista/NER-Evaluation/blob/master/example-full-named-entity-evaluation.ipynb
"""

import sklearn_crfsuite
import pickle
import os

from typing import List, Tuple

from absl import app
from absl import flags

flags.DEFINE_bool('train', None, 'Train')
flags.DEFINE_bool('predict', None, 'Predict')
flags.DEFINE_string('trainset', None, 'Training data')
flags.DEFINE_string('valset', None, 'Validation data')
flags.DEFINE_string('output_dir', None, 'Folder where model and predictions are saved')
FLAGS = flags.FLAGS


def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]


def sent2tokens(sent):
    return [token for token, label in sent]


def conll_file_to_sents(conll: str) -> List[List[Tuple[str, str]]]:
    with open(conll, 'r') as src:
        all_sents = []
        curr_sent = []
        for line in src:
            if line.isspace():
                all_sents.append(curr_sent)
                curr_sent = []
                continue
            token, tag = line.split()
            curr_sent.append((token, tag))

    return all_sents


def main(_):
    if FLAGS.train:
        # ## Train a CRF
        print('Collect and Prepare Data')
        train_sents = conll_file_to_sents(FLAGS.trainset)

        x_train = [sent2features(s) for s in train_sents]
        y_train = [sent2labels(s) for s in train_sents]

        # ## Training
        print('Train CRF...')
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(x_train, y_train)

        print('Save CRF')
        pickle.dump(crf, open(os.path.join(FLAGS.output_dir, 'crf.pkl'), 'wb'))

    if FLAGS.predict:
        print('Collect and Prepare Data')
        test_sents = conll_file_to_sents(FLAGS.valset)
        x_test = [sent2features(s) for s in test_sents]

        print('Predict')
        crf: sklearn_crfsuite.CRF = pickle.load(open(os.path.join(FLAGS.output_dir, 'crf.pkl'), 'rb'))
        y_pred = crf.predict(x_test)

        assert len(test_sents) == len(y_pred)

        print('Store results')
        with open(os.path.join(FLAGS.output_dir, 'test_predictions.txt'), 'w') as out:
            for sent, preds in zip(test_sents, y_pred):
                for (token, _), tag in zip(sent, preds):
                    out.write(f'{token} {tag}\n')
                out.write('\n')


if __name__ == '__main__':
    app.run(main)
