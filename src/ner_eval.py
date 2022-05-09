import pickle
import os
import pandas as pd

from typing import List, Dict, Any
from sklearn.metrics import classification_report
from absl import app
from absl import flags

from ner_evaluation.ner_eval import Evaluator


flags.DEFINE_bool('reports', False, 'Produce the classification reports')
flags.DEFINE_bool('paper', False, 'Figures for the paper')
flags.DEFINE_string('true', None, 'CoNLL File with TRUE entities')
flags.DEFINE_string('pred', None, 'CoNLL File with PREDIDCTED entities')
flags.DEFINE_string('dst', None, 'Folder where output is saved')
FLAGS = flags.FLAGS


def f_to_s(fn: str) -> List[List[Dict[str, str]]]:
    """
    File to Sentences.
    Given a file in CoNLL format, create a List of List of Dict with keys `token` and `label`
    Example, with 2 sentences:
    [[{'token': 'done', 'label': 'O'}, {...}], [{'token': '...', 'label': '...'}, ...]]
    :param fn: Filename
    :return: List of List of Dict
    """
    data = []
    curr_sent = []
    for line in open(fn):
        if line.isspace():
            data.append(curr_sent)
            curr_sent = []
        else:
            sp = line.split()
            curr_sent.append({'token': sp[0], 'label': sp[1]})
    return data


def d_to_l(d: List[List[Dict[str, str]]]) -> List[List[str]]:
    """
    Returns the list of labels for the data.
    See f_to_s to get the proper data.
    :param d: List[List[Dict[str,str]]
    :return:
    """
    return list(map(lambda s: [x['label'] for x in s], d))


def main(_):
    if FLAGS.reports:
        true_data = f_to_s(FLAGS.true)
        pred_data = f_to_s(FLAGS.pred)

        true_labels = d_to_l(true_data)
        pred_labels = d_to_l(pred_data)

        labels = set([token['label'] for sentence in true_data for token in sentence])
        labels.remove('O')
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))

        true_labels_flat, pred_labels_flat = list(map(lambda l: [item for sublist in l for item in sublist],
                                                      (true_labels, pred_labels)))
        cls_report = classification_report(y_true=true_labels_flat, y_pred=pred_labels_flat,
                                           output_dict=True, zero_division=0)

        original_labels = set([x[2:] for x in sorted_labels])
        evaluator = Evaluator(true=true_labels, pred=pred_labels, tags=original_labels)

        res, agg = evaluator.evaluate()

        for k, v in {'classification_report.pkl': cls_report, 'res.pkl': res, 'agg.pkl': agg}.items():
            with open(os.path.join(FLAGS.dst, k), 'wb') as out:
                pickle.dump(v, out)

    if FLAGS.paper:
        cr: Dict[str, Any] = pickle.load(open(os.path.join(FLAGS.dst, 'classification_report.pkl'), 'rb'))
        df = pd.DataFrame(cr).transpose()
        sorted_labels = sorted(df.index, key=lambda name: (name[1:], name[0]))
        df: pd.DataFrame = df.reindex(sorted_labels).drop(['O', 'macro avg', 'accuracy', 'weighted avg'])

        total_support = df['support'].sum()

        macro_avgs = {'support': total_support}
        weighted_avgs = {'support': total_support}
        for m in ['precision', 'recall', 'f1-score']:
            macro_avgs[m] = df[m].mean()
            weighted_avgs[m] = (df[m] * df['support']).sum() / total_support

        df.loc['macro avg'] = macro_avgs
        df.loc['weighted avg'] = weighted_avgs

        # load it with pd.read_csv(..., index_col=0)
        df.to_csv(os.path.join(FLAGS.dst, 'paper_report.csv'), index=True)

if __name__ == '__main__':
    app.run(main)
