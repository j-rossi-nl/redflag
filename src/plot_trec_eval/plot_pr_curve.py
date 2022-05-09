import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from typing import List, Dict
from . eval_results import EvaluationResult

matplotlib.use('Agg')


def process(files: List[str] = None, list_results: List[Dict] = None, outfile: str = None):
    result_list = []
    if files is not None:
        result_list = [EvaluationResult(filepath=f) for f in files]
    elif list_results is not None:
        result_list = [EvaluationResult(results=r) for r in list_results]

    names = [r.runid for r in result_list]
    iprec = [[r.results['iprec_at_recall_0.00'],
              r.results['iprec_at_recall_0.10'],
              r.results['iprec_at_recall_0.20'],
              r.results['iprec_at_recall_0.30'],
              r.results['iprec_at_recall_0.40'],
              r.results['iprec_at_recall_0.50'],
              r.results['iprec_at_recall_0.60'],
              r.results['iprec_at_recall_0.70'],
              r.results['iprec_at_recall_0.80'],
              r.results['iprec_at_recall_0.90'],
              r.results['iprec_at_recall_1.00']] for r in result_list]

    recall = np.arange(0, 1.1, 0.1)

    plt.clf()
    plt.xlabel('Recall')
    plt.ylabel('Interpolated Precision')

    for p in iprec:
        plt.plot(recall, p)

    plt.legend(names)
    plt.yticks(np.arange(0.0, 1.2, 0.2))
    plt.xticks(np.arange(0.0, 1.2, 0.2))
    plt.xlim((-0.01, 1.01))
    plt.ylim((-0.01, 1.01))

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
    return plt.gcf()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Plot precision-recall curves.')
    argparser.add_argument('-f', '--output', help='Save the figure to specified file.',
                           default='pr_curve.pdf', required=False)
    argparser.add_argument('files',
                           help='Pass multiple files to plot all the runs in the same plot.',
                           type=str, nargs='+')
    args = argparser.parse_args()

    process(files=args.files, outfile=args.output)
