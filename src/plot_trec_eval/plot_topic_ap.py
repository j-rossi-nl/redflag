import matplotlib
import numpy as np
import argparse
import re
import matplotlib.pyplot as plt

from . eval_results import EvaluationResult

matplotlib.use('Agg')


def natural_sort(list_):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(list_, key=alphanum_key)


def plot_single(filepath, outfile, sort_on_ap=False):
    r = EvaluationResult(filepath)
    topic_names = natural_sort(r.queries.keys())
    ap = np.array([r.queries[q]['map'] for q in topic_names])
    ylabel = 'Average Precision'

    plot(ap, ylabel, topic_names, outfile, sort_on_ap)


def plot_double(filepath_1, filepath_2, outfile, sort_on_ap=False):
    r1 = EvaluationResult(filepath_1)
    r2 = EvaluationResult(filepath_2)

    assert all([k in r2.queries for k in r1.queries.keys()]), 'Topic set is not the same!'
    assert all([k in r1.queries for k in r2.queries.keys()]), 'Topic set is not the same!'

    topic_names = natural_sort(r1.queries.keys())

    ap1 = np.array([r1.queries[q]['map'] for q in topic_names])
    ap2 = np.array([r2.queries[q]['map'] for q in topic_names])

    ap = ap1 - ap2
    ylabel = 'Average Precision difference'

    plot(ap, ylabel, topic_names, outfile, sort_on_ap)


def plot(ap, ylabel, topic_names, outfile, sort_on_ap=False):
    if sort_on_ap:
        topic_names = [n for (d, n) in sorted(zip(ap, topic_names), reverse=True)]
        ap = sorted(ap, reverse=True)

    ind = np.arange(len(topic_names))
    width = 1.0
    plt.bar(ind, ap, width)
    plt.xticks(ind + 0.5 * width, topic_names, fontsize=10, rotation='vertical')
    plt.ylabel(ylabel)
    plt.xlabel('Topic')

    plt.savefig(outfile, bbox_inches='tight')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'Plot AP per topic for 1 run or per-topic difference for 2 runs.')
    argparser.add_argument('-f', '--output', help='Save the figure to specified file.',
                           default='topic_ap.pdf')
    argparser.add_argument('-s', '--sort', help='Sort the topics in descending AP/difference.',
                           type=bool, required=False, default=False)
    argparser.add_argument('files',
                           help='When passing two files the plotted difference is f1-f2.',
                           type=str, nargs='+')

    args = argparser.parse_args()
    if len(args.files) > 1:
        plot_double(args.files[0], args.files[1], args.output, args.sort)
    else:
        plot_single(args.files[0], args.output, args.sort)
