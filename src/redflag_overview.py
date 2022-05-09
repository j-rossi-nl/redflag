"""
Collects all evaluations and organize a summary.

Builds and display the overview.
Assumes that redflag_ranker has been run. All results JSON are saved in sub-folders of models_root.
$ python redflag_overview.py --build --display --models_root ../models --baseline_json ../models/ml.json
    --save_csv overview.csv


$ Only displays
$ python redflag_overview.py --display --save_csv overview.csv


"""

import pandas as pd
import glob
import os
import re
import json

from typing import Dict

from absl import app
from absl import flags

flags.DEFINE_bool('build', False, 'Create the overview and save it')
flags.DEFINE_bool('display', False, 'Display a summary')
flags.DEFINE_string('models_root', None, 'Path to parent folder of all models output_dir')
flags.DEFINE_string('baseline_json', None, 'Path to JSON results of baseline evaluation')
flags.DEFINE_string('save_csv', None, 'Path to CSV file to save the overview')

FLAGS = flags.FLAGS


def main(_):
    if FLAGS.build:
        # For the models: which subfolders have files 'eval.xxx.json' ?
        eval_files = glob.glob(os.path.join(FLAGS.models_root, '*', 'eval.*.json'), recursive=True)
        data = []
        for f in eval_files:
            results: Dict = json.load(open(f, 'r'))['redflag']
            results.update({'path': f})
            data.append(results)

        df = pd.DataFrame(data)
        df['model_name'] = df['path'].apply(lambda x: os.path.basename(os.path.dirname(x)))
        df['shortname'] = df['model_name']
        df['epoch'] = df['path'].apply(
            lambda x: re.match(r'eval\.(?P<epoch>\d{3})\.json', os.path.basename(x)).group('epoch')
        ).astype(int)

        baseline: Dict = json.load(open(FLAGS.baseline_json, 'r'))
        baseline.update({'shortname': 'TFIDF + ML'})
        df = df.append(baseline, ignore_index=True)

        df.to_csv(FLAGS.save_csv, index=False)

    if FLAGS.display:
        df = pd.read_csv(FLAGS.save_csv).fillna(0)
        print(df.loc[df.groupby('shortname')['map'].idxmax()][['shortname', 'epoch', 'map']])


if __name__ == '__main__':
    app.run(main)