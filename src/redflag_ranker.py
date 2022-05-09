"""
This script will use a trained classifier as a ranker.
The classifier is trained on discrimating whether there is a redflag in a text (relevance assessment).
We use the estimated probability of the positive class to rank all texts.

Script usage:
Generate the predictions for each sample, using a model checkpoint at a specific epoch.
The predictions are a numpy array (nb_samples, 1), pickled in a file preds.<epoch>.pkl
$ python redflag_ranker.py predict --valset ../data/valdata.csv --model_dir ../models/ --epoch 7
  --albert_name albert-base-v2 --output_dir ../results/

Generate the predictions for each sample, using all model checkpoints.
The predictions are a numpy array (nb_samples, 1), pickled in files preds.<epoch>.pkl, with as many files as checkpoints
$ python redflag_ranker.py predict --valset ../data/valdata.csv --model_dir ../models/ --all_epochs
  --albert_name albert-base-v2 --output_dir ../results/

Create the reports, using trec-eval.
$ python redflag_ranker.py --evaluate --valset ../data/valdata.csv--output_dir ../models/

Same, and log the PR curves on a COMET experience.
$ python redflag_ranker.py --evaluate --valset ../data/valdata.csv --output_dir ../models/ --comet_exp <exp_key>

2020. Anonymous authors.
"""

try:
    import comet_ml
    from comet_config import COMET_API_KEY, COMET_WORKSPACE, COMET_PROJECT_NAME
    _use_comet = True
except ImportError:
    comet_ml = None
    COMET_API_KEY, COMET_WORKSPACE, COMET_PROJECT_NAME = None, None, None
    _use_comet = False

import pickle
import os
import glob
import math
import numpy as np
import re
import pytrec_eval
import json
import pandas as pd

from transformers import AlbertConfig

from models import BinaryClf
from data import RFAlbertDataset, CSVDatasetFactory
from plot_trec_eval import plot_pr_curve

from absl import app
from absl import flags

flags.DEFINE_boolean('predict', False, 'Run the predictions')
flags.DEFINE_integer('batch_size', 256, ' Batch Size for Prediction')
flags.DEFINE_string('albert_name', None, 'Name of Pretrained model')
flags.DEFINE_string('valset', None, 'CSV File for Prediction Dataset')
flags.DEFINE_string('output_dir', None, 'Folder to output prediction pickle')
flags.DEFINE_string('model_dir', None, 'Path to the model checkpoints')
flags.DEFINE_string('tokenizer', None, 'Path to the Tokenizer')
flags.DEFINE_integer('epoch', None, 'Evaluate the Ranker for a checkpoint at a specific EPOCH')
flags.DEFINE_boolean('all_epochs', False, 'Evaluate the Ranker at each available checkpoint')
flags.DEFINE_bool('evaluate', False, 'Proceed with evaluation')
flags.DEFINE_string('comet_exp', None, 'API Key of an existing experiment')
FLAGS = flags.FLAGS


def main(_):
    global _use_comet
    _use_comet = _use_comet and FLAGS.comet_exp is not None

    if FLAGS.predict:
        # Only using the trained model at various epochs to predict the probabilities of being a redflag
        # Which epochs should be used for model weights
        assert (FLAGS.epoch is not None) != FLAGS.all_epochs
        epochs = []
        if FLAGS.epoch is not None:
            epochs = [FLAGS.epoch]
        elif FLAGS.all_epochs:
            model_ckpts = BinaryClf.list_checkpoints(FLAGS.model_dir)
            epochs = [BinaryClf.which_epoch(x) for x in model_ckpts]

        # Load the data
        custom_config = AlbertConfig.from_pretrained(FLAGS.albert_name)
        if FLAGS.tokenizer is not None:
            val_set = RFAlbertDataset(FLAGS.valset, albert_name=FLAGS.albert_name, tokenizer_dir=FLAGS.tokenizer)
            custom_config.vocab_size = val_set.tokenizer.vocab_size
        else:
            val_set = RFAlbertDataset(FLAGS.valset, albert_name=FLAGS.albert_name)
        val_dataset = val_set.dataset.batch(FLAGS.batch_size)
        val_steps = math.ceil(len(val_set) / FLAGS.batch_size)

        # Instantiate the model, untrained

        # For each of the requested epoch, initialize the model with the weights in a specific checkpoint
        # Then predict the probabilities of belonging to the positive class for each sample
        # Record these probabilities in a pickle file
        for epoch in epochs:
            model = BinaryClf.load_custom_model(
                checkpoint_folder=FLAGS.model_dir,
                epoch=epoch,
                config=custom_config
            )
            preds = model.predict(
                x=val_dataset,

                steps=val_steps,
                verbose=1
            )
            # preds is array with shape (nb_samples, 1)
            pickle.dump(preds, open(os.path.join(FLAGS.output_dir, f'preds.{epoch:03d}.pkl'), 'wb'))

    if FLAGS.evaluate:
        # Load the data
        val_df = CSVDatasetFactory.get_dataset(FLAGS.valset, 'redflag').df.copy()
        qrel = {'redflag': {str(key): value for key, value in val_df.to_dict()['binary'].items()}}

        experiment = None
        if _use_comet:
            experiment = comet_ml.ExistingExperiment(
                api_key=COMET_API_KEY,
                previous_experiment=FLAGS.comet_exp
            )

            # In case they do exist, remove existing 'RANK' figures
            api_exp = comet_ml.APIExperiment(
                api_key=COMET_API_KEY,
                previous_experiment=FLAGS.comet_exp
            )

            remove_ids = [x['assetId'] for x in api_exp.get_asset_list() if x['fileName'].startswith('RANKER')]
            for aid in remove_ids:
                api_exp.delete_asset(aid)

        preds_pkls = glob.glob(os.path.join(FLAGS.output_dir, 'preds.*.pkl'))

        p_recall_1 = []   # Keep track of Precision for Recall = 1.0
        for s, pkl in enumerate(preds_pkls):
            epoch = int(re.match(r'^preds\.(?P<epoch>\d{3})\.pkl$', os.path.basename(pkl)).group('epoch'))
            preds_nd: np.ndarray = pickle.load(open(pkl, 'rb'))
            preds = {'redflag': {str(key): float(value) for key, value in zip(val_df.index, preds_nd.ravel())}}

            evaluator = pytrec_eval.RelevanceEvaluator(qrel, pytrec_eval.supported_measures)
            results = evaluator.evaluate(preds)
            results['redflag']['runid'] = f'EPOCH {epoch:03d}'
            json.dump(results, open(os.path.join(FLAGS.output_dir, f'eval.{epoch:03d}.json'), 'w'))

            p_recall_1.append({'epoch': epoch, 'P@R=1': results['redflag']['iprec_at_recall_1.00']})

            if _use_comet and experiment is not None:
                experiment.log_figure(
                    figure_name=f'RANKER PR Epoch {epoch:d}',
                    figure=plot_pr_curve.process(list_results=[results['redflag']]),
                    overwrite=True
                )
                experiment.log_metric('ranker_auc_pr', results['redflag']['map'], step=s + 1)

        if _use_comet and experiment is not None:
            ax = pd.DataFrame(p_recall_1).plot.bar(x='epoch', y='P@R=1', ylim=(0.0, 1.01))
            experiment.log_figure(
                figure_name=f'RANKER P@R=1',
                figure=ax.figure,
                overwrite=True
            )


if __name__ == '__main__':
    app.run(main)
