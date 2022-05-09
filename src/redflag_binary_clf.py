"""
This script will train a ALBERT-based classifier.
The classifier is trained on discrimating whether there is a redflag in a text (relevance assessment).
This script uses Huggingface for ALBERT models, and comet_ml for experiment logging.
Setup for comet_ml: add a `comet_config.py` file which defines constants `COMET_API_KEY`, `COMET_WORKSPACE`,
`COMET_PROJECT_NAME`.

Script usage:
$ python redflag_binary_clf.py <args>

Args:
    --train: Start a training
    --task: one of 'redflag' or 'clause'
    --trainset: CSV file with training data
    --valset: CSV file with validation data
    --batch_size: (default 64)
    --val_batch_size: (default 256)
    --num_epochs: (default 3)
    --albert_name: the huggingface name of the Albert model to use. (default is 'albert-base-v1')
    --output_dir: Path to an existing folder where the model checkpoints will be stored. 1 checkpoint per epoch.
    --restart_from_pt: Training restarts from an existing PyTorch checkpoint. If this is not given, output_dir.
    --restart_from_tf: Training restarts from an existing Tensorflowcheckpoint. If this is not given, output_dir.
    --restart_from: Training restarts from an existing checkpoint. Just indicate the EPOCH number. (TF Only)
    --comet if comet.ml should be used for experiment tracking
    --comet_tags: Comma-separated tags for the experiment on comet_ml.

Fine-Tune starting from an Off-The-Shelf Pre-Trained model
$ python redflag_binary_clf.py --train --albert_name albert-base-v2 --trainset ../train.csv --valset ../val.csv
    --output_dir ../folder --num_epochs 10 --lr 1e-6

Fine-Tuning starting from a specific Fine-Tuning checkpoint
$ python redflag_binary_clf.py --train --albert_name albert-base-v2 --trainset ../train.csv --valset ../val.csv
    --output_dir ../folder --num_epochs 10 --restart_from_tf ../previousrun --restart_from 7 --lr 1e-6

Fine-Tuning starting from a Pytorch checkpoint (see albert_pretraining.py)
$ python redflag_binary_clf.py --train --albert_name albert-base-v2 --trainset ../train.csv --valset ../val.csv
    --output_dir ../folder --num_epochs 10 --lr 1e-6 --restart_from_pt ../pretraining/checkpoint-25000

Use a specific tokenizer:
$ --tokenizer <folder>
The folder should contain a file 'spiece.model'.
The same tokenizer should be used every time this model is being worked on.

2020. Anonymous authors.
"""

try:
    import comet_ml
    from comet_config import COMET_API_KEY, COMET_WORKSPACE, COMET_PROJECT_NAME
    from comet_helper import LogPRCurve
    _use_comet = True
except ImportError:
    comet_ml = None
    COMET_API_KEY, COMET_PROJECT_NAME, COMET_WORKSPACE = None, None, None
    LogPRCurve = None
    _use_comet = False

import pickle
import logging

from absl import app
from absl import flags

from models import BinaryClf
from data import csv_to_albert, CSVDatasetFactory
from train_utils import train_binary_clf_model

flags.DEFINE_bool('train', False, 'Training')
flags.DEFINE_enum('task', None, CSVDatasetFactory.TASKS, 'Which task from redflag / clause')
flags.DEFINE_integer('batch_size', 64, 'Batch Size for Training')
flags.DEFINE_integer('val_batch_size', 256, ' Batch Size for Validation')
flags.DEFINE_integer('num_epochs', 3, 'Number of Epochs')
flags.DEFINE_string('albert_name', 'albert-base-v1', 'Name of Pretrained model')
flags.DEFINE_string('trainset', None, 'CSV File for Training Dataset')
flags.DEFINE_string('valset', None, 'CSV File for Validation Dataset')
flags.DEFINE_string('output_dir', 'output', 'Folder to save models')
flags.DEFINE_string('restart_from_pt', None, 'PyTorch Checkpoint Folder')
flags.DEFINE_string('restart_from_tf', None, 'Folder with the TF Checkpoint to start from')
flags.DEFINE_integer('restart_from', 0, 'Restart from a specific EPOCH from a TF folder')
flags.DEFINE_string('tokenizer', None, 'Folder that contains a tokenizer SentencePiece model')
flags.DEFINE_float('lr', default=1e-5, help='Initial Learning Rate')
flags.DEFINE_bool('comet', False, 'Use COMET.ml for logging')
flags.DEFINE_list('comet_tags', 'PROD', 'Tags for Comet Experience')
flags.DEFINE_bool('predict', False, 'Predict')
flags.DEFINE_string('preds_file', 'preds.pkl', 'File where predictions are saved')
FLAGS = flags.FLAGS


def main(_):
    global _use_comet
    _use_comet = _use_comet and FLAGS.comet

    if FLAGS.train:
        # TRAINING

        # Creating instances of training and validation set
        train_set = csv_to_albert(FLAGS.trainset, task=FLAGS.task,
                                  albert_name=FLAGS.albert_name, tokenizer_dir=FLAGS.tokenizer)
        val_set = csv_to_albert(FLAGS.valset, task=FLAGS.task,
                                albert_name=FLAGS.albert_name, tokenizer_dir=FLAGS.tokenizer)

        if FLAGS.restart_from > 0:
            # Restart training from an existing checkpoint
            load_model_from_dir = FLAGS.restart_from_tf if FLAGS.restart_from_tf is not None else FLAGS.output_dir
            logging.info('Loading weights from EPOCH {:d}'.format(FLAGS.restart_from))
            model = BinaryClf.load_model(
                albert_type=FLAGS.albert_name,
                checkpoint_folder=load_model_from_dir,
                epoch=FLAGS.restart_from
            )
        elif FLAGS.restart_from_pt is not None:
            # Restart from a PyTorch checkpoint folder
            model = BinaryClf.load_model_pt(FLAGS.restart_from_pt)
        else:
            # Just pick-up a Pre-Trained off-the-shelf model
            model = BinaryClf.create_model(albert_type=FLAGS.albert_name)

        callbacks = []
        if _use_comet:
            experiment = comet_ml.Experiment(
                api_key=COMET_API_KEY,
                workspace=COMET_WORKSPACE,
                project_name=COMET_PROJECT_NAME,
            )
            experiment.add_tags(FLAGS.comet_tags)
            callbacks.append(LogPRCurve(validation_data=val_set, experiment=experiment))

        train_binary_clf_model(
            model=model,
            train_set=train_set,
            train_batch_size=FLAGS.batch_size,
            val_set=val_set,
            val_batch_size=FLAGS.val_batch_size,
            output_dir=FLAGS.output_dir,
            num_epochs=FLAGS.num_epochs,
            lr=FLAGS.lr,
            add_callbacks=callbacks,
            init_epoch=FLAGS.restart_from
        )

    elif FLAGS.predict:
        # PREDICT

        val_set = csv_to_albert(FLAGS.valset, task=FLAGS.task,
                                albert_name=FLAGS.albert_name, tokenizer_dir=FLAGS.tokenizer)
        val_dataset = val_set.dataset.batch(FLAGS.val_batch_size)

        model = BinaryClf.load_model(
            albert_type=FLAGS.albert_name,
            checkpoint_folder=FLAGS.output_dir,
            epoch=FLAGS.restart_from
        )

        preds = model.predict(
            x=val_dataset,
            steps=len(val_set) // FLAGS.val_batch_size,
            verbose=1
        )

        # preds is array with shape (nb_samples, 1)
        pickle.dump(preds, open(FLAGS.preds_file, 'wb'))


if __name__ == '__main__':
    app.run(main)
