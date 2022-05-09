"""
Callback that log additional information to Comet

2020. Anonymous authors.
"""

import comet_ml
import tensorflow as tf
import math

from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, average_precision_score
from absl import flags

from data import RFAlbertDataset

FLAGS = flags.FLAGS


class LogPRCurve(tf.keras.callbacks.Callback):
    """
    A Comet logger to the PR curve
    """
    def __init__(self, validation_data: RFAlbertDataset, experiment: comet_ml.Experiment):
        super().__init__()
        self.inputs = validation_data.dataset
        self.targets = validation_data.data.targets
        self.experiment = experiment

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(
            x=self.inputs.batch(FLAGS.val_batch_size),
            steps=math.ceil(len(self.targets) / FLAGS.val_batch_size)
        )

        p, r, t = precision_recall_curve(y_true=self.targets, probas_pred=y_pred)
        map_ = average_precision_score(y_true=self.targets, y_score=y_pred)
        pr_curve = PrecisionRecallDisplay(precision=p, recall=r, average_precision=map_, estimator_name='albert')
        pr_curve = pr_curve.plot()
        pr_curve.ax_.set_xlim(-0.01, 1.01)
        pr_curve.ax_.set_ylim(-0.01, 1.01)
        self.experiment.log_figure(
            figure_name=f'PR Epoch {epoch+1:d}',
            figure=pr_curve.plot().figure_
        )
