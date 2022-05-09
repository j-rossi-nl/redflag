"""
Model Training.
Boilerplate code.
"""

import math
import tensorflow as tf

from models import BinaryClf
from data import RFAlbertDataset

from tensorflow.keras.callbacks import Callback

from typing import List


def train_binary_clf_model(
        model: BinaryClf,
        train_set: RFAlbertDataset,
        train_batch_size: int,
        val_set: RFAlbertDataset,
        val_batch_size: int,
        output_dir: str,
        num_epochs: int,
        lr: float,
        add_callbacks: List[Callback],
        init_epoch: int = 0
):

    train_dataset = train_set.dataset.\
        shuffle(len(train_set)).\
        batch(train_batch_size).\
        repeat(num_epochs).\
        prefetch(4)
    val_dataset = val_set.dataset.batch(val_batch_size)

    training_steps = math.ceil(len(train_set) / train_batch_size)
    validation_steps = math.ceil(len(val_set) / val_batch_size)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr, epsilon=1e-8),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
            tf.keras.metrics.AUC(name='auc_pr', curve='PR'),
            tf.keras.metrics.Precision(),  # PRECISION for the POSITIVE class
            tf.keras.metrics.Recall(),     # RECALL    for the POSITIVE class
        ]
    )

    callbacks = [model.create_model_checkpoint_callback(output_dir)]
    callbacks.extend(add_callbacks)

    model.fit(
        train_dataset,
        epochs=num_epochs + init_epoch,
        validation_data=val_dataset,
        steps_per_epoch=training_steps,
        validation_steps=validation_steps,
        callbacks=callbacks,
        initial_epoch=init_epoch
    )
