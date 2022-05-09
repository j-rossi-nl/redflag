"""
Models are handled as instances of custom classes
Makes it easy to load / save / etc...

2020. Anonymous authors.
"""

import tensorflow as tf
import os
import glob
import re

from abc import ABC
from transformers import TFAlbertPreTrainedModel, TFAlbertForTokenClassification, TFAlbertMainLayer, AlbertConfig

from typing import List, Any


class MyAlbertModel:
    """
    All our models will share a common ancestry.
    Mainly this is dealing with saving checkpoints / restoring from checkpoints
    """
    save_model_name = 'model'
    save_model_suffix = 'h5'

    @classmethod
    def create_model_checkpoint_callback(cls, output_dir: str) -> tf.keras.callbacks.Callback:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=cls.checkpoint_template(output_dir),
            save_weights_only=True,
            save_freq='epoch'
        )
        return checkpoint

    @classmethod
    def create_model(cls, albert_type: str, *args, **kwargs) -> Any:
        raise NotImplementedError

    @classmethod
    def load_model(cls, albert_type: str, checkpoint_folder: str, epoch: int = 0, *args, **kwargs) -> Any:
        raise NotImplementedError

    @classmethod
    def load_model_pt(cls, checkpoint_folder: str, *args, **kwargs) -> Any:
        raise NotImplementedError

    @classmethod
    def checkpoint_filename(cls, checkpoint_folder: str, epoch: int) -> str:
        return os.path.join(checkpoint_folder, f'{cls.save_model_name}.{epoch:03d}.{cls.save_model_suffix}')

    @classmethod
    def checkpoint_template(cls, checkpoint_folder: str) -> str:
        return os.path.join(
            checkpoint_folder,
            f'{cls.save_model_name}.{{epoch:03d}}.{cls.save_model_suffix}'
        )

    @classmethod
    def list_checkpoints(cls, checkpoint_folder: str) -> List[str]:
        return glob.glob(os.path.join(checkpoint_folder, f'{cls.save_model_name}.*.{cls.save_model_suffix}'))

    @classmethod
    def which_epoch(cls, checkpoint: str) -> int:
        match = re.match(
            r'^' + cls.save_model_name + r'\.(?P<epoch>\d{3})\.' + cls.save_model_suffix + r'$',
            os.path.basename(checkpoint)
        )
        if match is None:
            raise ValueError('Does not look like a model checkpoint')
        return int(match.group('epoch'))


class BinaryClf(TFAlbertPreTrainedModel, ABC, MyAlbertModel):
    """
    An Albert model that we can finetune for the binary classification task 'Redflag or Normal text'
    """
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = 1

        self.albert = TFAlbertMainLayer(config, name="albert")
        self.dropout = tf.keras.layers.Dropout(config.classifier_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            self.num_labels, name="classifier", activation='sigmoid'
        )

    def call(self, inputs, **kwargs):
        outputs = self.albert(inputs, **kwargs)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output, training=kwargs.get("training", False))
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)

    @classmethod
    def create_model(cls, albert_type: str, *args, **kwargs) -> 'BinaryClf':
        return cls.from_pretrained(albert_type)

    @classmethod
    def load_model(cls, albert_type: str, checkpoint_folder: str, epoch: int = 0, *args, **kwargs) -> 'BinaryClf':
        model = cls.create_model(albert_type)
        model.load_weights(cls.checkpoint_filename(checkpoint_folder, epoch))
        return model

    @classmethod
    def load_model_pt(cls, checkpoint_folder: str, *args, **kwargs) -> 'BinaryClf':
        model = cls.from_pretrained(checkpoint_folder, from_pt=True)
        return model

    @classmethod
    def load_custom_model(cls, checkpoint_folder: str, epoch: int,
                          config: AlbertConfig) -> 'BinaryClf':
        model = cls.from_pretrained(cls.checkpoint_filename(checkpoint_folder, epoch), config=config)
        return model


class AlbertNER(TFAlbertForTokenClassification, ABC, MyAlbertModel):
    """
    Albert model for NER.
    """

    @classmethod
    def create_model(cls, albert_type: str, *args, **kwargs) -> 'AlbertNER':
        """

        :param albert_type:
        :param args:
        :param kwargs: num_entities will indicate the number of entity classes
        :return:
        """
        return TFAlbertForTokenClassification.from_pretrained(albert_type, num_labels=kwargs['num_entities'])

    @classmethod
    def load_model(cls, albert_type: str, checkpoint_folder: str, epoch: int = 0, *args, **kwargs) -> 'AlbertNER':
        model = cls.create_model(albert_type, **kwargs)
        model.load_weights(cls.checkpoint_filename(checkpoint_folder, epoch))
        return model

    @classmethod
    def load_model_pt(cls, checkpoint_folder: str, *args, **kwargs) -> 'AlbertNER':
        model = cls.from_pretrained(checkpoint_folder, from_pt=True, num_labels=kwargs['num_entities'])
        return model

    @classmethod
    def load_custom_model(cls, checkpoint_folder: str, epoch: int,
                          config: AlbertConfig, **kwargs) -> 'AlbertNER':
        model = cls.from_pretrained(cls.checkpoint_filename(checkpoint_folder, epoch),
                                    config=config, num_labels=kwargs['num_entities'])
        return model
