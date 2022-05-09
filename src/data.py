"""
Helper classes to handle the training data for different tasks.
For the classification tasks, the dataset are all quite small in size, so we use on-the-fly encoding.
We use a cache for the Language Modeling dataset.

2020. Anonymous authors.
"""

import tensorflow as tf
import pandas as pd
import os
import logging
import pickle
import hashlib
import torch
import numpy as np
import tempfile

from transformers import AlbertTokenizer, PreTrainedTokenizer
from torch.utils.data import Dataset

from collections import UserDict
from abc import ABC, abstractmethod

from utils_ner import TFNerDataset, Split

logger = logging.getLogger(__name__)


class MetaDataset(ABC):
    """
    Abstract class for handling of CSV-based datasets (redflag, clause, ...)
    Subclasses have to implement 2 properties: features and targets
    """

    @property
    @abstractmethod
    def features(self):
        pass

    @property
    @abstractmethod
    def targets(self):
        pass

    def __len__(self):
        return len(self.features)


class RedflagCSVDataset(MetaDataset):
    """
    Implementation of MetaDataset for the Redflag task
    """

    def __init__(self, csvfile):
        self.df = pd.read_csv(csvfile).dropna(subset=['raw_text'])
        self.df['binary'] = self.df['type'].map(lambda x: 0 if x == 'none' else 1)

    @property
    def features(self):
        return self.df['raw_text'].copy()

    @property
    def targets(self):
        return self.df['binary'].copy()


class ClauseNumberCSVDataset(MetaDataset):
    """
    Implementation of MetaDataset for the Clause Number Detection task
    """

    def __init__(self, csvfile):
        self.df = pd.read_csv(csvfile).dropna(subset=['text'])
        self.df['is_number'] = self.df['clause_type'].map(
            lambda x: 1 if x in ('clause_number', 'sub_clause_number') else 0
        )

    @property
    def features(self):
        return self.df['text'].copy()

    @property
    def targets(self):
        return self.df['is_number'].copy()


class ClauseTitleCSVDataset(MetaDataset):
    """
    Implementation of MetaDataset for the Clause Number Detection task
    """

    def __init__(self, csvfile):
        self.df = pd.read_csv(csvfile).dropna(subset=['text'])
        self.df['is_title'] = self.df['clause_type'].map(
            lambda x: 1 if x in ('clause_title', 'sub_clause_title') else 0
        )

    @property
    def features(self):
        return self.df['text'].copy()

    @property
    def targets(self):
        return self.df['is_title'].copy()


class CSVEntitiesDataset(MetaDataset):
    """
    Implementation of MetaDataset for the Entities Detection task.
    This dataset is built on a CSV file as generated by `data_prepare.py entities`.
    """

    # We have settled to focus on only some entities
    # REMOVED from the original annotation dataset:
    #    structure = ['sub_clause number', 'clause_number', 'clause_title', 'sub_clause_title',
    #                 'definition', 'definition_number', 'annex']
    #    inconsistent = ['indexation_rent', 'annex', 'type_lease']
    #    leave_apart = ['redflag']
    FOCUS_ON = ('lessor', 'lessee', 'start_date', 'end_date', 'extension_period',
                'term_of_payment', 'vat', 'signing_date', 'designated_use',
                'sub_clause_number', 'leased_space', 'expiration_date_of_lease',
                'notice_period')

    def __init__(self, csvfile):
        self.raw_df = pd.read_csv(csvfile).dropna(subset=['full_text'])
        self.raw_df = self.raw_df[self.raw_df['class_id'].isin(CSVEntitiesDataset.FOCUS_ON)]

    @property
    def features(self):
        return self.raw_df[['full_text', 'entity_start', 'entity_end']]

    @property
    def targets(self):
        return self.raw_df['class_id']


class CSVDatasetFactory:
    """
    Factory interface
    """
    _TASKS_CLASS = {
        'redflag': RedflagCSVDataset,
        'clause_title': ClauseTitleCSVDataset,
        'clause_number': ClauseNumberCSVDataset
    }

    TASKS = _TASKS_CLASS.keys()

    @classmethod
    def get_dataset(cls, csvfile, task):
        return CSVDatasetFactory._TASKS_CLASS[task](csvfile)


class MetaTokenizedDataset:

    def __init__(self, data: MetaDataset, albert_name, tokenizer_dir=None):
        self.data = data
        # (Tokenizer_dir is given AND Albert_name) OR (Albert_name is given)
        assert albert_name is not None

        if tokenizer_dir is not None and albert_name is not None:
            self.tokenizer = AlbertTokenizer.from_pretrained(tokenizer_dir)
            self.tokenizer.model_max_length = AlbertTokenizer.max_model_input_sizes[albert_name]  # otherwise faulty
        else:
            self.tokenizer = AlbertTokenizer.from_pretrained(albert_name)


class RFAlbertDataset(MetaTokenizedDataset):
    """
    A helper class to handle the dataset for the Fine-Tuning of the Binary Classification 'Redflag or Not'.
    Generates Tensorflow Tensors as inputs.
    """

    @property
    def dataset(self) -> tf.data.Dataset:
        data_encoded = self.tokenizer.batch_encode_plus(
            self.data.features,
            max_length=self.tokenizer.model_max_length,
            pad_to_max_length=True,
        )

        targets = self.data.targets

        def gen_inputs_targets():
            for i, label in enumerate(targets):
                yield (
                    {
                        'input_ids': data_encoded['input_ids'][i],
                        'attention_mask': data_encoded['attention_mask'][i],
                        'token_type_ids': data_encoded['token_type_ids'][i]
                    },
                    label
                )

        return tf.data.Dataset.from_generator(
            gen_inputs_targets,
            ({'input_ids': tf.int32, 'attention_mask': tf.int32, 'token_type_ids': tf.int32}, tf.int32),
            (
                {
                    'input_ids': tf.TensorShape([None]),
                    'attention_mask': tf.TensorShape([None]),
                    'token_type_ids': tf.TensorShape([None]),
                },
                tf.TensorShape([])
            )
        )

    def __len__(self):
        return len(self.data.features)


class LMDataset(Dataset):
    """
    A helper class to handle the dataset used for Language Modeling task (aka Pre-Training).
    Generates PyTorch tensors.
    Uses a built-in cache feature to avoid processing the same input file twice.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str):
        assert os.path.isfile(file_path)

        # Cache / Get out of cache
        # Assumes a folder 'cache' in the same folder as the input file
        cache_file_path = os.path.join(os.path.dirname(file_path), 'cache', md5_file_contents(file_path))
        if os.path.isfile(cache_file_path):
            logger.info("Using Cached data")
            self.examples: np.ndarray = pickle.load(open(cache_file_path, 'rb'))
        else:
            logger.info("Creating features from dataset file at %s", file_path)

            with open(file_path, encoding="utf-8") as f:
                lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

            batch_encoding: UserDict = tokenizer.batch_encode_plus(     # Leaving BatchEncoding created type warnings
                lines,
                add_special_tokens=True,
                max_length=tokenizer.model_max_length,
            )
            self.examples: np.ndarray = np.array(batch_encoding['input_ids'])
            logger.info("Caching features")
            os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
            pickle.dump(self.examples, open(cache_file_path, 'wb'), protocol=4)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


def md5_file_contents(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def csv_to_albert(
        csvfile: str,
        task: str,
        albert_name: str,
        tokenizer_dir: str = None
) -> RFAlbertDataset:
    """
    Useful function

    :param csvfile:
    :param task:
    :param albert_name:
    :param tokenizer_dir:
    :return:
    """
    data = CSVDatasetFactory.get_dataset(csvfile=csvfile, task=task)
    dataset = RFAlbertDataset(data, albert_name=albert_name, tokenizer_dir=tokenizer_dir)

    return dataset
