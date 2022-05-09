"""

2020. Anonymous authors.
"""

import pandas as pd
import spacy
import warnings
import random

from typing import List, Tuple, Dict
from pathlib import Path
from spacy.util import minibatch, compounding
from spacy.language import Language
from spacy.pipeline import EntityRecognizer


from absl import app
from absl import flags


flags.DEFINE_bool('train', False, 'Training')
flags.DEFINE_integer('batch_size', 64, 'Batch Size for Training')
flags.DEFINE_integer('val_batch_size', 256, ' Batch Size for Validation')
flags.DEFINE_integer('num_epochs', 3, 'Number of Epochs')
flags.DEFINE_string('trainset', None, 'CSV File for Training Dataset')
flags.DEFINE_string('valset', None, 'CSV File for Validation Dataset')
flags.DEFINE_string('labels', None, 'File with List of labels')
flags.DEFINE_string('output_dir', 'output', 'Folder to save models')
flags.DEFINE_bool('comet', False, 'Use COMET.ml for logging')
flags.DEFINE_list('comet_tags', 'PROD', 'Tags for Comet Experience')
flags.DEFINE_bool('predict', False, 'Predict')
flags.DEFINE_string('preds_file', 'preds.pkl', 'File where predictions are saved')
FLAGS = flags.FLAGS


def df_to_spacy(df: pd.DataFrame) -> List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]]:
    """
    Converts the pandas dataframe created from `data_prepare.py entities` to a Spacy training data
    :param df: a dataframe
    :return: Spacy training data for NER (see https://spacy.io/usage/training)
    """

    spacy_data = []
    grp = df.groupby(['uuid', 'part_id'])
    for gk, gd in grp:
        starts = gd['entity_start']
        ends = gd['entity_end']
        tags = gd['class_id']

        text = gd['full_text'].values[0]

        spacy_sample = (text, {'entities': [(s, e, t) for s, e, t in zip(starts, ends, tags)]})
        spacy_data.append(spacy_sample)

    return spacy_data


def main(_):
    if FLAGS.train:
        trainset = pd.read_csv(FLAGS.trainset)
        train_data = df_to_spacy(trainset)
        labels = set([line[2:] for line in open(FLAGS.labels, 'r')])   # Move the BILUO, 'B-TAG' -> 'TAG'

        nlp: Language = spacy.blank('en')
        ner: EntityRecognizer = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
        optimizer = nlp.begin_training()

        for label in labels:
            ner.add_label(label)

        # get names of other pipes to disable them during training
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

        # only train NER
        with nlp.disable_pipes(*other_pipes) and warnings.catch_warnings():
            # show warnings for misaligned entity spans once
            warnings.filterwarnings("once", category=UserWarning, module='spacy')

            for itn in range(FLAGS.num_epochs):
                random.shuffle(train_data)

                # batch up the examples using spaCy's minibatch
                batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(
                        texts,  # batch of texts
                        annotations,  # batch of annotations
                        drop=0.5,  # dropout - make it harder to memorise data
                        sgd=optimizer
                    )

        # save model to output directory
        if FLAGS.output_dir is not None:
            output_dir = Path(FLAGS.output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.to_disk(output_dir)
            print("Saved model to", output_dir)

    if FLAGS.predict:
        valset = pd.read_csv(FLAGS.valset)
        val_data = df_to_spacy(valset)

        # test the saved model
        print("Loading from", FLAGS.output_dir)
        nlp2 = spacy.load(FLAGS.output_dir)
        for text, _ in val_data:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == '__main__':
    app.run(main)
