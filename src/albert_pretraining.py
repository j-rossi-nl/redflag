"""
Pretraining is an unsupervised task.
In this case, there is not a lot of material, we will perform additional pretraining to an existing
pre-trained model.
Unlike the other modules, we will make use of the PyTorch framework.
A tokenizer can be trained, it will be a Sentence-Piece model, encapsulated in an AlbertTokenizer object.

Largely inspired by https://huggingface.co/blog/how-to-train

Usage:

In all cases, experiences can be tracked by COMET (see https://comet.ml).
Use the following flags in the command line:
--comet : activate the comet tracking
--comet_tags TAG1,TAG2,TAG3 : give tags to the experiment

Pre-Training from Scratch:
With an existing tokenizer (model file located in output_dir)
$ python albert_pretraining.py --albert_name albert-base-v2 --trainset ../mycorpus.txt --batch_size 32 --epochs 10
    --output_dir ../models/pretrain --save_every 1000

Starting by training a tokenizer.
$ python albert_pretraining.py --albert_name albert-base-v2 --trainset ../mycorpus.txt --vocab_size 50000
    --batch_size 32 --epochs 10 --output_dir ../models/pretrain --save_every 1000

Pre-Training starting from a registered huggingface model (domain adaptation).
In this case, it uses the tokenizer associated to the original model.
$ python albert_pretraining.py --albert_name albert-base-v2 --additional --trainset ../mycorpus.txt
    --batch_size 32 --epochs 10 --output_dir ../models/pretrain --save_every 1000

2020. Anonymous authors.
"""

try:
    import comet_ml
    from comet_config import COMET_API_KEY, COMET_PROJECT_NAME, COMET_WORKSPACE
    _use_comet = True
except ImportError:
    comet_ml = None
    COMET_API_KEY, COMET_PROJECT_NAME, COMET_WORKSPACE = None, None, None
    _use_comet = False

import logging
import sentencepiece as spm
import os
import glob

from transformers import AlbertTokenizer, AlbertForMaskedLM, AlbertConfig
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers.tokenization_albert import VOCAB_FILES_NAMES

# Modified version of transformers Trainer
from trainer import Trainer

from data import LMDataset

from absl import app
from absl import flags

flags.DEFINE_string('albert_name', None, 'Name of Pretrained model')
flags.DEFINE_string('trainset', None, 'TXT file')
flags.DEFINE_integer('batch_size', 64, 'Batch Size for Training')
flags.DEFINE_integer('epochs', 3, 'Number of Epochs')
flags.DEFINE_string('output_dir', 'output', 'Folder to save checkpoints'
                                            'When training from scratch: should contain the tokenizer model.')
flags.DEFINE_integer('vocab_size', None, 'Size of vocabulary for tokenizer training')
flags.DEFINE_integer('save_every', 10000, 'Create a checkpoint every N steps')
flags.DEFINE_bool('fp16', False, 'Use Half-Precision')
flags.DEFINE_bool('additional', False, 'Perform some additional pre-training from an OFF-THE-SHELF model')
flags.DEFINE_bool('comet', False, 'Use COMET for logging')
flags.DEFINE_list('comet_tags', None, 'Tags for Comet Experience')
FLAGS = flags.FLAGS


def main(_):
    """
    Usual plan:
    Create the dataset, Create the model, Train...
    :return:
    """
    global _use_comet
    _use_comet = _use_comet and FLAGS.comet

    if _use_comet:
        experiment = comet_ml.Experiment(
            api_key=COMET_API_KEY,
            workspace=COMET_WORKSPACE,
            project_name=COMET_PROJECT_NAME,
        )
        experiment.add_tags(FLAGS.comet_tags)

    logger = logging.getLogger('Albert Pretraining')
    logger.info('Let\'s go !!')

    if not os.path.exists(os.path.join(FLAGS.output_dir, VOCAB_FILES_NAMES['vocab_file'])) and not FLAGS.additional:
        # If there is no tokenizer model saved in the output folder, and it is a pre-training from scratch
        # Tokenizer Training
        logger.info('Tokenizer Training - START')
        tokenizer_model_name, _ = VOCAB_FILES_NAMES['vocab_file'].split('.')

        spm_train_args = {
            'input': FLAGS.trainset,
            'model_prefix': tokenizer_model_name,
            'vocab_size': FLAGS.vocab_size
        }
        spm.SentencePieceTrainer.train(
            ' '.join(f'--{k}={v}' for k, v in spm_train_args.items())
        )
        for f in glob.glob(f'{tokenizer_model_name}.*'):
            os.rename(f, os.path.join(FLAGS.output_dir, os.path.basename(f)))
        logger.info('Tokenizer Training - END')

    # AlbertTokenizer
    if FLAGS.additional:
        # Additional pre-training from an off-the-shelf model
        # We keep the tokenizer attached to this model
        tokenizer = AlbertTokenizer.from_pretrained(FLAGS.albert_name)
    else:
        tokenizer = AlbertTokenizer.from_pretrained(FLAGS.output_dir)

    # Dataset
    train_set = LMDataset(
        tokenizer=tokenizer,
        file_path=FLAGS.trainset,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Model
    if FLAGS.additional:
        # Additional pre-training on an off-the-shelf model
        model = AlbertForMaskedLM.from_pretrained(FLAGS.albert_name)
    else:
        # Using the same config as standard Albert model
        # But adapting the embedding layer to the vocabulary size of the tokenizer
        config = AlbertConfig.from_pretrained(FLAGS.albert_name)
        model = AlbertForMaskedLM(config=config)
        model.resize_token_embeddings(new_num_tokens=tokenizer.vocab_size)

    # Training
    train_args = TrainingArguments(
        output_dir=FLAGS.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=FLAGS.epochs,
        per_gpu_train_batch_size=FLAGS.batch_size,
        save_steps=FLAGS.save_every,
        logging_steps=0,
        fp16=FLAGS.fp16
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_set,
        prediction_loss_only=True,
    )

    trainer.train()


if __name__ == '__main__':
    app.run(main)
