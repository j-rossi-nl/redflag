"""
This module deals with data pre-processing.
The original data is made of separate JSON files, during preprocessing these files are
consolidated into a single CSV file to make future ML usage easier

Example:
    $ python data_prepare.py entities --original ../data/original --dest ../data/consolidated/entities.csv

2020. Anonymous authors.
"""

import pandas as pd
import sys
import re
import tqdm
import os

from nltk.tokenize import sent_tokenize, word_tokenize
from argparse import ArgumentParser, Namespace
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import train_test_split

from utils_ner import Split
from utils import Contract, OriginalSample, OriginalDataset, ENTITY_NAME_2_ID, ENTITY_ID_2_NAME, prev_part


# The arguments of the command are presented as a global module variable, so all functions require no arguments
_args: Namespace


def consolidate_clauses() -> None:
    """
    Uses the data in the original format (files are located in the folder _args.original)
    Create a consolidated CSV file for Clause Detection in the _args.dest file
    The CSV file has the following columns: uuid, text, clause_begin (bool), clause_type
    :return:
    """
    raw_dataset = OriginalDataset(original_folder=_args.original)
    data = []

    # Iterate through annotations
    for uuid, d in tqdm.tqdm(raw_dataset):
        annotations: Dict[str, Any] = d.contract_annotations
        contract_text: Contract = d.contract_text

        # we focus on clause / subclause titles
        # A clause starts at the part AFTER the title and ends with the part BEFORE the next clause
        # The last clause / subclause ends with the part BEFORE the part flagged as 'annex'
        clauses = {x['part']: x for x in annotations['entities']
                   if x['classId'] in [ENTITY_NAME_2_ID['clause_title'],
                                       ENTITY_NAME_2_ID['clause_number'],
                                       ENTITY_NAME_2_ID['annex'],
                                       ENTITY_NAME_2_ID['sub_clause_title'],
                                       ENTITY_NAME_2_ID['sub_clause_number']]}

        if not clauses:
            # The dictionary clauses is empty, there was no clause annotated in this document
            # We'd rather skip that document, manual check on a sample prooved that all these documents
            # had clause, so we would generate bad training data if these texts were making it into the dataset
            continue

        for p in contract_text.parts():
            clause_type = 'none'
            if p in clauses:
                clause_type = ENTITY_ID_2_NAME[clauses[p]['classId']]

            data.append({'uuid': uuid,
                         'part': p,
                         'text': contract_text[p],
                         'clause_begin': p in clauses,
                         'clause_type': clause_type})

    pd.DataFrame(data).to_csv(_args.dest, index=False)


def consolidate_redflags() -> None:
    """
    Uses the data in the original format (files are located in the folder _args.original).
    Creates a consolidated CSV file for Red Flag Detection in the _args.dest file.
    The CSV file has the following columns: text, redflag (boolean), redflag_type (categorical)
    :return:
    """
    # Collect the data
    raw_dataset = OriginalDataset(original_folder=_args.original)
    data = []

    # Iterate through annotations
    for uuid, d in tqdm.tqdm(raw_dataset):
        annotations: Dict[str, Any] = d.contract_annotations
        contract_text: Contract = d.contract_text

        # All redflags from this contract
        redflags = [x for x in annotations['entities'] if x['classId'] == ENTITY_NAME_2_ID['redflag']]
        document_parts = set(contract_text.parts())
        parts_with_redflags = set([x['part'] for x in redflags])
        parts_without_redflags = document_parts - parts_with_redflags

        # A few rules coming from the annotation process:
        # 2 consecutive red flags (ie the same redflag type in 2 consecutive document parts) should be merged
        # One document part can contain nested red flags. Still consider the outer redflag for merging with next part
        # We fill up the dictionary data like this
        # key = ('s1p30-expansion', 's1p31-expansion') so 'PART-FLAGTYPE'
        # When processing a new redflag, the process is: IF the previous part is at the end of one the keys (each
        # key is an ordered tuple) THEN extend this key, accumulate text, replace the key in the dictionary
        # The value is always a dictionary for a redflag {'type', 'text', 'start'}
        def make_key(part_id: str, _type: str) -> str:
            return '{}-{}'.format(part_id, _type)

        def add_key_to_keychain(existing_keys: Tuple[str], _new_key: str) -> Tuple[str]:
            return tuple(list(existing_keys) + [_new_key])

        def new_keychain(_first_key: str) -> Tuple[str]:
            return tuple([_first_key])

        def find_keychain(end_with_key: str) -> Optional[Tuple[str]]:
            _found_key = None
            for k in document_data.keys():
                # k is a tuple ('PART-TYPE', 'PART-TYPE', ...)
                if k[-1] == end_with_key:
                    _found_key = k
                    break
            return _found_key

        # To work properly, we need to sort the redflags by ascending value of part
        sorted_redflags = sorted(redflags, key=lambda x: x['part'])

        document_data = {}
        for r in sorted_redflags:
            if 'f_24' not in r['fields']:
                continue
            redflag_part = r['part']
            redflag_type = r['fields']['f_24']['value']
            redflag_text = r['offsets'][0]['text']
            redflag_start = r['offsets'][0]['start']

            # Is it a standalone redflag, or is it continuing an existing one ?
            # Is the current part the next of an existing redflag ?
            curr_part_as_key = make_key(redflag_part, redflag_type)
            prev_part_as_key = make_key(prev_part(redflag_part), redflag_type)
            found_keychain = find_keychain(prev_part_as_key)

            if found_keychain is not None:
                # We extend an existing redflag
                # We pop the item, as we will modify the key by adding an element to the tuple
                existing_redflag: Dict = document_data.pop(found_keychain)
                new_keychain_ = add_key_to_keychain(found_keychain, curr_part_as_key)

                modified_redflag = existing_redflag.copy()
                modified_redflag['text'] = '{}\n{}'.format(existing_redflag['text'], redflag_text)
                document_data[new_keychain_] = modified_redflag
            else:
                # This is something completly new
                document_data[new_keychain(curr_part_as_key)] = {'type': redflag_type,
                                                                 'text': redflag_text,
                                                                 'start': redflag_start}

        # For one contract we flatten all our observations:
        # All parts without redflag annotation are taken as negative samples
        # All parts with redflags are positive samples
        def key_chain_to_parts(key_chain_: Tuple[str]) -> Tuple[str]:
            key_chain_re = re.compile(r'^(?P<partid>\w+)-(?P<type>\w+)$')
            t = []
            for k in key_chain_:
                m = key_chain_re.match(k)
                assert m is not None
                t.append(m.group('partid'))

            return tuple(t)

        for key_chain, v in document_data.items():
            raw_text = '\n'.join((contract_text[x] for x in key_chain_to_parts(key_chain)))
            data.append({'uuid': uuid,
                         'text': v['text'],
                         'type': v['type'],
                         'raw_text': raw_text,
                         'start': v['start'],
                         'end': v['start'] + len(v['text'])})

        for p in parts_without_redflags:
            data.append({'uuid': uuid,
                         'text': '',
                         'type': 'none',
                         'raw_text': contract_text[p],
                         'start': 0,
                         'end': 0})

    # Just output the CSV file
    pd.DataFrame(data).to_csv(_args.dest, index=False)


def consolidate_easy_redflags() -> None:
    """
    Uses the data in the original format (files are located in the folder _args.original).
    Creates a consolidated CSV file for 'EASY' Red Flag Detection in the _args.dest file.
    The CSV file has the following columns: text, redflag (boolean), redflag_type (categorical)
    :return:
    """
    # Collect the data
    raw_dataset = OriginalDataset(original_folder=_args.original)
    data = []

    # Iterate through annotations
    for uuid, d in tqdm.tqdm(raw_dataset):
        annotations: Dict[str, Any] = d.contract_annotations
        contract_text: Contract = d.contract_text

        # All redflags from this contract
        redflags = [x for x in annotations['entities'] if x['classId'] == ENTITY_NAME_2_ID['redflag']]
        document_parts = set(contract_text.parts())
        parts_with_redflags = set([x['part'] for x in redflags])
        parts_without_redflags = document_parts - parts_with_redflags

        for r in redflags:
            if 'f_24' not in r['fields']:
                continue
            redflag_part = r['part']
            redflag_type = r['fields']['f_24']['value']
            redflag_text = r['offsets'][0]['text']
            redflag_start = r['offsets'][0]['start']

            data.append({
                'uuid': uuid,
                'part': redflag_part,
                'text': redflag_text,
                'type': redflag_type,
                'raw_text': contract_text[redflag_part],
                'start': redflag_start,
                'end': redflag_start + len(redflag_text),
            })

        for p in parts_without_redflags:
            data.append({
                'uuid': uuid,
                'part': p,
                'text': '',
                'type': 'none',
                'raw_text': contract_text[p],
                'start': 0,
                'end': 0
            })

    # Just output the CSV file
    pd.DataFrame(data).to_csv(_args.dest, index=False)


def consolidate_docs_types() -> None:
    """
    Uses the data in the original format (files are located in the folder _args.original).
    Creates a consolidated CSV file for Document Classification in the _args.dest file
    :return:
    """
    # Collect the data
    raw_dataset = OriginalDataset(original_folder=_args.original)
    data = []

    # Iterate through the annotations
    for k, d in raw_dataset:
        k: str
        d: OriginalSample
        uuid = k
        annotations: Dict[str, Any] = d.contract_annotations
        contract_full_text = str(d.contract_text)
        try:
            contract_class = annotations['metas']['m_22']['value']
        except KeyError:
            continue

        data.append({'uuid': uuid,
                     'document_full_text': contract_full_text,
                     'document_class': contract_class})

    # Save to file
    pd.DataFrame(data).to_csv(_args.dest, index=False)


# Entities that will be filtered out if '--filter' is used.
# Structure mining is a known task
structure = ('sub_clause_number', 'clause_number', 'clause_title', 'sub_clause_title',
             'definition', 'definition_number', 'annex')

# Not enough material, or not consistently annotated
inconsistent = ('indexation_rent', 'annex', 'type_lease')

# Redflag is a separate task
leave_apart = ('redflag',)

FILTER_OUT = structure + inconsistent + leave_apart


def consolidate_entities() -> None:
    """
    Uses the data in the original format (files are located in the folder _args.original).
    Creates a consolidated CSV file for Entity Recognition in the _args.dest file
    The CSV has the following fields: UUID, CLASS_ID, FULL_TEXT, ENTITY_TEXT, ENTITY_START
    :return:
    """
    # Collect the data
    raw_dataset = OriginalDataset(original_folder=_args.original)
    csv_data = []
    conll_data: List[List[Dict[str, Any]]] = []

    # Iterate through the annotations
    for k, d in tqdm.tqdm(raw_dataset):
        k: str
        d: OriginalSample
        uuid = k
        annotations: Dict[str, Any] = d.contract_annotations
        text: Contract = d.contract_text

        annotation_entities = annotations['entities']
        contract_data = []
        part_map = {}
        for entity in annotation_entities:
            # entity is a Dict
            # {
            #       "classId": "e_6",
            #       "part": "s1p3",
            #       "offsets": [
            #         {
            #           "start": 6,
            #           "text": "June 25, 2013"
            #         }
            #       ],
            #       "coordinates": [],
            #       "confidence": {
            #         "state": "pre-added",
            #         "who": [
            #           "user:terezalat"
            #         ],
            #         "prob": 1
            #       },
            #       "fields": {},
            #       "normalizations": {}
            # },
            entity_class_id = ENTITY_ID_2_NAME[entity['classId']]
            if _args.filter and entity_class_id in FILTER_OUT:
                # The option '--filter' activates the filtering out of some entities
                continue

            part_id = entity['part']
            part_text = text[part_id]
            entity_text = entity['offsets'][0]['text']
            entity_start = entity['offsets'][0]['start']
            entity_end = entity_start + len(entity_text)

            contract_data.append(
                {
                    'uuid': uuid,
                    'part_id': part_id,
                    'class_id': entity_class_id,
                    'full_text': part_text,
                    'entity_text': entity_text,
                    'entity_start': entity_start,
                    'entity_end': entity_end
                }
            )

            part_map_item = part_map.get(part_id, [])
            part_map_item.append(
                {
                    'start': entity_start,
                    'end': entity_end,
                    'tag': entity_class_id
                }
            )
            part_map[part_id] = part_map_item

        if _args.csv is not None:
            csv_data.extend(contract_data)

        parts_with_entities = set([x['part_id'] for x in contract_data])
        parts_without_entities = set(text.parts()) - parts_with_entities

        if len(parts_with_entities) == 0:
            # A contract where no part was annotated with entities should not be in the dataset
            # It would generate bad training samples
            continue

        if _args.conll is not None:
            for p in parts_without_entities:
                conll_data.extend(text_to_conll(text[p], []))

            for p in parts_with_entities:
                conll_data.extend(text_to_conll(text[p], part_map[p]))

    # Save to file
    if _args.csv is not None:
        pd.DataFrame(csv_data).to_csv(_args.csv, index=False)

    # Now for CoNLL data
    if _args.conll is not None:
        train, test = train_test_split(conll_data, test_size=0.2)
        for split_name, split_data in zip([Split.train.value, Split.test.value], [train, test]):
            with open(os.path.join(_args.conll, f'{split_name}.txt'), 'w') as dst:
                for txt in split_data:
                    for tag in txt:
                        dst.write(f'{tag["word"]} {tag["tag"]}\n')
                    dst.write('\n')

        labels = set([x['tag'] for y in conll_data for x in y])
        with open(os.path.join(_args.conll, 'labels.txt'), 'w') as dst:
            dst.write('\n'.join(sorted(labels)))


def text_to_conll(txt: str, entities: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Transform a text to CoNLL annotations.
    The list entities is a list of dicts with keys 'start', 'end', 'tag'.
    It will add the IOB2 flags
    :param txt:
    :param entities:
    :return: list of dicts with keys 'word', 'tag'
    """
    # We will cut the text into sentences
    sentences = sent_tokenize(txt)

    conll = []   # List of List of dicts with keys 'word' / 'tag'

    # tracking where we are in the original sentence
    # The entities start / end are given in characters from the original string
    # So we have to be careful that we are well aligned: there could be more than one space between words
    curr_pos = 0
    for s in sentences:
        # Now word tokenization
        # Tags are assigned to tokens
        tokens = word_tokenize(s)
        tokens = tokens[:int(_args.max_seq_len * 0.7)]    # Max number of tokens = 70% of max_seq_len

        # Fun fact, word_tokenizer modified the double quotes " into `` and ''
        # See https://github.com/nltk/nltk/issues/1630
        # We correct it
        tokens = list(map(lambda x: '"' if x in ['``', "''"] else x, tokens))

        curr_tag: Optional[str] = None
        conll_sentence: List[Dict[str, str]] = []  # List of dict with keys 'word' / 'tag'
        for curr_tok, next_tok in zip(tokens, tokens[1:] + [None]):
            # Move cursor to the current token
            while txt[curr_pos] != curr_tok[0]:
                curr_pos += 1

            # Deal with curren token
            tag: Optional[str] = None
            for e in entities:
                if e['start'] <= curr_pos <= e['end']:
                    tag = e['tag']
                    break

            if tag is None:
                # The current token has no entity associated to it
                iob_tag = 'O'
            elif curr_tag is None:
                # No current tag, so we are in a new Tag-> BEGIN tag
                iob_tag = f'B-{tag}'
                curr_tag = tag
            elif tag == curr_tag:
                # tag is the same for previous word and current word -> we are INSIDE a tag
                iob_tag = f'I-{tag}'
            else:
                # We start a new tag -> BEGIN tag
                iob_tag = f'B-{tag}'
                curr_tag = tag

            conll_sentence.append(
                {
                    'word': curr_tok,
                    'tag': iob_tag
                }
            )

            # Move cursor over the current token
            curr_pos += len(curr_tok)

        # Transformed 1 sentence into CoNLL format
        conll.append(conll_sentence)

    return conll


def parse_args(argstxt=None) -> Namespace:
    if argstxt is None:
        argstxt = sys.argv[1:]
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title='Subcommands', description='Valid subcommands',
                                       help='Additional help')

    # Generate a CSV file for the dataset for Entity Detection / Information Extraction
    parser_entities = subparsers.add_parser('entities')
    parser_entities.add_argument('--original', type=str, help='Path to folder containing the original data')
    parser_entities.add_argument('--csv', type=str, help='Path to output the CSV with consolidated dataset')
    parser_entities.add_argument('--conll', type=str, help='Path to output the CoNLL dataset')
    parser_entities.add_argument('--filter', action='store_true', help='Filter the entities. '
                                                                       'Uses `prepare_data.ENTITIES_OUT`')
    parser_entities.add_argument('--max_seq_len', type=int, default=None, help='Maximum sentence length.')
    parser_entities.set_defaults(func=consolidate_entities)

    # Generate a CSV file for the dataset for Document Classification
    parser_docclass = subparsers.add_parser('docclass')
    parser_docclass.add_argument('--original', type=str, help='Path to folder containing the original data')
    parser_docclass.add_argument('--dest', type=str, help='Path to output the CSV with consolidated dataset')
    parser_docclass.set_defaults(func=consolidate_docs_types)

    # Generate a CSV file for RedFlag detection task
    parser_redflags = subparsers.add_parser('redflags')
    parser_redflags.add_argument('--original', type=str, help='Path to folder containing the original data')
    parser_redflags.add_argument('--dest', type=str, help='Path to output the CSV with consolidated dataset')
    parser_redflags.set_defaults(func=consolidate_redflags)

    # Generate a CSV file for 'EASY' RedFlag detection task (no multi-part redflag)
    parser_redflags = subparsers.add_parser('easy_redflags')
    parser_redflags.add_argument('--original', type=str, help='Path to folder containing the original data')
    parser_redflags.add_argument('--dest', type=str, help='Path to output the CSV with consolidated dataset')
    parser_redflags.set_defaults(func=consolidate_easy_redflags)

    # Generate a CSV file for Clause detection task
    parser_clauses = subparsers.add_parser('clauses')
    parser_clauses.add_argument('--original', type=str, help='Path to folder containing the original data')
    parser_clauses.add_argument('--dest', type=str, help='Path to output the CSV with consolidated dataset')
    parser_clauses.set_defaults(func=consolidate_clauses)

    return parser.parse_args(argstxt)


def main():
    global _args
    _args = parse_args()
    _args.func()


if __name__ == '__main__':
    main()
#    try:
#    except Exception as excp:
#        import pdb
#        pdb.post_mortem()
