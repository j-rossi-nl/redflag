"""
Helper classes and functions to manipulate the original data.

2020. Anonymous authors.
"""

import re
import os
import json
import bs4
import random
import string

from bs4 import BeautifulSoup
from zipfile import ZipFile
from pathlib import Path
from typing import Optional, Dict, Iterable, Any


# The organization of the original files
_ORIGINAL_DATA = {
    'datazip': 'pool.zip',
    'datafolder': 'pool',
    'legends': 'annotations-legend.json'
}

# Dictionary for entity names
ENTITY_ID_2_NAME: Dict[str, str] = {
    'e_1': 'lessor',
    'e_2': 'lessee',
    'e_3': 'leased_space',
    'e_4': 'designated_use',
    'e_5': 'type_lease',
    'e_6': 'signing_date',
    'e_8': 'expiration_date_of_lease',
    'e_9': 'term_of_payment',
    'e_10': 'indexation_rent',
    'e_11': 'rent_review_date',
    'e_12': 'notice_period',
    'e_13': 'extension_period',
    'e_14': 'vat',
    'e_15': 'clause_title',
    'e_16': 'clause_number',
    'e_17': 'sub_clause_title',
    'e_18': 'sub_clause_number',
    'e_19': 'definition',
    'e_20': 'definition_number',
    'e_23': 'redflag',
    'e_25': 'start_date',
    'e_26': 'end_date',
    'e_27': 'general_terms',
    'e_28': 'annex',
    'f_24': 'redflags',
    'm_22': 'Agreement_Type'}


# Reverse Dictionary
ENTITY_NAME_2_ID: Dict[str, str] = {v: k for k, v in ENTITY_ID_2_NAME.items()}


# The naming convention is :
# HTML : <hashcode>-<text>_xxx.plain.html
# JSON : <hashcode>-<text>_xxx.ann.json
# The raw text and annotation sh <hashcode>-<text>_xxx prefix
# We use <hashcode>-<text>_xxx as UUID
json_re = re.compile(r'^' + _ORIGINAL_DATA['datafolder'] + r'/.*\.ann\.json$')
uuid_re = re.compile(r'^' + _ORIGINAL_DATA['datafolder'] + r'/(?P<uuid>.+_[0-9]+)(\.plain\.html|\.ann\.json)$')


class Contract:
    """
    Represents a contract from our database.
    Facilitates navigation in parts of the text:

    `c = Contract(...)`
    `c[i] -> text of part with id=i`
    `str(c) -> full text`
    """
    def __init__(self, bsoup: BeautifulSoup):
        self.bsoup: BeautifulSoup = bsoup

    def __getitem__(self, item: str) -> str:
        """
        Retrieves the PART with the corresponding ID.
        Raises TypeError and KeyError.
        :param item: the part ID, as a string
        :return: The text for this document part, as a string
        """
        if not isinstance(item, str):
            raise TypeError('Expected str, received {}'.format(type(item)))

        # Type hinting raises an error. find() returns bs4.element.PageElement
        # This is an abstract class, and objects are actually of class bs4.element.Tag
        part_in_html: Any = self.bsoup.find('p', id=item)
        part_in_html: bs4.element.Tag
        if part_in_html is None:
            raise KeyError(item)

        return part_in_html.string

    def __str__(self) -> str:
        """
        Returns the full text of the contract
        :return: full text
        """
        return '\n'.join((self[x] for x in self.parts()))

    def parts(self) -> Iterable[str]:
        """
        Returns an iterator over the list of part_id that are in thise contract. This is an Iterable of str
        :return:
        """
        all_parts = (x['id'] for x in self.bsoup.find_all('p', id=True))
        return all_parts


class OriginalSample:
    """
    Represents one sample of the original dataset.
    Has 2 attributes:
       * contract_text : a Contract object of the original contract
       * contract_annotations : a Dict containing the annotations
    """
    def __init__(self, contract_text: Contract, contract_annotations: Dict[str, Any]):
        self.contract_text: Contract = contract_text
        self.contract_annotations: Dict[str, Any] = contract_annotations


class OriginalDataset:
    """
    Represents the original dataset
    """
# The organization of the original files
    _ORIGINAL_DATA = {
        'datazip': 'pool.zip',
        'datafolder': 'pool',
        'legends': 'annotations-legend.json'
    }

    def __init__(self, original_folder: Optional[str] = None, poolzipfile: Optional[str] = None):
        if original_folder is not None:
            self.datazip = ZipFile(Path(original_folder) / _ORIGINAL_DATA['datazip'], 'r')
        else:
            self.datazip = ZipFile(poolzipfile, 'r')

        # JSON files contain the annotation of the raw text
        self.namelist = self.datazip.namelist()
        self.dataset = {}

        for js in (x for x in self.datazip.namelist() if is_json_filename(x)):
            ht = get_html_for_json(js)
            if ht not in self.namelist:
                # This JSON has no corresponding HTML
                continue

            uuid = extract_uuid(js)
            annotations = json.loads(self.datazip.open(js).read())
            contract = Contract(BeautifulSoup(self.datazip.open(ht).read(), 'html.parser'))

            self.dataset[uuid] = OriginalSample(contract_text=contract, contract_annotations=annotations)

    def __iter__(self):
        """
        Builds an iterator to go through the dataset
        :return:
        """
        return iter(self.dataset.items())

    def __len__(self):
        """
        The number of annotated contracts in the dataset.
        :return:
        """
        return len(self.dataset)


def extract_uuid(s):
    """
    Extract the UUID from a file name
    :param s: filename
    :return: UUID
    """
    m = uuid_re.match(s)
    if m is None:
        return None

    return m.group('uuid')


def build_html_name(uuid):
    """
    Build the HTML name corresponding to the UUID
    :param uuid: UUID
    :return: HTML filename
    """
    return os.path.join(_ORIGINAL_DATA['datafolder'], uuid + '.plain.html')


def is_json_filename(name) -> bool:
    return json_re.match(name) is not None


def get_html_for_json(name) -> Optional[str]:
    if is_json_filename(name):
        return build_html_name(extract_uuid(name))
    else:
        return None


# Used to split a part_id into a section_number and part_number
part_re = re.compile(r'^s(?P<section>\d+)p(?P<part>\d+)$')


def is_next_part(a_part_id: str, b_part_id) -> bool:
    """
    Indicates whether b_part_id is the part immediatly following a_part_id
    A part_id is a string "sXpY". The rule is that `Xa==Xb` and `Yb==1+Ya`
    :param a_part_id:
    :param b_part_id:
    :return:
    """
    m_a = part_re.match(a_part_id)
    m_b = part_re.match(b_part_id)

    if m_a is None or m_b is None:
        return False

    return (m_a.group('section') == m_b.group('section')) and (int(m_a.group('part')) == int(m_b.group('part')) - 1)


def prev_part(part_id: str) -> str:
    """
    Gives the ID for the previous part
    :param part_id:
    :return:
    """
    m = part_re.match(part_id)
    if m is None:
        return ''

    # There is no sXp0 but in the worse case we return a non-existing name
    return 's{}p{}'.format(int(m.group('section')), max(0, int(m.group('part')) - 1))


def experience_uuid(n=4) -> str:
    """
    Generates a random string as a unique identifier for a model
    :return:
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))
