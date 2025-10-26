import json
import re
import spacy
import unicodedata
import numpy as np
from hashlib import md5
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from src.utils.consts import ABBREVIATIONS, ORG_SUFFIXES

nlp = spacy.load("en_core_web_sm")


text_template = "<heading>{}</heading>\n{}\n"

def load_json_file(file_path):
    """
    Loads a JSON file and returns its contents as a Python dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file as a dictionary.
    """
    with open(file_path, "r") as file:
        data = json.load(file)

    return data

def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs

def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers

def construct_regex(keyword):
    # split by all special character
    # parts = re.split(r'[-_\s]', keyword)
    parts = re.split(r"[^a-zA-Z0-9]", keyword)
    parts = [p for p in parts if len(p) > 0]

    # allow the special character to be optional
    # out = '_*-*\\s*'.join(parts)
    out = "[^a-zA-Z0-9]*".join(parts)

    # add word boundary to keyword
    return r"\b" + out + r"\b"

def remove_overlapping_mentions(mentions):
    # Sort mentions by start, end in descending order, and length in descending order
    mentions.sort(key=lambda x: x[1] - x[0], reverse=True)

    output = []

    for m in mentions:
        # Check if this mention overlaps with any mention in the output
        if not any(m[0] <= n[1] and m[1] >= n[0] for n in output):
            # If it doesn't, add it to the output
            output.append(m)
    return output

def remove_common_keys(dict1, dict2):
    # This code will remove all items from dict1 if the key appears in dict2.
    return {key: dict1[key] for key in dict1 if key not in dict2}

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()

def basic_normalize(text: str) -> str:
    if not text:
        return ""
    # Unicode normalization (NFKC - compatibility decomposition followed by canonical composition)
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = ' '.join(text.split())
    text = re.sub(r'[^\w\s\-]', ' ', text)
    text = ' '.join(text.split())
    return text.strip()

def normalize_organization_name(name: str) -> str:
    normalized = basic_normalize(name)
    words = normalized.split()
    
    # Remove common suffixes
    filtered_words = []
    for word in words:
        if word not in ORG_SUFFIXES:
            filtered_words.append(word)
    
    # If all words were suffixes, keep original
    if not filtered_words:
        filtered_words = words
    return ' '.join(filtered_words)

def expand_abbreviations(text: str) -> str:
    words = text.split()
    expanded_words = []
    for word in words:
        # Check if word is an abbreviation
        clean_word = word.strip('.,;:!?')
        if clean_word in ABBREVIATIONS:
            expanded_words.append(ABBREVIATIONS[clean_word])
        else:
            expanded_words.append(word)
    return ' '.join(expanded_words)

def lemmatization(text: str) -> str:
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return ' '.join(lemmas)

def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))