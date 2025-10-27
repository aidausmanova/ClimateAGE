import re
import string
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Callable

from .logging_utils import get_logger

logger = get_logger(__name__)


def normalize_answer(answer: str) -> str:
    """
    Normalize a given string by applying the following transformations:
    1. Convert the string to lowercase.
    2. Remove punctuation characters.
    3. Remove the articles "a", "an", and "the".
    4. Normalize whitespace by collapsing multiple spaces into one.

    Args:
        answer (str): The input string to be normalized.

    Returns:
        str: The normalized string.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def calculate_recall_k(gold_docs: List[List[str]], retrieved_docs: List[List[str]], k_list: List[int] = [1, 5, 10, 20]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Calculates Recall@k for each example and pools results for all queries.

    Args:
        gold_docs (List[List[str]]): List of lists containing the ground truth (relevant documents) for each query.
        retrieved_docs (List[List[str]]): List of lists containing the retrieved documents for each query.
        k_list (List[int]): List of k values to calculate Recall@k for.

    Returns:
        Tuple[Dict[str, float], List[Dict[str, float]]]: 
            - A pooled dictionary with the averaged Recall@k across all examples.
            - A list of dictionaries with Recall@k for each example.
    """
    k_list = sorted(set(k_list))
    
    example_eval_results = []
    pooled_eval_results = {f"Recall@{k}": 0.0 for k in k_list}

    for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_docs):
        if len(example_retrieved_docs) < k_list[-1]:
            logger.warning(f"Length of retrieved docs ({len(example_retrieved_docs)}) is smaller than largest topk for recall score ({k_list[-1]})")
        
        example_eval_result = {f"Recall@{k}": 0.0 for k in k_list}

        # Compute Recall@k for each k
        for k in k_list:
            # Get top-k retrieved documents
            top_k_docs = example_retrieved_docs[:k]
            # Calculate intersection with gold documents
            relevant_retrieved = set(top_k_docs) & set(example_gold_docs)
            # Compute recall
            if example_gold_docs:  # Avoid division by zero
                example_eval_result[f"Recall@{k}"] = len(relevant_retrieved) / len(set(example_gold_docs))
            else:
                example_eval_result[f"Recall@{k}"] = 0.0
        
        # Append example results
        example_eval_results.append(example_eval_result)
        
        # Accumulate pooled results
        for k in k_list:
            pooled_eval_results[f"Recall@{k}"] += example_eval_result[f"Recall@{k}"]

    # Average pooled results over all examples
    num_examples = len(gold_docs)
    for k in k_list:
        pooled_eval_results[f"Recall@{k}"] /= num_examples

    # round off to 4 decimal places for pooled results
    pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}
    return pooled_eval_results, example_eval_results

def calculate_exact_match(gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Calculates the Exact Match (EM) score.

    Args:
        gold_answers (List[List[str]]): List of lists containing ground truth answers.
        predicted_answers (List[str]): List of predicted answers.
        aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

    Returns:
        Tuple[Dict[str, float], List[Dict[str, float]]]: 
            - A dictionary with the averaged EM score.
            - A list of dictionaries with EM scores for each example.
    """
    assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

    example_eval_results = []
    total_em = 0

    for gold_list, predicted in zip(gold_answers, predicted_answers):
        em_scores = [1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0 for gold in gold_list]
        aggregated_em = aggregation_fn(em_scores)
        example_eval_results.append({"ExactMatch": aggregated_em})
        total_em += aggregated_em

    avg_em = total_em / len(gold_answers) if gold_answers else 0.0
    pooled_eval_results = {"ExactMatch": avg_em}

    return pooled_eval_results, example_eval_results

def calculate_f1(gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Calculates the F1 score.

    Args:
        gold_answers (List[List[str]]): List of lists containing ground truth answers.
        predicted_answers (List[str]): List of predicted answers.
        aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

    Returns:
        Tuple[Dict[str, float], List[Dict[str, float]]]: 
            - A dictionary with the averaged F1 score.
            - A list of dictionaries with F1 scores for each example.
    """
    assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

    def compute_f1(gold: str, predicted: str) -> float:
        gold_tokens = normalize_answer(gold).split()
        predicted_tokens = normalize_answer(predicted).split()
        common = Counter(predicted_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(predicted_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        return 2 * (precision * recall) / (precision + recall)

    example_eval_results = []
    total_f1 = 0.0

    for gold_list, predicted in zip(gold_answers, predicted_answers):
        f1_scores = [compute_f1(gold, predicted) for gold in gold_list]
        aggregated_f1 = aggregation_fn(f1_scores)
        example_eval_results.append({"F1": aggregated_f1})
        total_f1 += aggregated_f1

    avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
    pooled_eval_results = {"F1": avg_f1}

    return pooled_eval_results, example_eval_results