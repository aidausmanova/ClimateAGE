# This file is used to run graph construction task

import os
import argparse
from typing import List
import json
import datetime

from .graph.kg import ReportKnowledgeGraph
from .utils.consts import *
from .utils.basic_utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


start_time = datetime.datetime.now()
print(f"Start time: {start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClimateAGE Graph")
    parser.add_argument('--report', type=str, default='')
    # parser.add_argument('--corpus', type=str, default='context')
    parser.add_argument('--experiment', type=str, default='base')
    parser.add_argument('--taxonomy', type=str)
    args = parser.parse_args()
    report_name = args.report
    # corpus_type = args.corpus
    experiment = args.experiment
    taxonomy = args.taxonomy

    print("\n=== Experiment INFO ===")
    print("[INFO] Task: Graph Construction")
    print("[INFO] Report: ", report_name)
    
    graph = ReportKnowledgeGraph(report_name, experiment, taxonomy)
    
    print("[INFO] Starting retreival ...")
    samples = json.load(open(f"{PATH['weakly_supervised']['path']}{report_name}/gold.json", "r"))
    all_queries = [s['question'] for s in samples]

    gold_docs = get_gold_docs(samples, report_name)
    gold_answers = get_gold_answers(samples)
    assert len(all_queries) == len(gold_docs) == len(gold_answers), "Length of queries, gold_docs, and gold_answers should be the same."


    if gold_docs is not None:
        queries, overall_retrieval_result = graph.retrieve(queries=all_queries, num_to_retrieve=15, gold_docs=gold_docs)
    else:
        queries = graph.retrieve(queries=all_queries, num_to_retrieve=15)

    print(f"Time now: {datetime.datetime.now()}. Time elapsed: {datetime.datetime.now() - start_time}")
    exit()