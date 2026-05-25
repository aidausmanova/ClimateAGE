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

def run_graph_build_retrieval(
        report_name, taxonomy, question_type=None,
        use_link_edges=True, use_synonym_edges=True,
        use_llm_reranking=True, retrieval_mode='auto',
        condition_tag=None):

    print("\n=== Experiment INFO ===")
    print("[INFO] Task: Graph Construction")
    print("[INFO] Report: ", report_name)
    print("[INFO] Condition: ", condition_tag or "baseline")

    graph = ReportKnowledgeGraph(report_name, taxonomy,
                                 use_link_edges=use_link_edges,
                                 use_synonym_edges=use_synonym_edges,
                                 use_llm_reranking=use_llm_reranking,
                                 retrieval_mode=retrieval_mode,
                                 condition_tag=condition_tag)

    print("[INFO] Starting retreival ...")
    all_samples = json.load(open(f"{PATH['weakly_supervised']['path']}{report_name}/gold.json", "r"))
    samples = []
    if question_type:
        for s in all_samples:
            if s['type'] == question_type:
                samples.append(s)
    else:
        samples = all_samples

    all_queries = [s['question'] for s in samples]

    # Descriptive, non-factoid / Factoid
    gold_docs = get_gold_docs(samples, report_name)
    gold_answers = get_gold_answers(samples)

    if gold_docs is not None and gold_answers is not None:
        assert len(all_queries) == len(gold_docs) == len(gold_answers), \
            "Length of queries, gold_docs, and gold_answers should be the same."
        queries, overall_retrieval_result = graph.retrieve(queries=all_queries, num_to_retrieve=15, gold_docs=gold_docs)
    elif gold_docs is not None:
        assert len(all_queries) == len(gold_docs), "Length of queries and gold_docs should be the same."
        queries, overall_retrieval_result = graph.retrieve(queries=all_queries, num_to_retrieve=15, gold_docs=gold_docs)
    else:
        queries = graph.retrieve(queries=all_queries, num_to_retrieve=15)
        overall_retrieval_result = None

    print(f"Time now: {datetime.datetime.now()}. Time elapsed: {datetime.datetime.now() - start_time}")

    return overall_retrieval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClimateAGE Graph")
    parser.add_argument('--report', type=str, default='')
    parser.add_argument('--taxonomy', type=str)
    parser.add_argument('--question_type', type=str)
    # parser.add_argument('--use_link_edges', type=bool, default=True)
    # parser.add_argument('--use_synonym_edges', type=bool, default=True)
    # parser.add_argument('--use_llm_reranking', type=bool, default=True)
    
    parser.add_argument('--use_link_edges', type=lambda x: x.lower() != 'false', default=True)
    parser.add_argument('--use_synonym_edges', type=lambda x: x.lower() != 'false', default=True)
    parser.add_argument('--use_llm_reranking', type=lambda x: x.lower() != 'false', default=True)
    parser.add_argument('--retrieval_mode', type=str, default='auto')
    parser.add_argument('--condition_tag', type=str, default=None)

    args = parser.parse_args()
    report_name = args.report
    taxonomy = args.taxonomy
    question_type = args.question_type
    use_link_edges = args.use_link_edges
    use_synonym_edges = args.use_synonym_edges
    use_llm_reranking = args.use_llm_reranking
    retrieval_mode = args.retrieval_mode
    condition_tag = args.condition_tag

    run_graph_build_retrieval(report_name, taxonomy,
                              question_type,
                              use_link_edges,
                              use_synonym_edges,
                              use_llm_reranking,
                              retrieval_mode,
                              condition_tag)
    
    exit()