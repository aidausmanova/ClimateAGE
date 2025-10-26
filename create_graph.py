# This file is used to run graph construction task

import os
import argparse
from typing import List
import json
import datetime

from src.graph.kg import ReportKnowledgeGraph
from src.embedding_model.NVEmbedV2 import NVEmbedV2EmbeddingModel
from src.utils.embedding_store import EmbeddingStore, retrieve_knn
from src.utils.consts import *
from src.utils.basic_utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


start_time = datetime.datetime.now()
print(f"Start time: {start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClimateAGE Graph")
    parser.add_argument('--report', type=str, default='')
    args = parser.parse_args()
    report_name = args.report

    print("\n=== Experiment INFO ===")
    print("[INFO] Task: Graph Construction")
    print("[INFO] Report: ", report_name)
    
    graph = ReportKnowledgeGraph(report_name)
    
    print("[INFO] Starting retreival ...")
    samples = json.load(open(f"{PATH['weakly_supervised']['path']}{report_name}/gold.json", "r"))
    all_queries = [s['question'] for s in samples]

    gold_docs = get_gold_docs(samples, report_name)
    gold_answers = get_gold_answers(samples)
    assert len(all_queries) == len(gold_docs) == len(gold_answers), "Length of queries, gold_docs, and gold_answers should be the same."


    if gold_docs is not None:
        queries, overall_retrieval_result = graph.retrieve(queries=all_queries, num_to_retrieve=10, gold_docs=gold_docs)
    else:
        queries = graph.retrieve(queries=all_queries, num_to_retrieve=10)


    # embedding_model = NVEmbedV2EmbeddingModel(batch_size=8)
    # taxonomy_embedding_store = EmbeddingStore(embedding_model, "outputs/graph/taxonomy_embeddings", embedding_model.batch_size, 'taxonomy')

    # with open("data/ifrs_taxonomy_enriched-Llama70B.json", "r") as f:
    #         taxonomy_data = json.load(f)

    # taxonomy_texts = []
    # for uid, concept in taxonomy_data.items():
    #     taxonomy_texts.append((uid, f"Label: {concept['prefLabel']}\nDefinition:{concept['enriched_definition']}\nRelated terms: {concept['relatedTerms']}"))
    # taxonomy_embedding_store.insert_strings(taxonomy_texts)
    
    print(f"Time now: {datetime.datetime.now()}. Time elapsed: {datetime.datetime.now() - start_time}")
    exit()