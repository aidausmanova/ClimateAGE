import os
import argparse
from typing import List
import json
import datetime

from src.graph.kg import KnowledgeGraphBuilder
from src.embedding_model.NVEmbedV2 import NVEmbedV2EmbeddingModel
from src.utils.embedding_store import EmbeddingStore, retrieve_knn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


start_time = datetime.datetime.now()
print(f"Start time: {start_time}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="ClimateAGE Graph")
    # parser.add_argument('--report', type=str, default='')
    # args = parser.parse_args()

    # report_name = args.report
    report_name = "Netflix ESG Report 2022"
    print("\n=== Experiment INFO ===")
    print("[INFO] Task: Graph Construction")
    print("[INFO] Report: ", report_name)
    
    graph = KnowledgeGraphBuilder(report_name)
#     embedding_model = NVEmbedV2EmbeddingModel(batch_size=8)
#     taxonomy_embedding_store = EmbeddingStore(embedding_model, "outputs/graph/taxonomy_embeddings", embedding_model.batch_size, 'taxonomy')

#     with open("data/ifrs_taxonomy_enriched-Llama70B.json", "r") as f:
#             taxonomy_data = json.load(f)

#     taxonomy_texts = []
#     for uid, concept in taxonomy_data.items():
#         taxonomy_texts.append((uid, f"Label: {concept['prefLabel']}\nDefinition:{concept['enriched_definition']}\nRelated terms: {concept['relatedTerms']}"))
#     taxonomy_embedding_store.insert_strings(taxonomy_texts)

    
