# This file is used to link extracted entities with taxonomy concepts

import os
import argparse
import datetime
import json
from collections import defaultdict
from tqdm import tqdm

from .utils.consts import *
from .utils.basic_utils import *
from .extract_nouns import Retriever

start_time = datetime.datetime.now()
print(f"Start time: {start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=str, default='')
    parser.add_argument('--llm', type=str, default='Llama-3.3-70B-InstructB')
    parser.add_argument('--corpus', type=str, default='context')
    parser.add_argument('--experiment', type=str, default='no_relation')
    parser.add_argument('--threshold', type=int, default=50)
    args = parser.parse_args()

    report_name = args.report
    threshold = args.threshold
    print("\n=== Experiment INFO ===")
    print("[INFO] Task: Entity Linking")
    print("[INFO] Report: ", report_name)

    input_dir = f"outputs/openie/{args.corpus}_{args.experiment}_{args.llm}"
    output_dir = f"outputs/postRAG/{args.corpus}_{args.experiment}_{args.llm}"
    os.makedirs(output_dir, exist_ok=True)

    # input_files = os.listdir(input_dir)
    # input_files = [
    #     file for file in input_files if os.path.isfile(f"{output_dir}/{file}") == False
    # ]
    # pbar = tqdm(input_files)
    RAG = Retriever(report=report_name)
    TAX = load_json_file(PATH["TAX"])
    
    preds = load_json_file(f"{input_dir}/{report_name}.json")
    post = defaultdict(list)
    
    print("Linking entities ...")
    for entity in tqdm(preds['entities']):
        uuid, score = RAG.retrieve_by_def(entity["canonical_name"], entity["definition"])
        if score > threshold:
            entity.update({"taxonomy_uuid": uuid, "score": score})

    # for chunk_id, paragraph_key in enumerate(preds.keys()):
    #     for pred in preds[paragraph_key]["entities"]:
    #         uuid, score = RAG.retrieve_by_def(pred["canonical_name"], pred["definition"])
    #         if score > threshold:
    #             pred.update({"taxonomy_uuid": uuid, "score": score})
    #             # post[paragraph_key].append(pred)
    # # RAG.save_retrieved()

    with open(f"{output_dir}/{report_name}.json", "w") as f:
        json.dump(preds, f, indent=2)

    print(f"Time now: {datetime.datetime.now()}. Time elapsed: {datetime.datetime.now() - start_time}")
    exit()
