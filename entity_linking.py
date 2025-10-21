import os
import argparse
import datetime
import json
from collections import defaultdict
from tqdm import tqdm

from src.utils.consts import *
from src.utils.basic_utils import *
from extract_nouns import Retriever

start_time = datetime.datetime.now()
print(f"Start time: {start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=str, default='')
    parser.add_argument('--llm', type=str, default='Llama-3.3-70B-InstructB')
    parser.add_argument('--experiment', type=str, default='no_relation')
    parser.add_argument('--threshold', type=int, default=50)
    args = parser.parse_args()

    report_name = args.report
    threshold = args.threshold
    print("\n=== Experiment INFO ===")
    print("[INFO] Task: Entity Linking")
    print("[INFO] Report: ", report_name)

    input_dir = f"outputs/openie/{args.experiment}_{args.llm}"
    output_dir = f"outputs/postRAG/{args.experiment}_{args.llm}"

    input_files = os.listdir(input_dir)
    input_files = [
        file for file in input_files if os.path.isfile(f"{output_dir}/{file}") == False
    ]
    pbar = tqdm(input_files)
    RAG = Retriever(report=report_name)
    TAX = load_json_file(PATH["TAX"])
    
    for file in pbar:
        if os.path.isfile(f"{output_dir}/{file}"):
            continue
        preds = load_json_file(f"{input_dir}/{file}")
        post = defaultdict(list)
        nChunks = len(preds)
        for chunk_id, paragraph_key in enumerate(preds.keys()):
            pbar.set_description(f"Processing {file} paragraph {chunk_id}/{nChunks}")
            for pred in preds[paragraph_key]["entities"]:
                uuid, score = RAG.retrieve_by_def(pred["name"], pred["description"])
                if score > threshold:
                    pred.update({"taxonomy_uuid": uuid, "score": score})
                    post[paragraph_key].append(pred)
        RAG.save_retrieved()

        with open(f"{output_dir}/{file}", "w") as f:
            json.dump(post, f, indent=2)

    print(f"Time now: {datetime.datetime.now()}. Time elapsed: {datetime.datetime.now() - start_time}")
    exit()
