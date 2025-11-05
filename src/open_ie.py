# This file is used to extract all named entities with LLMs

import os
import torch
import json
import datetime
import argparse
from tqdm import tqdm
from huggingface_hub import login

from src.utils.consts import *
from src.utils.basic_utils import *
from extract_nouns import Retriever
from src.llm.meta_llama import InfoExtractor

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"\n=== GPU {i} ===")
        props = torch.cuda.get_device_properties(i)
        print(f"Name: {props.name}")
        print(f"Total Memory: {props.total_memory / 1e9:.2f} GB")
        # print(f"Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
        # print(f"Reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
        # print(f"Free: {(props.total_memory - torch.cuda.memory_allocated(i)) / 1e9:.2f} GB")
        
    # Clear cache
    # torch.cuda.empty_cache()
    # print("\n✓ Cleared CUDA cache")
else:
    print("CUDA not available")

start_time = datetime.datetime.now()
print(f"Time now: {start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=str, default='')
    parser.add_argument('--llm', type=str, default='Llama-3.3-70B-InstructB')
    parser.add_argument('--corpus', type=str, default='context')
    parser.add_argument('--experiment', type=str, default='no_relation')
    args = parser.parse_args()

    report_name = args.report
    print("\n=== Experiment INFO ===")
    print("[INFO] Task: Named Entity Extraction")
    print("[INFO] Report: ", report_name)
    print("[INFO] LLM model: ", args.llm)

    output_dir = f"outputs/openie/{args.corpus}_{args.experiment}_{args.llm}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")
    # prompt_dir = f"./outputs_prompts/{args.experiment}"
    # os.makedirs(prompt_dir, exist_ok=True)
    # conversations_dir = "outputs/conversations"
    today = (datetime.date.today().strftime("%Y-%m-%d") + f"_{args.corpus}_{args.experiment}_{args.llm}")

    MODEL = InfoExtractor(engine=args.llm, exp=args.experiment, use_vllm=False)
    RETRIEVER = Retriever(report=report_name)

    if args.corpus == "granular":
        corpus_file = "/corpus.json"
    else:
        corpus_file = "/corpus_1.json"
    with open(PATH["weakly_supervised"]['path']+report_name+corpus_file, "r") as f:
        data = json.load(f)
    

    ############# Run per each paragraph #############
    # text_chunks = []
    # for d in data:
    #     text_chunks.append((d['title']+" "+d['text'], d['idx']))
    # pbar = tqdm(text_chunks)
    # outputs = {}
    # conversations = {}
    # for text, idx in pbar:
    #     pbar.set_description(f"File [{report_name}] Paragraph [{idx}] Retrieving entities")
    #     retrieved_nodes = RETRIEVER.run(text)
    #     print(len(retrieved_nodes), " # nodes retreived")

    #     pbar.set_description(f"File [{report_name}] Chunk [{idx}] Running MODEL")
    #     output, conversation = MODEL.run(text, retrieved_nodes)
    #     outputs[idx] = output
    #     conversations[idx] = conversation
    #     pbar.set_description(f"Output paragraph [{idx}]: {output}")



    ############# Run per paragraph batch #############
    text_chunks = []
    ids = []
    for d in data:
        text_chunks.append(d['title']+" "+d['text'])
        ids.append(d['idx'])

    retrieved_nodes = []
    for text in text_chunks:
        node = RETRIEVER.run(text)
        if node: retrieved_nodes.append(RETRIEVER.run(text))

    print("[INFO] Eaxtrating named entities ...")
    model_outputs = MODEL.generate_responses(text_chunks, retrieved_nodes, 8)
    # model_outputs, model_conversations = MODEL.run_batch(text_chunks, list(retrieved_nodes))
    outputs = {idx: output for idx, output in zip(ids, model_outputs)}
    # conversations = {idx: conv for idx, conv in zip(ids, model_conversations)}

    # with open(f"{conversations_dir}/{report_name}.json", "w") as f:
    #     json.dump(conversations, f)
    with open(f"{output_dir}/{report_name}.json", "w") as f:
        json.dump(outputs, f, indent=4)
    
    print(f"Time now: {datetime.datetime.now()}. Time elapsed: {datetime.datetime.now() - start_time}")
    exit()
