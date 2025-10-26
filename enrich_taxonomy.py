# This file is used to enrich taxonomy concept defintion with LLM 
# for NER in climate disclosure and based on its hierarchical relationships

import os
import json
import uuid
import torch
import transformers
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams

from src.prompts.prompt_template_manager import PROMPT_REFINE_DEFINITIONS

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'   
os.environ['VLLM_TORCH_COMPILE_LEVEL'] = '0'     

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='meta-llama/Llama-3.1-70B-Instruct')
    parser.add_argument('--taxonomy', type=str, default='data/taxonomy.json')
    parser.add_argument('--use_vllm', type=bool, default=False)
    args = parser.parse_args()
    
    MODEL_NAME = args.llm
    USE_VLLM = args.use_vllm
    taxonomy_path = args.taxonomy

    if USE_VLLM:
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.9,
            max_tokens=2048,
        )
        model = LLM(
            model=MODEL_NAME,
            task="generate",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_model_len=4096, # 4096,  # 25390,
            enable_prefix_caching=True,
        )
    else:
        # device = 0 if torch.cuda.is_available() else -1
        model = transformers.pipeline(
                        "text-generation",
                        model=MODEL_NAME,
                        model_kwargs={"torch_dtype": torch.bfloat16},
                        device_map="auto",
                    )
        processor = [
            model.tokenizer.eos_token_id,
            model.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    
    print(f"[INFO] Model {MODEL_NAME} loaded")

    with open(taxonomy_path, "r") as f:
        taxonomy_data = json.load(f)

    print(f"[INFO] Taxonomy loaded")

    enriched_tax_data = {}
    entity_id = 0
    entity_uuid_dict = {}
    prompts = []
    conversations = []

    for tax_entity in taxonomy_data:
        entity_metadata = f"Label: {tax_entity['prefLabel']}\nOntology path: {' > '.join(tax_entity['path_label'])}\nDefinition: {tax_entity['definitions'][0]['text']}"
        prompt = PROMPT_REFINE_DEFINITIONS.format(metadata=entity_metadata)
        prompts.append(prompt)
        conversations.append({"role": "user", "content": prompt})

    print(f"Processing {len(prompts)} entities...")
    if USE_VLLM:
        outputs = model.generate(prompts, sampling_params=sampling_params)
    else:
        outputs = model(
                        prompts,
                        max_new_tokens=4000,
                        pad_token_id=model.tokenizer.eos_token_id,
                        # eos_token_id=terminators,
                        do_sample=True,
                        temperature=0.6,
                        return_full_text=False,
                        batch_size=8,
                        # top_p=0.9,
                    )

    for tax_entity, output in tqdm(zip(taxonomy_data, outputs), total=len(taxonomy_data)):
        if USE_VLLM:
            pred_content = output.outputs[0].text
        else:
            pred_content = output["generated_text"]
    
        uid = str(uuid.uuid4())
        entity_uuid_dict[tax_entity['prefLabel']] = uid
        path_ids = [entity_uuid_dict[pl] for pl in tax_entity['path_label']]
        print(f"[INFO] Processing entity [{tax_entity['prefLabel']}]")
        
        enriched_tax_data[uid] = {
            "id": entity_id,
            "prefLabel": tax_entity['prefLabel'],
            "isLeaf": tax_entity['isLeaf'],
            "definitions": tax_entity['definitions'],
            "tags": tax_entity['tags'],
            "relatedTerms": tax_entity['relatedTerms'],
            "path_label": tax_entity['path_label'],
            "path_id": path_ids,
            "enriched_definition": pred_content
        }
        entity_id += 1

    with open("data/ifrs_taxonomy_enriched.json", "w") as f:
        json.dump(enriched_tax_data, f)

    print("[INFO] Taxonomy enriched saved in data/taxonomy_enriched.json")
