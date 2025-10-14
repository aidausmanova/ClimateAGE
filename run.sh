#!/bin/bash
#SBATCH --gpus-per-node=2
#SBATCH --constraint=48GB
#SBATCH --job-name=climateage
#SBATCH -o /storage/usmanova/ClimateAGE/logs/Embeddings/run_%j.out # STDOUT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# python enrich_taxonomy.py --llm 'meta-llama/Llama-3.1-70B-Instruct' --taxonomy 'data/taxonomy.json'
# python create_embeddings.py --taxonomy 'data/ifrs_taxonomy_enriched-Llama70B.json'
# python extract_nouns.py

python open_ie.py --report 'Netflix ESG Report 2022' --llm 'Llama-3.3-70B-Instruct' --experiment 'no_relation'