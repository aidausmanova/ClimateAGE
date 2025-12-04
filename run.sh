#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --constraint=48GB
#SBATCH --job-name=climateage
#SBATCH -o /storage/usmanova/ClimateAGE/logs/Graph/run_%j.out # STDOUT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

report="Meta 2023 Sustainability Report" # "2022 Microsoft Environmental Sustainability Report"

# meta-llama/Llama-3.1-70B-Instruct, Llama-3.1-8B-Instruct
# python -m src.enrich_taxonomy --llm 'meta-llama/Llama-3.1-8B-Instruct' --taxonomy 'data/ifrs_sds_taxonomy' --load_type openai
# python -m src.embedding_model.create_embeddings --taxonomy 'data/ifrs_sds_taxonomy_enriched_Llama-3.1-70B-Instruct'

# python -m src.extract_nouns --report "$report"
# python -m src.open_ie --report "$report" --llm 'Llama-3.1-70B-Instruct'  --corpus 'granular' --experiment 'base'
# python -m src.entity_linking --report "$report" --llm 'Llama-3.1-70B-Instruct'  --corpus 'granular' --experiment 'base' --threshold 50
python -m src.create_graph --report "$report" --corpus 'granular' --experiment 'base' --taxonomy 'ifrs_sds_taxonomy_enriched_Llama-3.1-70B-Instruct'