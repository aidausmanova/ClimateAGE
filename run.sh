#!/bin/bash
#SBATCH --gpus-per-node=2
#SBATCH --constraint=48GB
#SBATCH --job-name=climateage
#SBATCH -o /storage/usmanova/ClimateAGE/logs/Pipe/run_%j.out # STDOUT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

report="uniqa_insurance_group_ag" # baywa_ag_vna

# meta-llama/Llama-3.1-70B-Instruct, Llama-3.1-8B-Instruct
# python -m src.enrich_taxonomy --llm 'meta-llama/Llama-3.1-8B-Instruct' --taxonomy 'data/ifrs_sds_taxonomy' --load_type openai
# python -m src.embedding_model.create_embeddings --taxonomy 'data/ifrs_sds_taxonomy_enriched_Llama-3.1-70B-Instruct'

# python -m src.extract_nouns --report "$report"
# python -m src.open_ie --report "$report" --llm 'Llama-3.1-70B-Instruct'
# python -m src.entity_linking --report "$report" --llm 'Llama-3.1-70B-Instruct' --threshold 50
# python -m src.create_graph --report "$report" --taxonomy 'ifrs_sds_taxonomy_enriched_Llama-3.1-70B-Instruct' --question_type 'Descriptive, non-factoid' #'Factoid'

python -m run
# python -m src.bm25
# python -m src.utils.embedding_store