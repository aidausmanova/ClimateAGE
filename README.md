# Taxonomy-Populated Graph for Evidence Retrieval

### Abstract
Detecting greenwashing requires locating specific evidence scattered across corporate sustainability reports (CSRs) that span hundreds of pages of technically and jargon dense text. 
This is fundamentally an evidence retrieval problem. Yet standard embedding-based retrieval and Retrieval-Augmented Generation (RAG) fail to resolve the terminology gaps, boilerplate phrasing, and cross-paragraph dependencies that are common for regulatory disclosure documents.
We present a taxonomy-populated graph-based retrieval method that embeds expert-curated disclosure taxonomy concepts directly as semantic anchors in a unified knowledge graph, bridging the gap between auditor query language and corporate disclosure language. 
Evaluated on two sustainability reporting benchmarks, ClimRetrieve and SustainableQA, our method achieves 29.6% and 24.7% relative improvement in Recall@5 over the respective retrieval baselines. 
Ablation confirms that taxonomy grounding provides consistent gains for semantically ambiguous audit queries, which are most relevant to greenwashing detection.

---------
#### Table 5. Average statistics on generated graphs.

| Entity | Value |
| --- | ---: |
| Total nodes | 2,097 |
| Total edges | 6,110 |
| Concept nodes | 427 |
| Entity nodes | 1,485 |
| Paragraph nodes | 185 |
| *IsSubtopicOf* edges | 744 |
| *IsLinkedTo* edges | 541 |
| *IsExtractedFrom* edges | 2,548 |
| *IsSynonymOf* edges | 897 |
| Predicate edges | 1,380 |
| Entity nodes with *IsLinkedTo* edges | 23.5% |
| Entity nodes with *IsSynonymOf* edges | 27.7% |

#### Table 6. Noun filtering threshold results.

| τ | R@1 | R@5 | R@10 | R@15 | N@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 0.204 | 0.578 | 0.699 | 0.749 | 0.607 | 0.712 |
| 0.45 | 0.230 | 0.532 | 0.693 | 0.732 | 0.591 | 0.696 |
| 0.55 | 0.170 | 0.498 | 0.702 | 0.749 | 0.546 | 0.612 |
| 0.65 | 0.180 | 0.473 | 0.693 | 0.749 | 0.558 | 0.696 |

#### Table 7. Entity-linking threshold results.

| τ | R@1 | R@5 | R@10 | N@10 | MRR | % linked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 0.165 | 0.410 | 0.502 | 0.429 | 0.539 | 83.5 |
| 0.40 | 0.162 | 0.387 | 0.473 | 0.416 | 0.528 | 47.8 |
| 0.50 | 0.175 | 0.400 | 0.493 | 0.435 | 0.557 | 21.0 |
| 0.60 | 0.126 | 0.391 | 0.517 | 0.413 | 0.495 | 8.2 |
| 0.70 | 0.131 | 0.342 | 0.485 | 0.387 | 0.501 | 2.6 |

#### Table 8. Synonym clustering threshold results.

| τ | R@1 | R@5 | R@10 | R@15 | N@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.60 | 0.171 | 0.409 | 0.513 | 0.613 | 0.448 | 0.547 |
| 0.70 | 0.178 | 0.409 | 0.509 | 0.588 | 0.445 | 0.568 |
| 0.75 | 0.183 | 0.402 | 0.507 | 0.583 | 0.445 | 0.569 |
| 0.80 | 0.175 | 0.400 | 0.493 | 0.592 | 0.435 | 0.557 |
| 0.85 | 0.166 | 0.388 | 0.483 | 0.591 | 0.419 | 0.530 |
| 0.90 | 0.162 | 0.388 | 0.483 | 0.591 | 0.413 | 0.511 |

#### Table 9. Ablation study isolating the contribution of individual pipeline components on SustainableQA Factoid questions.

| Configuration | Recall@1 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| w/o taxonomy | 0.638 | 0.886 | 0.93 | 0.773 | 0.788 | 0.742 |
| w/o *IsLinkedTo* edges | 0.6347 | 0.8948 | 0.9543 | 0.7769 | 0.7976 | 0.7462 |
| w/o LLM judge | 0.3412 | 0.7661 | 0.9369 | 0.5556 | 0.6138 | 0.5125 |
| PPR only | 0.6343 | 0.8940 | 0.9708 | 0.7764 | 0.7975 | 0.7460 |

#### Table 10. Retrieval performance by embedding model on ClimRetrieve.

| Embedding Model | R@5 | R@10 | R@15 |
| --- | ---: | ---: | ---: |
| NV-Embed-v2 | 0.293 | 0.431 | 0.521 |
| BGE-large-en-v1.5 | 0.164 | 0.269 | 0.348 |
| all-mpnet-base-v2 | 0.156 | 0.263 | 0.320 |

---------
### Setup
1. Clone repository and setup environment
   ```bash
   conda create --name venv python=3.11
   conda activate venv
   pip install -r requirements.txt
   ```
3. The `data/reports/` folder contains sustainability reports. Original data can be downloaded from [ClimRetrieve](https://github.com/tobischimanski/ClimRetrieve/blob/main/Report-Level%20Dataset/ClimRetrieve_ReportLevel_V1.csv) and [SustainableQA](https://github.com/DataScienceUIBK/SustainableQA/tree/main/Data) repositories.
4. Create `outputs/` and `logs/` folders

### Running Experiments
There are four modules in the pipeline: noun extraction, triple extraction, entity linking and graph construction. Each module can be run separetly. To run the full pipeline with retrieval follow `run.sh` instructions.

1. Intitalize variables
   ```bash
   report=ReportName # name of the document from data/reports folder
   model=Llama-3.1-70B-Instruct # or any other llm
   ```
2. Running
   
   ```bash
   bash run.sh
   ```
