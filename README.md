# Claims for Scrutiny: Taxonomy-Geounded Graph for Evidence Retrieval in Sustainability Reporting

### Abstract
Corporate sustainability reports are the medium for companies to communicate their environmental goals and are open for public scrutiny.
Auditors, journalists, and civil society organisations must locate relevant disclosures despite mismatch in terminology between their questions and corporate reporting language, and despite evidence scattered across lengthy reports.
We address this retrieval problem with a taxonomy-grounded graph retrieval method that connects audit terminology to corporate reporting language.
The method embeds expert-curated IFRS Sustainability Disclosure Taxonomy concepts as semantic anchors in a graph of extracted entities and source paragraphs. 
Evaluated on two datasets, ClimRetrieve and SustainableQA, the method achieves 29.6\% and 24.7\% Recall@5 relative gain.
Ablation confirms that taxonomy grounding helps to disambiguate queries and improve evidence retrieval.
This study presents domain-grounded information retrieval as a tool to support public scrutiny and highlights the conditions necessary for its practical benefit.

---------
#### Average statistics on generated graphs.

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

#### Noun filtering threshold results.

| τ | R@1 | R@5 | R@10 | R@15 | N@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 0.204 | 0.578 | 0.699 | 0.749 | 0.607 | 0.712 |
| 0.45 | 0.230 | 0.532 | 0.693 | 0.732 | 0.591 | 0.696 |
| 0.55 | 0.170 | 0.498 | 0.702 | 0.749 | 0.546 | 0.612 |
| 0.65 | 0.180 | 0.473 | 0.693 | 0.749 | 0.558 | 0.696 |

#### Entity-linking threshold results.

| τ | R@1 | R@5 | R@10 | N@10 | MRR | % linked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 0.165 | 0.410 | 0.502 | 0.429 | 0.539 | 83.5 |
| 0.40 | 0.162 | 0.387 | 0.473 | 0.416 | 0.528 | 47.8 |
| 0.50 | 0.175 | 0.400 | 0.493 | 0.435 | 0.557 | 21.0 |
| 0.60 | 0.126 | 0.391 | 0.517 | 0.413 | 0.495 | 8.2 |
| 0.70 | 0.131 | 0.342 | 0.485 | 0.387 | 0.501 | 2.6 |

#### Synonym clustering threshold results.

| τ | R@1 | R@5 | R@10 | R@15 | N@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.60 | 0.171 | 0.409 | 0.513 | 0.613 | 0.448 | 0.547 |
| 0.70 | 0.178 | 0.409 | 0.509 | 0.588 | 0.445 | 0.568 |
| 0.75 | 0.183 | 0.402 | 0.507 | 0.583 | 0.445 | 0.569 |
| 0.80 | 0.175 | 0.400 | 0.493 | 0.592 | 0.435 | 0.557 |
| 0.85 | 0.166 | 0.388 | 0.483 | 0.591 | 0.419 | 0.530 |
| 0.90 | 0.162 | 0.388 | 0.483 | 0.591 | 0.413 | 0.511 |

#### Retrieval performance by embedding model on ClimRetrieve.

| Embedding Model | R@5 | R@10 | R@15 |
| --- | ---: | ---: | ---: |
| NV-Embed-v2 | 0.293 | 0.431 | 0.521 |
| BGE-large-en-v1.5 | 0.164 | 0.269 | 0.348 |
| all-mpnet-base-v2 | 0.156 | 0.263 | 0.320 |

Replacing NV-Embed-v2 with BGE-large-en-v1.5 dramatically drops the performance of the pipeline. The graph structure does not compensate for weaker embeddings. Hence, there is a need for domain fine-tuned embedding models.

### Reproducibility

All experiments were conducted on a server with two NVIDIA RTX A6000 GPUs
(48GB VRAM each), 128GB RAM, and an AMD EPYC 7542 CPU. For entity extraction,
we utilise Llama-3.1-70B-Instruct. As an embedding model in all steps, we use
nvidia/Nv-Embed-v2 with default parameters. To construct a graph we used directed
graphs with self loops (DiGraph) from NetworkX library. PPR damping factor is
𝛼 = 0.5, following the established configuration of HippoRAG. The cosine similarity thresholds for entity linking (IsLinkedTo edges) and synonyms (IsSynonymOf edges)
were set to the values that best ensure string semantic matching. The KNN neighbourhood
size for synonym clustering is 𝑘 = 5. All results are reported from single runs, due to
deterministic outputs of LLM inference calls.

All datasets used in the experiments are publicly available research benchmarks
(ClimRetrieve and SustainableQA). Each with licences permitting academic use.
No new data were collected, all textual data was taken from the original datasets.
In general, for a report with ∼150 passages, information extraction takes approximately
25 minutes, while entity linking and evidence retrieval account for 2 minutes.

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
