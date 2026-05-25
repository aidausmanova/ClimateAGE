# Taxonomy-Populated Graph for Evidence Retrieval

### Abstract
Detecting greenwashing requires locating specific evidence scattered across corporate sustainability reports (CSRs) that span hundreds of pages of technically and jargon dense text. 
This is fundamentally an evidence retrieval problem. Yet standard embedding-based retrieval and Retrieval-Augmented Generation (RAG) fail to resolve the terminology gaps, boilerplate phrasing, and cross-paragraph dependencies that are common for regulatory disclosure documents.
We present a taxonomy-populated graph-based retrieval method that embeds expert-curated disclosure taxonomy concepts directly as semantic anchors in a unified knowledge graph, bridging the gap between auditor query language and corporate disclosure language. 
Evaluated on two sustainability reporting benchmarks, ClimRetrieve and SustainableQA, our method achieves 29.6% and 24.7% relative improvement in Recall@5 over the respective retrieval baselines. 
Ablation confirms that taxonomy grounding provides consistent gains for semantically ambiguous audit queries, which are most relevant to greenwashing detection.

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
