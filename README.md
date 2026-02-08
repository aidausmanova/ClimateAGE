# Taxonomy-Guided Graph Construction for Evidence Retrieval

### Abstract
Corporate sustainability reports (CSRs) are essential for accountability, enabling regulators, investors and NGOs to verify company claims and identify greenwashing. However, retrieving specific evidences from lengthy and jargon-dense texts is a challenging task. Standard embedding-based retrieval and RAG approaches struggle when dealing with corporate documents due to the domain terminology and evidences scattered across multiple paragraphs.

We propose a multi-stage retrieval pipeline that represents CSRs as knowledge graphs (KGs) and incorporates domain expert knowledge from taxonomy. By extracting named entities and their relationships the graph is capable to capture complex multi-hop relations, coupled with infused domain knowledge it provides explicit semantic anchors linking entities to CSR disclosure standards. Our approach achieves 25.7% relative improvement in recall over SOTA dense retrievers on analyst-generated queries, demonstrating that explicit domain structure significantly enhances evidence retrieval for accountability and verification tasks.

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
