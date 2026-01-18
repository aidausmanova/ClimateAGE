# Taxonomy-Guided Graph Construction for Evidence Retrieval

### Abstract
Corporate sustainability reports (CSRs) are essential for accountability, enabling regulators, investors and NGOs to verify company claims and identify greenwashing. However, retrieving specific evidences from lengthy and jargon-dense texts is a challenging task. Standard embedding-based retrieval and RAG approaches struggle when dealing with corporate documents due to the domain terminology and evidences scattered across multiple paragraphs.

We propose a multi-stage retrieval pipeline that represents CSRs as knowledge graphs (KGs) and incorporates domain expert knowledge from taxonomy. By extracting named entities and their relationships the graph is capable to capture complex multi-hop relations, coupled with infused domain knowledge it provides explicit semantic anchors linking entities to CSR disclosure standards. Our approach achieves 25.7% relative improvement in recall over SOTA dense retrievers on analyst-generated queries, demonstrating that explicit domain structure significantly enhances evidence retrieval for accountability and verification tasks.

---------
### Usage:
1. Clone repository
2. Environment setup
   ```bash
   conda create --name venv python=3.11
   conda activate venv
   pip install -r requirements.txt
   ```
3. The `data/` folder contains sustainability reports. Original data can be downloaded from [ClimRetrieve](https://github.com/tobischimanski/ClimRetrieve/blob/main/Report-Level%20Dataset/ClimRetrieve_ReportLevel_V1.csv) and [SustainableQA](https://github.com/DataScienceUIBK/SustainableQA/tree/main/Data) repositories.
4. Create `outputs/` and `logs/` folders
5. Running
   
   ```bash
   bash run.sh
   ```
