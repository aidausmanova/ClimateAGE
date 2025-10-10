import json
import torch
import uuid
from typing import List, Optional

from transformers import AutoModel, AutoTokenizer
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.embeddings import BaseEmbedding

queries = [
    "Do the environmental/sustainability targets set by the company align with external climate change adaptation goals/targets?",
    "Do the environmental/sustainability targets set by the company reference external climate change adaptation goals/targets?",
    "Does the company identify any impacts of its business activities on the environment?",
    "Does the company have a strategy on waste management?",
    "Does the company report short-term actions taken or planned to reduce its waste generation?",
    "Does the company report a plan to engage with downstream partners on water consumption or water pollution?",
    "Does the company encourage downstream partners to carry out climate-related risk assessments?",
    "Does the company have a specific process in place to identify risks arising from climate change?",
    "Does the company refer to any third party scenarios when identifying climate-related risks or opportunities (e.g. IPCC trajectories, NGFS scenarios, etc.)?",
    "Does the company report how adjustments to its business operations will allow it to adapt to climate change?",
    "Does the company report the methodology used to identify the dependencies and impact of its business activities on the environment?",
    "Does the company report the climate change scenarios used to test the resilience of its business strategy?",
    "Has the company identified any synergies between its climate change adaptation goals and other business goals?",
    "Does the company seek to adjust its business model to better provide climate change adaptation products and services?",
    "Does the company provide definitions for climate change adaptation?"
    ]


def get_taxonomy_corpus(is_llama_doc=True):
    with open("data/ifrs_taxonomy_enriched-Llama70B.json", "r") as f:
        taxonomy_data = json.load(f)
    
    taxonomy_documents = []
    if is_llama_doc:
        for uid, taxonomy_metadata in taxonomy_data.items():
            related_terms = ", ".join(taxonomy_metadata['relatedTerms'])

            taxonomy_documents.append(
                Document(
                    text = f"Label: {taxonomy_metadata['prefLabel']}\n Definition: {taxonomy_metadata['enriched_definition']}\n Related terms: {related_terms}",
                    metadata = {
                        "uuid": uid,
                        "label": taxonomy_metadata['prefLabel'],
                        "tags": taxonomy_metadata['tags'],
                        "definition": taxonomy_metadata['enriched_definition'],
                        "relatedTerms": taxonomy_metadata['relatedTerms'],
                        "path_id": taxonomy_metadata['path_id'],
                        "path_label": taxonomy_metadata['path_label']
                    }
                )
            )
    else:
        for d in taxonomy_data.values():
            taxonomy_documents.append(f"Label: {d['prefLabel']}\n Definition: {d['enriched_definition']}")

    return taxonomy_documents


class NVidiaEmbedder(BaseEmbedding):
    def __init__(self, model_name="nvidia/NV-Embed-v2", device="cuda", max_length=512, add_instructions=True, normalize_embeddings=True, **kwargs):
        super().__init__(**kwargs)
        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self._device = device
        self._add_instructions = add_instructions
        self._normalize_embeddings = normalize_embeddings
        self._model.eval()

        self._passage_instruction = "Represent this climate disclosure taxonomy concept for retrieval: "
        self._query_instruction = "Represent this climate-related query for searching relevant taxonomy concepts: "


    @classmethod
    def class_name(cls) -> str:
        return "nvidia_nv_embed_v2"
    
    def _add_instruction_prefix(self, text: str, is_query: bool = False) -> str:
        """Add instruction prefix to guide the model."""
        if not self._add_instructions:
            return text
        
        instruction = self._query_instruction if is_query else self._passage_instruction
        return instruction + text
    
    def _normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """L2 normalization for better cosine similarity."""
        if not self._normalize_embeddings:
            return embeddings
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def _embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        if self._add_instructions:
            texts = [self._add_instruction_prefix(t, is_query) for t in texts]
        inputs = self._tokenizer(texts, padding=True, 
                                 truncation=True, max_length=self._max_length,
                                 return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        embeddings = outputs['sentence_embeddings'][0]
        if self._normalize_embeddings:
            embeddings = self._normalize(embeddings)

        embeddings_list = [
            [float(x) for x in emb.detach().cpu().numpy()] 
            for emb in embeddings
        ]
        return embeddings_list

    def _get_text_embedding(self, text: str):
        return self._embed([text])[0]

    def _get_query_embedding(self, query: str):
        return self._get_text_embedding(query)

    def _get_text_embeddings(self, texts):
        return self._embed(texts)
        
    async def _aget_query_embedding(self, query: str):
        return self._get_text_embedding(query)
    
    async def _aget_text_embedding(self, text):
        return self._get_text_embedding(text)


if __name__ == "__main__":
    # taxonomy_documents = get_taxonomy_corpus()
    taxonomy_documents = []
    for query in queries:
        taxonomy_documents.append(
                Document(
                    text = f"Query: {query}",
                )
            )
    print("[INFO] Taxonomy loaded")

    embed_model = NVidiaEmbedder(device="cuda")
    Settings.embed_model = embed_model

    print("[INFO] Start generating embeddings")
    index = VectorStoreIndex.from_documents(taxonomy_documents, embed_model=Settings.embed_model, show_progress=True)
    # index.storage_context.persist(persist_dir="data/ifrs_enriched_Llama70B_llmdefin_NV-Embed-v2")
    index.storage_context.persist(persist_dir="data/questions_Llama70B_llmdefin_NV-Embed-v2")
