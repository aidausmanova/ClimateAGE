import os
import json
import uuid
import logging
import networkx as nx
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from tqdm import tqdm
# from igraph import Graph
# import igraph as ig

from ..embedding_model.NVEmbedV2 import NVEmbedV2EmbeddingModel
from ..utils.embedding_store import EmbeddingStore, retrieve_knn
from ..utils.basic_utils import *
from ..utils.consts import *


logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    def __init__(self, report, synonym_sim_threshold: float = 0.8):
        self.report_name = report
        self.synonym_sim_threshold = synonym_sim_threshold
        self.embedding_model = NVEmbedV2EmbeddingModel(batch_size=8) #, precomputed_embeddings_path="data/ifrs_enriched_Llama70B_NVEmbedV2")
        # self.embedding_store = EmbeddingStore()
        self.working_dir = os.path.join(PATH['KG'], f"{self.report_name}")

        self.entity_embeddings = {}
        self.entities = []
        self.paragraph_uuids = {}
        self.entity_uuids = {}
        self.concept2concept_relations = 0
        self.entity2concept_relations = 0
        self.entity2paragraph_relations = 0
        self.synonym_relations = 0


        self.graph = nx.DiGraph()

        self.paragraph_embedding_store = EmbeddingStore(self.embedding_model,
                                                        os.path.join(self.working_dir, "paragraph_embeddings"),
                                                        self.embedding_model.batch_size, 'paragraph')
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.embedding_model.batch_size, 'entity')
        self.taxonomy_embedding_store = EmbeddingStore(self.embedding_model,
                                                       os.path.join(self.working_dir, "taxonomy_embeddings"),
                                                       self.embedding_model.batch_size, 'taxonomy')
        
        print(f"[GRAPH] Graph inititialized for report {self.report_name}")
        # self.rerank_filter = DSPyFilter(self)

        self.build_graph()


    # def initialize_graph(self):
    #     """
    #     Initializes a graph using a GraphML file if available or creates a new graph.

    #     The function attempts to load a pre-existing graph stored in a GraphML file. If the file
    #     is not present or the graph needs to be created from scratch, it initializes a new directed
    #     or undirected graph based on the global configuration. If the graph is loaded successfully
    #     from the file, pertinent GRAPHrmation about the graph (number of nodes and edges) is logged.

    #     Returns:
    #         ig.Graph: A pre-loaded or newly initialized graph.

    #     Raises:
    #         None
    #     """
    #     self._graphml_xml_file = os.path.join(
    #         self.working_dir, f"graph.graphml"
    #     )

    #     preloaded_graph = None

    #     if not self.global_config.force_index_from_scratch:
    #         if os.path.exists(self._graphml_xml_file):
    #             preloaded_graph = ig.Graph.Read_GraphML(self._graphml_xml_file)

    #     if preloaded_graph is None:
    #         return ig.Graph(directed=self.global_config.is_directed_graph)
    #     else:
    #         logger.GRAPH(
    #             f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
    #         )
    #         return preloaded_graph
        
    # def index(self, docs: List[str]):
    #     """
    #     Indexes the given documents based on the HippoRAG 2 framework which generates an OpenIE knowledge graph
    #     based on the given documents and encodes passages, entities and facts separately for later retrieval.

    #     Parameters:
    #         docs : List[str]
    #             A list of documents to be indexed.
    #     """
    #     # with open(f"gri_matches/{self.global_config.dataset}.json", "r") as file:
    #     #     gri_passage_matches = json.load(file)

    #     # updated_docs = []
    #     # for passage in docs:
    #     #     for match in gri_passage_matches:
    #     #         if passage == match['passage']:
    #     #             updated_docs.append(passage + "\n" + match['gri'])

    #     logger.GRAPH(f"Indexing Documents")

    #     logger.GRAPH(f"Performing OpenIE")

    #     if self.global_config.openie_mode == 'offline':
    #         self.pre_openie(docs)

    #     logger.GRAPH(f"Encoding Passages")
    #     self.chunk_embedding_store.insert_strings(docs)
    #     chunks = self.chunk_embedding_store.get_text_for_all_rows()

    #     all_openie_GRAPH, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
    #     new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

    #     if len(chunk_keys_to_process) > 0:
    #         new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
    #         self.merge_openie_results(all_openie_GRAPH, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

    #     if self.global_config.save_openie:
    #         self.save_openie_results(all_openie_GRAPH)

    #     ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_GRAPH)

    #     assert len(chunks) == len(ner_results_dict) == len(triple_results_dict)

    #     # prepare data_store
    #     chunk_ids = list(chunks.keys())

    #     chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
    #     entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
    #     facts = flatten_facts(chunk_triples)

    #     logger.GRAPH(f"Encoding Entities")
    #     self.entity_embedding_store.insert_strings(entity_nodes)

    #     logger.GRAPH(f"Encoding Facts")
    #     self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

    #     ######################### ADD GRI NODES ################################
    #     logger.GRAPH(f"Encoding GRI Indicators")
    #     self.gri_embedding_store.insert_strings(gri_texts)

    #     logger.GRAPH(f"Constructing Graph")

    #     self.node_to_node_stats = {}
    #     self.ent_node_to_num_chunk = {}

    #     ######################### LINK PASSAGE NODES to GRI NODES ################################
    #     gri_indicators = self.gri_embedding_store.get_text_for_all_rows()
    #     num_new_indicators = self.add_indicator_edges(gri_indicators, docs, chunks)

    #     self.add_fact_edges(chunk_ids, chunk_triples)
    #     num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

    #     if num_new_chunks > 0 or num_new_indicators > 0:
    #         logger.GRAPH(f"Found {num_new_chunks} new chunks to save into graph.")
    #         logger.GRAPH(f"Found {num_new_indicators} new indicators to save into graph.")
    #         self.add_synonymy_edges()

    #         self.augment_graph()
    #         self.save_igraph()
    
    def load_taxonomy_concepts(self):
        with open("data/ifrs_taxonomy_enriched-Llama70B.json", "r") as f:
            taxonomy_data = json.load(f)
        # taxonomy_texts = [(concept_uuid, f"Label: {concept['prefLabel']}\nDefinition:{concept['enriched_definition']}\nRelated terms: {concept['relatedTerms']}") for concept in taxonomy_data.items()]
        # self.taxonomy_embedding_store.insert_strings(taxonomy_texts)

        print("[GRAPH] Loading taxonomy concepts into graph ...")
        for concept_uuid, concept_data in taxonomy_data.items():
            self.graph.add_node(
                f"concept_{concept_uuid}",
                node_type='concept',
                uuid=concept_uuid,
                label=concept_data['prefLabel'],
                definition=concept_data['enriched_definition'],
                tags=concept_data['tags'],
                relatedTerms=concept_data['relatedTerms'],
                path_label=concept_data['path_label'],
                path_id=concept_data['path_id']
            )

            path_ids = concept_data['path_id']
            if path_ids:
                # First item in path_id is the immediate parent
                parent_uuid = path_ids[0]
                self.graph.add_edge(
                    f"concept_{concept_uuid}",
                    f"concept_{parent_uuid}",
                    weight = 1.0, 
                    edge_type = "hierarchical", 
                    relationship = "child_of"
                )
                self.concept2concept_relations += 1
        print(f"[GRAPH] Loaded {len(taxonomy_data)} concept nodes into graph, with {self.concept2concept_relations} relations")

    def load_paragraphs(self):
        print("[GRAPH] Loading paragraph chunks into embedding store ...")
        with open(PATH['weakly_supervised']['path']+self.report_name+"/corpus.json", "r") as f:
            paragraphs_data = json.load(f)
        
        paragraph_texts = []
        for paragraph in paragraphs_data:
            paragraph_uuid = str(uuid.uuid4())
            self.paragraph_uuids[paragraph['idx']] = paragraph_uuid
            paragraph_texts.append((paragraph_uuid, f"{paragraph['title']}\n{paragraph['text']}"))
        # paragraph_texts = [f"{chunk['title']}\n{chunk['text']}" for chunk in paragraphs_data]
        self.paragraph_embedding_store.insert_strings(paragraph_texts)

        print("[GRAPH] Loading paragraph chunks into graph ...")
        for paragraph in paragraphs_data:
            uid = self.paragraph_uuids[paragraph['idx']]
            paragraph_id = f"paragraph_{uid}"
            self.graph.add_node(
                    paragraph_id,
                    node_type='paragraph',
                    title=paragraph['title'],
                    text=paragraph['text'],
                    uuid=uid,
                    idx=paragraph['idx']
                )

        print(f"[GRAPH] Loaded {len(self.paragraph_uuids)} paragraph nodes into graph")

    def load_entities(self):
        print("[GRAPH] Loading extracted entities into embedding store ...")
        with open(PATH['RAG']['post_retrieved']+self.report_name+".json", "r") as f:
            entities_data = json.load(f)
        entity_texts = []
        for paragraph_id, entity_list in entities_data.items():
            for entity_data in entity_list:
                entity_uuid = str(uuid.uuid4())
                self.entity_uuids[entity_uuid] = entity_data['name']
                entity_data['entity_uuid'] = entity_uuid
                entity_texts.append((entity_uuid, f"{entity_data['name']}\n{entity_data['description']}"))
        # entity_texts = [f"{entity['name']}\n{entity['description']}" for entity in entities_data.values()]
        self.entity_embedding_store.insert_strings(entity_texts)
        
        print("[GRAPH] Loading extracted entities into graph ...")
        for paragraph_id, entity_list in entities_data.items():
            for entity_data in entity_list:
                entity = {
                    "uuid": entity_data['entity_uuid'],
                    "name": entity_data['name'],
                    "label": entity_data['label'],
                    "description": entity_data['description']
                }
                self.entities.append(entity)
                
                self.graph.add_node(
                    f"entity_{entity['uuid']}",
                    node_type='entity',
                    name=entity['name'],
                    label=entity['label'],
                    description=entity['description'],
                    uuid=entity['uuid'],
                    paragraph_idx=paragraph_id,
                    paragraph_uuid=self.paragraph_uuids[paragraph_id]
                )
                
                # Add edge between entity and pargaraph
                self.add_node_to_paragraph_edge(entity['uuid'], self.paragraph_uuids[paragraph_id])
                self.entity2paragraph_relations += 1
                # Add edge between entity and taxonomy concept
                self.add_node_to_concept_edge(entity['uuid'], entity_data['taxonomy_uuid'], entity_data['score'])
                self.entity2concept_relations += 1

        print(f"[GRAPH] Loaded {len(self.entities)} entity nodes into graph")

    def detect_synonyms(self) -> None:
        print("[GRAPH] Detecting synonym entities...")
        
        similar_pairs = self.embedding_store.compute_all_similarities(
            threshold=self.similarity_threshold
        )
        
        # Add synonym edges (bidirectional)
        for entity_id1, entity_id2, similarity in similar_pairs:
            self.graph.add_edge(
                entity_id1,
                entity_id2,
                edge_type='synonym',
                similarity=similarity
            )
            self.graph.add_edge(
                entity_id2,
                entity_id1,
                edge_type='synonym',
                similarity=similarity
            )
        
        print(f"[GRAPH] Added {len(similar_pairs)} synonym pairs")

    def connect_synonyms(self):
        self.entity_id_to_row = self.entity_embedding_store.get_text_for_all_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.GRAPH(f"Performing KNN retrieval for each entity nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=2047,
                                                    query_batch_size=1000,
                                                    key_batch_size=10000)
        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for curr_node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []
            entity = self.entity_id_to_row[curr_node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                synonym_nodes = query_node_key2knn_node_keys[curr_node_key]

                num_nns = 0
                for synonym_node_key, synonym_node_score in zip(synonym_nodes[0], synonym_nodes[1]):
                    if synonym_node_score < self.synonym_sim_threshold or num_nns > 100:
                        break

                    synonym_entity = self.entity_id_to_row[synonym_node_key]["content"]

                    if synonym_node_key != curr_node_key and synonym_entity != '':
                        # sim_edge = (node_key, nn)
                        synonyms.append((synonym_node_key, synonym_node_score))
                        num_synonym_triple += 1
                        num_nns += 1

            synonym_candidates.append((curr_node_key, synonyms))

        # Add synonym edges to graph
        new_syn_edges = 0
        for curr_node_key, syn_nodes in synonym_candidates:
            for curr_syn_node_key, curr_syn_score in syn_nodes:
                self.graph.add_edge(
                    f"entity_{curr_node_key}"
                    f"entity_{curr_syn_node_key}",
                    weight = curr_syn_score, 
                    edge_type = "synonym", 
                    relationship = "similar_to"
                )
                # Add synonymity edges both ways
                self.graph.add_edge(
                    f"entity_{curr_syn_node_key}",
                    f"entity_{curr_node_key}",
                    weight = curr_syn_score, 
                    edge_type = "synonym", 
                    relationship = "similar_to"
                )
                new_syn_edges += 2
        self.synonym_relations += new_syn_edges
        print(f"[GRAPH] Added {new_syn_edges} new synonym edges")

    def build_graph(self):
        """
        Build the complete knowledge graph
        
        Args:
            entities_data: Dictionary mapping paragraph IDs to entity lists
            taxonomy_data: Dictionary mapping UUIDs to concept data
            
        Returns:
            NetworkX directed graph
        """
        print("[GRAPH] Start loading embedding store and graph nodes ...")
        # Load data into embedding stor and create nodes
        self.load_taxonomy_concepts()
        self.load_paragraphs()
        self.load_entities()

        print("[GRAPH] All nodes loaded. Start synonymity detection ...")
        
        # Compute embeddings and detect synonyms
        self.connect_synonyms()

        self.save_graph(self.working_dir)
        print["[GRAPH] Graph construction completed!"]
        print(self.get_graph_info())
        
        # return self.graph
    
    def save_graph(self, output_path: str, format: str = 'graphml') -> None:
        """
        Save graph to file
        
        Args:
            output_path: Output file path
            format: Format ('graphml', 'gexf', 'json', 'pickle')
        """
        print(f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges")
        graph_output_path = os.path.join(self.working_dir, f"graph.graphml")
        if format == 'graphml':
            nx.write_graphml(self.graph, graph_output_path)
        elif format == 'gexf':
            nx.write_gexf(self.graph, output_path)
        elif format == 'json':
            from networkx.readwrite import json_graph
            graph_data = json_graph.node_link_data(self.graph)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2)
        elif format == 'pickle':
            nx.write_gpickle(self.graph, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Graph saved to {output_path}")
    
    def get_graph_info(self):
        graph_GRAPH = {}
        entity_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_GRAPH["num_entity_nodes"] = len(set(entity_nodes_keys))
        paragraph_nodes_keys = self.paragraph_embedding_store.get_all_ids()
        graph_GRAPH["num_paragraph_nodes"] = len(set(paragraph_nodes_keys))
        concept_nodes_keys = self.taxonomy_embedding_store.get_all_ids()
        graph_GRAPH["num_concept_nodes"] = len(set(concept_nodes_keys))
        graph_GRAPH["num_total_nodes"] = graph_GRAPH["num_entity_nodes"] + graph_GRAPH["num_paragraph_nodes"] + graph_GRAPH["num_concept_nodes"]

        graph_GRAPH['nun_hierarchical_edges'] = self.concept2concept_relations
        graph_GRAPH['nun_paragraph_mention_edges'] = self.entity2paragraph_relations
        graph_GRAPH['num_entity_link_edges'] = self.entity2concept_relations
        graph_GRAPH['num_synonym_edges'] = self.synonym_relations
        graph_GRAPH['num_total_edges'] = self.concept2concept_relations + self.entity2paragraph_relations + self.entity2concept_relations + self.synonym_relations
        return graph_GRAPH

    def add_node_to_paragraph_edge(self, entity_uuid, paragraph_uuid, relation="extracted_from"):
        """
        Function to add edge between entity and its paragraph.
        """
        self.graph.add_edge(
                    f"entity_{entity_uuid}",
                    f"paragraph_{paragraph_uuid}",
                    weight = 1.0, 
                    edge_type = "mention", 
                    relationship = relation
                )

    def add_node_to_concept_edge(self, entity_uuid, concept_uuid, score=None, relation="links_to"):
        """
        Function to add edge between entity and taxonomy concept.
        """
        self.graph.add_edge(
                    f"entity_{entity_uuid}",
                    f"concept_{concept_uuid}",
                    weight = score, 
                    edge_type = "link", 
                    relationship = relation
                )