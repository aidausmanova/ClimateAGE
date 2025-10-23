import os
import json
import uuid
import logging
import networkx as nx
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
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

@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    doc_scores: np.ndarray = None
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None


    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()[:5]]  if self.doc_scores is not None else None,
            "gold_docs": self.gold_docs,
        }

class KnowledgeGraphBuilder:
    def __init__(self, report, synonym_sim_threshold: float = 0.8):
        self.report_name = report
        self.synonym_sim_threshold = synonym_sim_threshold
        self.embedding_model = NVEmbedV2EmbeddingModel(batch_size=8) #, precomputed_embeddings_path="data/ifrs_enriched_Llama70B_NVEmbedV2")
        self.working_dir = os.path.join(PATH['KG'], f"{self.report_name}")

        self.entity_embeddings = {}
        self.entities = {}
        self.entity_uuids = {}
        self.paragraph_uuids = {}
        self.active_paragraph_ids = []
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
    
    ############################# GRAPH CONSTRUCTION #############################
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
                tags=",".join(str(element) for element in concept_data['tags']),
                # related_terms=",".join(str(element) for element in concept_data['relatedTerms']),
                # path_label=",".join(str(element) for element in concept_data['path_label']),
                # path_id=",".join(str(element) for element in concept_data['path_id'])
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
            if paragraph['idx'] in self.active_paragraph_ids:
                paragraph_uuid = str(uuid.uuid4())
                self.paragraph_uuids[paragraph['idx']] = paragraph_uuid
                paragraph_texts.append((paragraph_uuid, f"{paragraph['title']}\n{paragraph['text']}"))
        # paragraph_texts = [f"{chunk['title']}\n{chunk['text']}" for chunk in paragraphs_data]
        self.paragraph_embedding_store.insert_strings(paragraph_texts)

        ############ Need to change that to retrieve from embeddings store ############
        print("[GRAPH] Loading paragraph chunks into graph ...")
        for paragraph in paragraphs_data:
            if paragraph['idx'] in self.active_paragraph_ids:
                uid = self.paragraph_uuids[paragraph['idx']]
                paragraph_id = f"paragraph_{uid}"
                self.graph.add_node(
                        paragraph_id,
                        node_type='paragraph',
                        uuid=uid,
                        label=paragraph['idx'],
                        title=paragraph['title'],
                        text=paragraph['text']
                    )

        print(f"[GRAPH] Loaded {len(self.paragraph_uuids)} paragraph nodes into graph")

    def preprocess_entities(self):
        print("[GRAPH] Loading extracted entities into embedding store ...")
        with open(PATH['RAG']['post_retrieved']+self.report_name+".json", "r") as f:
            entities_data = json.load(f)
        entity_texts = []
        for paragraph_id, entity_list in entities_data.items():
            self.active_paragraph_ids.append(paragraph_id)
            for entity_data in entity_list:
                curr_entity_name = entity_data['name']
                # Entity disambiguation
                normalized = basic_normalize(curr_entity_name)
                normalized = expand_abbreviations(normalized)
                if entity_data['label'] == 'organization': normalized = normalize_organization_name(curr_entity_name)
                normalized_entity_name = lemmatization(normalized)

                if normalized_entity_name not in self.entity_uuids:
                    entity_uuid = str(uuid.uuid4())
                    entity = {
                            "label": normalized_entity_name,
                            "entity_type": entity_data['label'],
                            "definition": entity_data['description'],
                            "paragraph_ids": [paragraph_id],
                            "taxonomy_concepts": [(entity_data['taxonomy_uuid'], entity_data['score'])]
                        }
                    self.entities[entity_uuid] = entity
                    self.entity_uuids[normalized_entity_name] = entity_uuid
                    entity_texts.append((entity_uuid, f"{normalized_entity_name}\n{entity_data['description']}"))
                else:
                    curr_entity_uuid = self.entity_uuids[normalized_entity_name]
                    self.entities[curr_entity_uuid]['paragraph_ids'].append(paragraph_id)

                    if entity_data['taxonomy_uuid'] not in self.entities[curr_entity_uuid]['taxonomy_concepts']:
                        self.entities[curr_entity_uuid]['taxonomy_concepts'].append((entity_data['taxonomy_uuid'], entity_data['score']))
               
        # entity_texts = [f"{entity['name']}\n{entity['description']}" for entity in entities_data.values()]
        self.entity_embedding_store.insert_strings(entity_texts)

    def load_entities(self):
        print("[GRAPH] Loading extracted entities into graph ...")
        for entity_uuid, entity_data in self.entities.items():
            self.graph.add_node(
                f"entity_{entity_uuid}",
                node_type='entity',
                uuid=entity_uuid,
                label=entity_data['label'],
                entity_type=entity_data['entity_type'],
                definition=entity_data['definition'],
                paragraphs=",".join(str(element) for element in entity_data['paragraph_ids']),
                concepts=",".join(str(element[0]) for element in entity_data['taxonomy_concepts']),
            )
                    
            # Add edge between entity and pargaraph
            for paragraph_idx in entity_data['paragraph_ids']:
                self.add_node_to_paragraph_edge(entity_uuid, self.paragraph_uuids[paragraph_idx])
                self.entity2paragraph_relations += 1
            
            for taxonomy_uuid, taxonomy_sim_score in entity_data['taxonomy_concepts']:
                # Add edge between entity and taxonomy concept
                self.add_node_to_concept_edge(entity_uuid, taxonomy_uuid, taxonomy_sim_score/100)
                self.entity2concept_relations += 1
            
        print(f"[GRAPH] Loaded {len(self.entities)} entity nodes into graph")

    def connect_synonyms(self):
        self.entity_id_to_row = self.entity_embedding_store.get_text_for_all_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each entity nodes ({len(entity_node_keys)}).")

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
                    curr_node_key,
                    curr_syn_node_key,
                    weight = curr_syn_score, 
                    edge_type = "synonym", 
                    relationship = "similar_to"
                )
                # Add synonymity edges both ways
                self.graph.add_edge(
                    curr_syn_node_key,
                    curr_node_key,
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
        self.preprocess_entities()
        self.load_paragraphs()
        self.load_entities()

        print("[GRAPH] All nodes loaded. Start synonymity detection ...")
        
        # Compute embeddings and detect synonyms
        self.connect_synonyms()

        self.save_graph(self.working_dir)
        print("[GRAPH] Graph construction completed!")
        print(self.get_graph_info())
        
        # return self.graph
    
    def save_graph(self, output_path: str, format: str = 'graphml') -> None:
        """
        Save graph to file
        
        Args:
            output_path: Output file path
            format: Format ('graphml', 'gexf', 'json', 'pickle')
        """
        print(f"Writing graph with {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
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
        # entity_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_GRAPH["num_entity_nodes"] = len(self.entities)
        # paragraph_nodes_keys = self.paragraph_embedding_store.get_all_ids()
        graph_GRAPH["num_paragraph_nodes"] = len(self.paragraph_uuids)
        # concept_nodes_keys = self.taxonomy_embedding_store.get_all_ids()
        graph_GRAPH["num_concept_nodes"] = 33 #len(set(concept_nodes_keys))
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
        
    ############################# GRAPH RETRIEVAL #############################
    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("[GRAPH] Preparing for fast retrieval")
        self.query_to_embedding: Dict = {'triple': {}, 'paragraph': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # a list of phrase node keys
        self.paragraph_node_keys: List = list(self.paragraph_embedding_store.get_all_ids()) # a list of passage node keys
        self.concept_node_keys: List = list(self.taxonomy_embedding_store.get_all_ids()) # a list of passage node keys
        # self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        # assert len(self.entity_node_keys) + len(self.paragraph_node_keys) + len(self.concept_node_keys) == self.graph.vcount()

        igraph_name_to_idx = {node["uuid"]: idx for idx, node in enumerate(self.graph.nodes())} # from node key to the index in the backbone graph
        self.node_name_to_vertex_idx = igraph_name_to_idx
        self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # a list of backbone graph node index
        self.paragraph_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.paragraph_node_keys] # a list of backbone passage node index
        self.concept_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.concept_node_keys]

        self.paragraph_key_to_idx = {key: idx for key, idx in zip(self.paragraph_node_keys, self.paragraph_node_idxs)}
        self.concept_key_to_idx = {key: idx for key, idx in zip(self.concept_node_keys, self.concept_node_idxs)}

        # logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.paragraph_embeddings = np.array(self.paragraph_embedding_store.get_embeddings(self.paragraph_node_keys))
        self.concept_embeddings = np.array(self.taxonomy_embedding_store.get_embeddings(self.concept_node_keys))
        # self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['paragraph']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding