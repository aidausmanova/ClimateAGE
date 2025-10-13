from tqdm import tqdm
from collections import defaultdict
import numpy as np
import copy
import torch
import spacy
import os

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from src.utils.basic_utlis import *
from src.utils.consts import *
from src.embedding_model import NVEmbedV2EmbeddingModel


class RAG:
    def __init__(self, report, init_prev_retrieved=True, use_gpu=True):
        print("Initializing retriever...")
        self.use_batching = True
        self.batch_size = 32
        self.threshold = 45 #0.5
        self.use_caching = True

        self.skip_words = SKIP_WORDS
        self.skip_chars = SKIP_CHARS
        self.skip_phrases = set(LABELS_DICT["label_mapper"].keys())
        self.skip_phrases.add("dataset")

        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.embed_model = NVEmbedV2EmbeddingModel(precomputed_embeddings_path="data/ifrs_enriched_Llama70B_NVEmbedV2")

        self.spacy_pipe = spacy.load("en_core_web_sm")
        self.TAX = load_json_file(PATH["TAX"])
        self.report_name = report
        print("Loading previous retrieved...")
        if init_prev_retrieved and os.path.exists(PATH["RAG"]["prev_retrieved"]+self.report_name+".json"):
            self.RETRIEVED = load_json_file(PATH["RAG"]["prev_retrieved"]+self.report_name+".json")
        else:
            self.RETRIEVED = {}
    
    def extract_noun_phrases(self, sentence):
        doc = self.spacy_pipe(sentence)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        return noun_phrases

    def filter_noun_phrases(self, noun_phrases):
        noun_phrases2 = set()

        for p in noun_phrases:
            p = p.replace("\n", "")
            skip = False
            # skip if p contains unicode string
            if any(ord(char) > 128 for char in p):
                continue
            
            p = p.strip()
            while len(p) > 1 and (not p[0].isalnum() or not p[-1].isalnum()):
                # remove starting non-characters and non-digits
                if not p[0].isalpha() and not p[0].isdigit():
                    p = p[1:]
                # remove ending non-characters and non-digits
                elif not p[-1].isalpha() and not p[-1].isdigit():
                    p = p[:-1]
                else:
                    break
            if len(p) <= 3:
                continue

            p_lower = p.lower()
            for w in self.skip_words:
                if p_lower.startswith(w + " "):
                    p = p[len(w) + 1 :]
                    p_lower = p.lower()
                    break

            if any(c in p_lower for c in self.skip_chars):
                continue
            
            if p_lower in self.skip_words:
                continue
            
            if p_lower in ["models", "number", "study", "phase", "people", 
                            "need", "what", "which", "year"]:
                continue
            
            if len(p) > 3:
                noun_phrases2.add(p)

        return noun_phrases2

    def run(self, text):
        noun_phrases = self.extract_noun_phrases(text)
        filtered_noun_phrases = self.filter_noun_phrases(noun_phrases)

        row = {}
        pbar = tqdm(filtered_noun_phrases)
        pbar.set_description("Retrieving noun phrases")
        phrase_i = 0
        for phrase in pbar:
            phrase_i += 1
            pbar.set_description(
                f"In doc, Retrieving noun phrases - [{phrase}]: [{phrase_i}/{len(filtered_noun_phrases)}] Total Retrieved: {len(all_retrieved):,}"
            )
            if phrase in self.skip_words:
                continue
            if phrase in self.RETRIEVED:
                if self.RETRIEVED[phrase]:
                    row[phrase] = self.RETRIEVED[phrase]
            else:
                # retrieve with the retriever
                node = self.embed_model.retrieve([phrase])[0][0]

                # filter results
                if node['score'] > self.threshold:
                    uuid = node['metadata']["uuid"]
                    # label = retriever.get_label(retriever.TAX[uuid]["tags"])
                    row[phrase] = [
                        "vriable",
                        uuid,
                        self.TAX[uuid]["prefLabel"],
                        node['score'],
                    ]
                    self.RETRIEVED[phrase] = row[phrase]
                else:
                    self.RETRIEVED[phrase] = []

        return row

    def save_retrieved(self):
        new = copy.deepcopy(self.RETRIEVED)
        if os.path.exists(PATH["RAG"]["prev_retrieved"]+f"{self.report_name}.json"):
            old = load_json_file(PATH["RAG"]["prev_retrieved"]+f"{self.report_name}.json")
        else:
            old = {}

        old.update(new)
        self.RETRIEVED = old
        with open(f"outputs/retrieved/{self.report_name}.json", "w") as f:
            json.dump(self.RETRIEVED, f, indent=4)

def _process_single_doc_worker(args):
    doc, report = args
    # Each worker needs its own retriever instance
    retriever = RAG(report=report)
    output = retriever.run(doc['title'] + " " + doc['text'])
    return doc['idx'], output

def process_documents_parallel(report, docs, output_dir, num_workers=4, use_threads=True):
    if use_threads:
        # ThreadPoolExecutor - better for I/O bound tasks (embedding retrieval)
        # No pickling issues, shares memory
        results = {}
        
        # Create ONE shared retriever to avoid reloading model
        shared_retriever = retriever(report=report, init_prev_retrieved=False)
        
        def process_doc_threaded(doc):
            """Process document using shared retriever."""
            output = shared_retriever.run(
                doc['title'] + " " + doc['text'], 
                tqdm_disable=True
            )
            return doc['idx'], output
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_doc_threaded, doc) for doc in docs]
            for future in tqdm(futures, desc="Processing documents (threaded)"):
                idx, output = future.result()
                # results[idx] = output
                results.update(output)
    else:
        # ProcessPoolExecutor - better for CPU-bound tasks
        # Each process loads its own model (more memory)
        args_list = [(doc, report) for doc in docs]
        results = {}
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_single_doc_worker, args) for args in args_list]
            for future in tqdm(futures, desc="Processing documents (multiprocess)"):
                idx, output = future.result()
                # results[idx] = output
                results.update(output)
    
    return results


if __name__ == "__main__":
    output_dir = PATH["weakly_supervised"]["RAG_preprocessed"]
    doc_dir = PATH["weakly_supervised"]["text"]
    report_name = REPORTS[5]

    with open(PATH["weakly_supervised"]['path']+report_name+"/corpus.json", "r") as f:
        data = json.load(f)
    print(f"Corpus {report_name} with {len(data)} paragraphs loaded")

    retriever = RAG(report=report_name, )

    all_retrieved = {}
    for doc in tqdm(data):
        text = doc['title']+": "+doc['text']
        output = retriever.run(text)
    
    retriever.save_retrieved()

    # print("\n=== Running parallel retrieval ===")
    # report_output = process_documents_parallel(
    #     report=report_name,
    #     docs=data,
    #     output_dir=output_dir,
    #     num_workers=4, 
    #     use_threads=True
    # )
    
    # with open(f"{output_dir}/{report}.json", "w") as f:
    #     json.dump(report_output, f)
    
    # if os.path.exists(PATH["RAG"]["prev_retrieved"]+f"{report_name}.json"):
    #     old = load_json_file(PATH["RAG"]["prev_retrieved"]+f"{report_name}.json")
    # else:
    #     old = {}

    # old.update(report_output)
    # with open(f"outputs/retrieved/{report_name}.json", "w") as f:
    #     json.dump(old, f, indent=4)