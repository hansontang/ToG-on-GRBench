# Filename: grbench_func.py
#   - GRBenchKG class automatically flattens nested graph structures upon initialization.
#   - SBERTRetriever class remains largely unchanged.

import json
import time
import pickle
import faiss
from sentence_transformers import SentenceTransformer

class GRBenchKG:
    def __init__(self, graph_file_path):
        print(f"Loading GRBench Knowledge Graph from {graph_file_path}...")
        start_time = time.time()
        try:
            with open(graph_file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Flatten graph in memory so IDs can be queried directly
            self.graph_data = {}
            
            if "item_nodes" in raw_data or "brand_nodes" in raw_data:
                print("Nested structure detected, merging nodes into main index...")
                if "item_nodes" in raw_data and isinstance(raw_data["item_nodes"], dict):
                    self.graph_data.update(raw_data["item_nodes"])
                if "brand_nodes" in raw_data and isinstance(raw_data["brand_nodes"], dict):
                    self.graph_data.update(raw_data["brand_nodes"])
                print(f"Merge complete. Total nodes: {len(self.graph_data)}.")
                
                # Release raw data to save memory
                del raw_data 
            else:
                self.graph_data = raw_data

            print(f"Graph loaded. Time taken: {time.time() - start_time:.2f} seconds")
            
        except FileNotFoundError:
            print(f"Error: Graph file not found: {graph_file_path}")
            raise
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON file: {graph_file_path}")
            raise

    def get_node_features(self, node_id):
        node = self.graph_data.get(node_id)
        if node:
            return node.get('features', {})
        return None

    def get_neighbors_by_relation(self, node_id, relation_type):
        node = self.graph_data.get(node_id)
        if node:
            neighbors = node.get('neighbors', {})
            return neighbors.get(relation_type, [])
        return []

    def get_all_relations(self, node_id):
        node = self.graph_data.get(node_id)
        if node:
            return list(node.get('neighbors', {}).keys())
        return []


class SBERTRetriever:
    def __init__(self, index_path, map_path, model_name='all-MiniLM-L6-v2'):
        print("--- Loading SBERT Retriever ---")
        try:
            print(f"Loading FAISS index: {index_path}")
            self.index = faiss.read_index(index_path)
            
            print(f"Loading ID-Title map: {map_path}")
            with open(map_path, 'rb') as f:
                self.id_title_map = pickle.load(f)
            
            print(f"Loading SBERT model: {model_name}")
            self.model = SentenceTransformer(model_name)
            print("--- SBERT Retriever Loaded ---")

        except Exception as e:
            print(f"Error loading SBERT Retriever: {e}")
            raise

    def search_entity(self, query_text: str, k: int = 1):
        query_embedding = self.model.encode([query_text], convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy().astype('float32')
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i in indices[0]:
            if i < len(self.id_title_map):
                results.append(self.id_title_map[i])
            else:
                print(f"[Warning] Index {i} out of map range, skipping.")
            
        if k == 1 and results:
            return results[0]
        return results