# Filename: build_sbert_index.py
#   One-time preprocessing script.
#   Reads graph.json, creates vector indices for all node titles using SBERT,
#   and saves the index and ID mapping to disk.

import json
import argparse
import pickle
import faiss
import torch
import gc
import sys
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Build SBERT Index")
    parser.add_argument('--graph_file', type=str, default="/shared/data3/xx19/GraphAgent/data/processed_data/amazon/graph.json", help="Path to graph.json")
    parser.add_argument('--output_index_path', type=str, default="grbench.faiss", help="Output path for FAISS index")
    parser.add_argument('--output_map_path', type=str, default="grbench_id_map.pkl", help="Output path for ID mapping")
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2', help="SBERT model name")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use, e.g., 'cpu', 'cuda:4', 'cuda:0'. Defaults to 'cpu' if conflict.")
    
    parser.add_argument('--chunk_size', type=int, default=200000, help="Chunk size (can be larger for CPU)")
    parser.add_argument('--batch_size', type=int, default=64, help="Inference batch size")
    args = parser.parse_args()

    print(f"--- 1. Configuration ---")
    print(f"Using device: {args.device}")
    print(f"Loading SBERT model: {args.model_name}")
    
    # Load model to specified device
    try:
        model = SentenceTransformer(args.model_name, device=args.device)
    except Exception as e:
        print(f"[!! ERROR !!] Failed to load model on {args.device}: {e}")
        print("Please check if the device is available or try --device cpu")
        return

    print(f"--- 2. Loading Graph: {args.graph_file} ---")
    id_title_map = []
    titles = []
    
    try:
        with open(args.graph_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        nodes_to_process = {}
        # Flatten nested structure
        if "item_nodes" in raw_data or "brand_nodes" in raw_data:
            print("Nested structure detected, flattening...")
            if "item_nodes" in raw_data and isinstance(raw_data["item_nodes"], dict):
                nodes_to_process.update(raw_data["item_nodes"])
            if "brand_nodes" in raw_data and isinstance(raw_data["brand_nodes"], dict):
                nodes_to_process.update(raw_data["brand_nodes"])
            del raw_data
            gc.collect()
        else:
            nodes_to_process = raw_data

        for node_id, data in tqdm(nodes_to_process.items(), desc="Extracting Titles"):
            if not isinstance(data, dict) or 'features' not in data:
                continue
            title = data.get('features', {}).get('title')
            if title and isinstance(title, str) and title.strip():
                titles.append(title)
                id_title_map.append((node_id, title))
        
        del nodes_to_process
        gc.collect()

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    total_titles = len(titles)
    print(f"Found {total_titles} valid titles.")
    if total_titles == 0: return

    # 3. Initialize Index
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embedding_dim}")
    index = faiss.IndexFlatL2(embedding_dim)

    # 4. Chunk Encoding
    print(f"--- 3. Chunk Encoding (Chunk Size: {args.chunk_size}) ---")
    
    for i in tqdm(range(0, total_titles, args.chunk_size), desc="Indexing"):
        chunk_titles = titles[i : i + args.chunk_size]
        
        try:
            with torch.no_grad():
                # Encode
                chunk_embeddings = model.encode(
                    chunk_titles, 
                    batch_size=args.batch_size, 
                    show_progress_bar=False, 
                    convert_to_numpy=True,
                    device=args.device # Explicitly specify device
                )
            
            if chunk_embeddings.dtype != 'float32':
                chunk_embeddings = chunk_embeddings.astype('float32')
                
            index.add(chunk_embeddings)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n[!! OOM ERROR !!] Device {args.device} out of memory.")
                torch.cuda.empty_cache()
                sys.exit(1)
            else:
                raise e
        
        # Cleanup
        del chunk_embeddings
        del chunk_titles
        if 'cuda' in args.device:
            torch.cuda.empty_cache()
        gc.collect()

    print(f"--- 4. Saving Index (Total: {index.ntotal}) ---")
    faiss.write_index(index, args.output_index_path)
    
    with open(args.output_map_path, 'wb') as f:
        pickle.dump(id_title_map, f)
        
    print(f"Done! Index saved to {args.output_index_path}")

if __name__ == "__main__":
    main()