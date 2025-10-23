import json
import argparse
import os
import time
import numpy as np
import torch
from tqdm import tqdm
import jsonlines
from sentence_transformers import SentenceTransformer, util
from utils import parse_llm_json_output, call_llm
from prompt_list import ENTITY_LINKING_PROMPT

def link_qa_dataset(qa_file, entity_id_to_name, sbert_model, output_file, llm_config, batch_size):
    """
    Iterates through a QA dataset, extracts entity mentions using an LLM, and links them via cosine similarity.
    This version uses a robust memory management strategy to handle very large knowledge graphs without OOM errors.
    """
    print("--- Linking QA dataset (Robust Memory Version)... ---")
    
    # 1. Pre-encode all entity names in batches, moving results to CPU immediately.
    print("--- Step 1: Encoding all entity names into vectors (memory safe)... ---")
    entity_ids = list(entity_id_to_name.keys())
    entity_names = [entity_id_to_name[eid] for eid in entity_ids]
    
    # --- FIX: Manually batch and move to CPU to prevent OOM ---
    all_entity_embeddings_cpu = []
    for i in tqdm(range(0, len(entity_names), batch_size), desc="Encoding Batches"):
        batch_names = entity_names[i:i+batch_size]
        # Encode a batch (this happens on GPU if available)
        batch_embeddings = sbert_model.encode(
            batch_names, 
            convert_to_tensor=True,
            show_progress_bar=False # Disable inner progress bar
        )
        # Immediately move the result to CPU RAM and detach from computation graph
        all_entity_embeddings_cpu.append(batch_embeddings.cpu())

    # Concatenate all CPU tensors into one large tensor in RAM
    all_entity_embeddings = torch.cat(all_entity_embeddings_cpu, dim=0)
    print(f"Encoded {len(entity_ids)} entities. Embeddings are stored in CPU RAM.")

    # 2. Link questions one by one
    print("--- Step 2: Linking questions to entities... ---")
    with jsonlines.open(qa_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        for item in tqdm(reader, desc="Linking questions"):
            question = item['question']
            
            # a. Use LLM to extract entity mention
            prompt = ENTITY_LINKING_PROMPT.format(question=question)
            raw_response = call_llm(prompt, llm_config)
            parsed_response = parse_llm_json_output(raw_response)
            entity_mention = parsed_response.get("entity_name")

            linked_entity_id = None
            if entity_mention and isinstance(entity_mention, str):
                # b. Encode the query text (on GPU for speed)
                query_embedding = sbert_model.encode(entity_mention, convert_to_tensor=True)
                
                # c. Compute cosine similarities (on CPU, as the large tensor is on CPU)
                # Move the small query embedding to CPU to match the large embedding tensor
                cosine_scores = util.cos_sim(query_embedding.cpu(), all_entity_embeddings)[0]
                
                # d. Find the index of the highest score
                best_match_index = torch.argmax(cosine_scores).item()
                
                # e. Get the corresponding entity ID
                linked_entity_id = entity_ids[best_match_index]

            item['topic_entity_id'] = linked_entity_id
            writer.write(item)
    
    print(f"\nLinked QA data saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link entities in GRBench QA data using LLM and FAISS.")
    parser.add_argument("--qa_file", type=str, default="/shared/data3/hansont2/GRbench/QA/amazon/data.json", help="Path to the input QA data file (.jsonl).")
    parser.add_argument("--entity_name_file", type=str, default="/shared/data3/hansont2/GRbench/processed/amazon/entity_id_to_name.json", help="Path to the entity_id_to_name.json file.")
    parser.add_argument("--output_dir", type=str, default="/shared/data3/hansont2/GRbench/processed/amazon", help="Directory to save the FAISS index and the linked QA file.")
    parser.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2", help="Name of the SentenceTransformer model to use.")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="Name of the LLM for entity extraction.")
    parser.add_argument("--api_key", type=str, default="", help="Your OpenAI API key.")
    
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding entity names to prevent OOM errors.")
    
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("API key must be provided via the --api_key argument or the OPENAI_API_KEY environment variable.")
    
    llm_config = {"api_key": args.api_key, "model_name": args.llm_model}
    
    print(f"Loading SentenceTransformer model: {args.sbert_model}...")
    sbert_model = SentenceTransformer(args.sbert_model)
    
    print("Loading entity names...")
    with open(args.entity_name_file, 'r', encoding='utf-8') as f:
        entity_id_to_name = json.load(f)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_qa_file = os.path.join(args.output_dir, os.path.basename(args.qa_file).replace('.jsonl', '_linked_no_faiss.jsonl'))
    
    link_qa_dataset(args.qa_file, entity_id_to_name, sbert_model, output_qa_file, llm_config, args.batch_size)
