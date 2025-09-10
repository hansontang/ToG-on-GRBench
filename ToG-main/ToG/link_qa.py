import json
import argparse
import re
from tqdm import tqdm
import jsonlines
from thefuzz import process
from collections import defaultdict
import time

STOP_WORDS = {'a', 'an', 'and', 'the', 'in', 'of', 'for', 'with', 'on', 'at', 'to', 'is', 'am', 'are', 'was', 'were'}

def extract_topic_entity_from_question(question: str):
    """
    Uses regex to find the topic entity.
    """
    
    patterns = [
        r"item '(.*?)'",
        r"(?:brand of|item|with|query:)\s+((?:[\w\s'-:]+(?:\s\(.+?\))?)+?)(?:\?|\s+have|\s+cost|\s+fall|\s+classified|\s+belong to)",
        r'history:\s*\[\s*"(.*?)"'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return match.group(1).strip()
            
    return None

def build_inverted_index(name2id: dict):
    """
    Builds an inverted index from words to full entity names.
    """
    print("Building inverted index for faster searching. This may take a few minutes...")
    inverted_index = defaultdict(list)
    word_regex = re.compile(r'\w+')
    
    for name in tqdm(name2id.keys(), desc="Indexing entity names"):
        words = set(word.lower() for word in word_regex.findall(name))
        for word in words:
            if word not in STOP_WORDS and len(word) > 1:
                inverted_index[word].append(name)
    print("Inverted index built successfully.")
    return inverted_index

def get_candidates_from_index(extracted_name: str, inverted_index: dict):
    """
    Uses the inverted index to generate a small candidate set.
    """
    word_regex = re.compile(r'\w+')
    query_words = set(word.lower() for word in word_regex.findall(extracted_name))
    
    if not query_words:
        return []

    candidate_set = set()
    for word in query_words:
        if word not in STOP_WORDS and len(word) > 1:
            candidate_set.update(inverted_index.get(word, []))
    
    if not candidate_set:
        return [] 
        
    return list(candidate_set)


def link_qa_data(qa_path, entity_name_path, output_path, threshold):
    """
    Main function: reads QA data, links entities, and saves results.
    """
    print("Step 1: Loading entity name maps...")
    try:
        with open(entity_name_path, 'r', encoding='utf-8') as f:
            id2name = json.load(f)
        name2id = {name: id for id, name in id2name.items()}
        print(f"Loaded {len(name2id)} unique entity names.")
    except FileNotFoundError:
        print(f"Error: Entity name file not found at {entity_name_path}")
        return

    inverted_index = build_inverted_index(name2id)

    print("Step 2: Processing and linking QA data...")
    successful_links = 0
    total_questions = 0

    try:
        with jsonlines.open(qa_path, mode='r') as reader, \
             jsonlines.open(output_path, mode='w') as writer:
            
            qa_items = list(reader)
            total_questions = len(qa_items)

            for item in tqdm(qa_items, desc="Linking Entities"):
                question = item.get('question')
                if not question:
                    continue

                extracted_name = extract_topic_entity_from_question(question)
                
                if extracted_name:
                    candidates = get_candidates_from_index(extracted_name, inverted_index)
                    
                    if candidates:
                        best_match, score = process.extractOne(extracted_name, candidates)
                        
                        if score >= threshold:
                            entity_id = name2id[best_match]
                            item['topic_entity_id'] = entity_id
                            item['topic_entity_name'] = best_match
                            successful_links += 1
                
                writer.write(item)

    except FileNotFoundError:
        print(f"Error: QA file not found at {qa_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    print("\n--- Linking Complete ---")
    print(f"Successfully linked {successful_links} out of {total_questions} questions.")
    print(f"Confidence threshold used: {threshold}")
    print(f"New linked data file saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-link GRBench QA data with knowledge graph entities.")
    parser.add_argument("--qa_file", type=str, 
                        default="/shared/data3/hansont2/GRbench/QA/amazon/data.json", 
                        help="Path to the original QA data JSON Lines file.")
    parser.add_argument("--entity_name_file", type=str, 
                        default="/shared/data3/hansont2/GRbench/processed/amazon/entity_id_to_name.json", 
                        help="Path to the entity_id_to_name.json file from preprocessing.")
    parser.add_argument("--output_file", type=str, 
                        default="/shared/data3/hansont2/GRbench/QA/amazon/data_linked.jsonl", 
                        help="Path to save the new, linked QA data file.")
    parser.add_argument("--threshold", type=int, default=85, 
                        help="Confidence score threshold (0-100) for fuzzy matching.")
    
    args = parser.parse_args()
    
    start_time = time.time()
    link_qa_data(args.qa_file, args.entity_name_file, args.output_file, args.threshold)
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")