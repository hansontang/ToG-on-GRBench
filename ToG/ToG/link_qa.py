import json
import argparse
import os
import time
from tqdm import tqdm
import jsonlines
from openai import OpenAI
import traceback # Import the traceback module

# --- SCRIPT VERSION IDENTIFIER ---
print("--- Running link_qa_final_debug.py (Version 4.0 - Final) ---")

# --- CORRECTED PROMPT TEMPLATE ---
# All literal curly braces are now doubled {{...}} to escape them for .format()
PROMPT_TEMPLATE = """
Your task is to accurately extract the core topic entity name from the user's question.
You need to understand the user's intent and return only the entity name itself, without extra words like "brand of" or "item".
Please strictly return the result in the JSON format {{"entity_name": "..."}}. If no clear entity is present in the question, return {{"entity_name": null}}.

---
[Example 1]
Question: "Could you specify the brand of Blackberry Playbook 7-Inch Tablet (64GB)?"
Answer: {{"entity_name": "Blackberry Playbook 7-Inch Tablet (64GB)"}}

---
[Example 2]
Question: "What brand does the item Sassy Developmental Bath Toy, Catch and Count Net belong to?"
Answer: {{"entity_name": "Sassy Developmental Bath Toy, Catch and Count Net"}}

---
[New Question]
Question: "{question}"
Answer:
"""

def get_entity_from_llm(client: OpenAI, question: str, model: str):
    # This function should now work correctly
    prompt = PROMPT_TEMPLATE.format(question=question)
    content_for_debugging = ""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        content_for_debugging = response.choices[0].message.content
        if not content_for_debugging:
            return None 
        result = json.loads(content_for_debugging)
        if not isinstance(result, dict):
            return None
        return result.get("entity_name")
    except Exception as e:
        print("\n--- ERROR INSIDE get_entity_from_llm ---")
        print(f"An error occurred while processing the question: '{question}'")
        print(f"Raw content received from API: {content_for_debugging}")
        print("Full Traceback:")
        traceback.print_exc()
        print("-----------------------------------------")
        return None

def link_qa_data_with_api(qa_path, entity_name_path, output_path, model, client):
    print("Step 1: Loading entity name maps for final linking...")
    try:
        with open(entity_name_path, 'r', encoding='utf-8') as f:
            id2name = json.load(f)
        name2id = {name: id for id, name in id2name.items()}
        print(f"Loaded {len(name2id)} unique entity names.")
    except FileNotFoundError:
        print(f"Error: Entity name file not found at {entity_name_path}")
        return

    print(f"Step 2: Processing QA data with model '{model}'...")
    successful_links = 0
    total_questions = 0

    try:
        with jsonlines.open(qa_path, mode='r') as reader, \
             jsonlines.open(output_path, mode='w') as writer:
            qa_items = list(reader)
            total_questions = len(qa_items)

            for item in tqdm(qa_items, desc="Linking Entities with API"):
                question = item.get('question')
                if not question:
                    writer.write(item)
                    continue

                extracted_name = get_entity_from_llm(client, question, model)
                
                if extracted_name:
                    entity_id = name2id.get(extracted_name)
                    if entity_id:
                        item['topic_entity_id'] = entity_id
                        item['topic_entity_name'] = extracted_name
                        successful_links += 1
                
                writer.write(item)
    except Exception as e:
        print("\n--- FATAL ERROR IN MAIN LOOP ---")
        print(f"An unexpected error occurred while processing the item: {item}")
        print("Full Traceback:")
        traceback.print_exc()
        print("--------------------------------")
        return

    print("\n--- Linking Complete ---")
    print(f"Successfully linked {successful_links} out of {total_questions} questions.")
    print(f"New linked data file saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link GRBench QA data with entities using OpenAI API (Final Version).")
    parser.add_argument("--qa_file", type=str, 
                        default="/shared/data3/hansont2/GRbench/QA/amazon/data.json", 
                        help="Path to the original QA data JSON Lines file.")
    parser.add_argument("--entity_name_file", type=str, 
                        default="/shared/data3/hansont2/GRbench/processed/amazon/entity_id_to_name.json", 
                        help="Path to the entity_id_to_name.json file from preprocessing.")
    parser.add_argument("--output_file", type=str, 
                        default="/shared/data3/hansont2/GRbench/QA/amazon/data_linked_api.jsonl", 
                        help="Path to save the new, API-linked QA data file.")
    parser.add_argument("--api_key", type=str, default="", 
                        help="Your OpenAI API key. Defaults to the OPENAI_API_KEY environment variable.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="The OpenAI model to use for extraction (e.g., gpt-4o-mini, gpt-3.5-turbo-0125).")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or pass it with --api_key.")
    
    client = OpenAI(api_key=api_key)
    
    start_time = time.time()
    link_qa_data_with_api(args.qa_file, args.entity_name_file, args.output_file, args.model, client)
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")