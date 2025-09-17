import argparse
import json
import os
import time
import openai # LEGACY: Import the module directly
from tqdm import tqdm
import jsonlines

# --- CORRECTED PROMPT TEMPLATE ---
# All literal curly braces {{}} are now correctly escaped.
JUDGE_PROMPT_TEMPLATE = """
You are an impartial and strict judge. Your task is to determine if the 'Model's Answer' is factually correct based on the 'Ground Truth Answer'.
Focus only on factual accuracy. Ignore differences in phrasing, style, or extra conversational text.

- If the model's answer contains the core information from the ground truth answer without adding any incorrect facts, it is 'Correct'.
- If the model's answer contradicts the ground truth, omits the core information, or claims it cannot answer when a ground truth exists, it is 'Incorrect'.

Please provide your judgment in a JSON format with two keys: "decision" (either "Correct" or "Incorrect") and "reason" (a brief explanation).

---
[Example 1]
Question: "Could you specify the brand of Dolica DC-BP511 1400mAh Canon Battery?"
Ground Truth Answer: "Dolica"
Model's Answer: "{{Yes}}. Based on the given knowledge triplets, the brand of the Dolica DC-BP511 1400mAh Canon Battery is {{Dolica}}."

Your Judgment:
{{
  "decision": "Correct",
  "reason": "The model correctly identified the brand as 'Dolica', which matches the ground truth."
}}

---
[Example 2]
Question: "How many items are in brand S-TECH?"
Ground Truth Answer: "1"
Model's Answer: "Based on the given knowledge triplets, the brand S-TECH has one item, which is the Slim Modern DESIGNER WATCH Retro Rotating Discs BLACK Jump Hour."

Your Judgment:
{{
  "decision": "Correct",
  "reason": "The model's answer correctly implies a count of 1, which matches the ground truth."
}}

---
[Example 3]
Question: "What brand does the item Sassy Developmental Bath Toy belong to?"
Ground Truth Answer: "Sassy"
Model's Answer: "I'm sorry, but based on the provided knowledge triplets, there is no information available for this item."

Your Judgment:
{{
  "decision": "Incorrect",
  "reason": "The model failed to find the answer, but a ground truth answer exists."
}}

---
[New Case]
Question: "{question}"
Ground Truth Answer: "{ground_truth}"
Model's Answer: "{model_answer}"

Your Judgment:
"""

def get_llm_judgment(question: str, ground_truth: str, model_answer: str, model: str):
    """Calls the judge LLM to get a decision using the legacy openai<1.0 syntax."""
    # This line will now work correctly
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        model_answer=model_answer
    )
    for _ in range(3): # Retry up to 3 times
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = response['choices'][0]['message']['content']

            start_index = content.find('{')
            end_index = content.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_string = content[start_index:end_index+1]
                result = json.loads(json_string)
                return result
            else:
                raise ValueError("Valid JSON object not found in the response.")

        except Exception as e:
            print(f"\nAPI call for judgment failed: {e}. Retrying...")
            time.sleep(5)
            
    return {"decision": "Error", "reason": "Failed to get a valid response from the API after 3 attempts."}

def main():
    parser = argparse.ArgumentParser(description="Evaluate ToG results using a legacy LLM-as-judge script.")
    parser.add_argument("--dataset", type=str, default="grbench", help="Choose the dataset.")
    parser.add_argument("--output_file", type=str, default="/home/hansont2/ToG/ToG/ToG_grbench.jsonl", help="The ToG output file to evaluate.")
    parser.add_argument("--ground_truth_file", type=str, default="/shared/data3/hansont2/GRbench/QA/amazon/data.json", help="The ground truth QA file.")
    parser.add_argument("--api_key", type=str, default="", help="Your OpenAI API key.")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini", help="The OpenAI model to use as the judge.")
    args = parser.parse_args()

    # --- Load Data ---
    with open(args.output_file, 'r', encoding='utf-8') as f:
        output_datas = [json.loads(line) for line in f]
    
    with open(args.ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth_list = [json.loads(line) for line in f]
    
    gt_lookup = {item['question']: item['answer'] for item in ground_truth_list}

    # --- LEGACY: Set API Key on the module ---
    openai.api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OpenAI API key not found.")

    # --- Main Evaluation Loop ---
    num_right = 0
    num_error = 0
    num_api_errors = 0

    for data in tqdm(output_datas, desc="Evaluating with LLM Judge (Legacy)"):
        question = data['question']
        model_answer = data['results']
        
        ground_truth_answer = gt_lookup.get(question)
        if not ground_truth_answer:
            print(f"\nWarning: Could not find ground truth for question: {question}")
            continue

        judgment = get_llm_judgment(question, ground_truth_answer, model_answer, args.judge_model)
        
        if judgment['decision'] == "Correct":
            num_right += 1
        elif judgment['decision'] == "Incorrect":
            num_error += 1
        else:
            num_api_errors += 1

    # --- Print Results ---
    total_judged = num_right + num_error
    print("\n--- LLM-as-Judge Evaluation Complete ---")
    print(f"Total Judged: {total_judged} / {len(output_datas)}")
    print(f"Correct: {num_right}")
    print(f"Incorrect: {num_error}")
    print(f"API Errors: {num_api_errors}")
    
    if total_judged > 0:
        accuracy = float(num_right / total_judged)
        print(f"Semantic Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()