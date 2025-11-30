import json
import time
import re
import sys
import os
import argparse
from tqdm import tqdm

# --- 1. Import Dependencies ---

# Import GRBench KG Interface
try:
    from grbench_func import GRBenchKG, SBERTRetriever
except ImportError:
    print("Error: Could not find 'grbench_func.py' or classes within it.")
    print("Please ensure grbench_func.py (V5 version) is in the same directory.")
    exit(1)

# Import OpenAI library
try:
    import openai
except ImportError:
    print("Error: Could not find 'openai' library. Please 'pip install openai'")
    exit(1)


REASONING_PROMPT = """
You are a helpful assistant reasoning on a knowledge graph to answer a question.
You must iteratively explore the graph to find the answer.
You have already been provided with an initial entity linked from the question.
Do not hallucinate. Your plan should be based on the knowledge gained so far.

{tools_description}

## Desired JSON Output Format
You MUST respond with a single JSON object enclosed in ```json ... ``` tags.
The JSON object must have a 'plan' key and a 'command' key.
The 'command' key's value must be *another JSON object* containing the command 'name' and its parameters.

Example of a response for getting neighbors:
{{
    "plan": "I have the entity ID 'Q123'. Now I need to find its 'brand' neighbors.",
    "command": {{
        "name": "get_neighbors_by_relation",
        "node_id": "Q123",
        "relation_type": "brand"
    }}
}}

Example of a response for finishing:
{{
    "plan": "I have found the answer. The brand is 'ExampleBrand'.",
    "command": {{
        "name": "finish",
        "answer": "ExampleBrand"
    }}
}}
Current Task
Question: {question} {knowledge}

Generate the JSON response for the next step based on the provided knowledge. """


TOOLS_DESCRIPTION = """
Available graph query commands:
- get_neighbors_by_relation(node_id: str, relation_type: str): Get neighbor IDs for a relation.
- get_node_features(node_id: str): Get features (like title, description) of a node.
- get_all_relations(node_id: str): List all available relations for a node.
- finish(answer: str): Provide the final answer when you have found it.
"""


# --- (B) LLM Response Function (using OpenAI API) ---

def get_my_llm_response(prompt: str, api_key: str, model_name: str) -> str:
    """
    Call LLM using OpenAI API.
    
    :param prompt: Complete prompt sent to LLM.
    :param api_key: User OpenAI API key (from --api_key).
    :param model_name: Model to use (from --model_name).
    :return: Raw text response from LLM (containing ```json ... ``` block).
    """
    try:
        # Initialize client with provided key on every call
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates JSON commands in the specified format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            seed=42,
            response_format={"type": "json_object"} # [!! New !!] Try to force JSON mode
        )
        return response.choices[0].message.content

    except openai.AuthenticationError as e:
        print(f"\n[!! CRITICAL ERROR !!] OpenAI API Key invalid or expired: {e}")
        print("Please check your --api_key argument.")
        return f"{{'plan': 'Error: OpenAI AuthenticationError', 'command': {{'name': 'error', 'error_message': '{str(e)}'}}}}"
    except openai.RateLimitError as e:
        print(f"\n[!! WARNING !!] OpenAI API Rate Limit Reached: {e}")
        print("Pausing for 60 seconds before retrying...")
        time.sleep(60)
        return get_my_llm_response(prompt, api_key, model_name)
    except openai.BadRequestError as e:
        print(f"\n[!! WARNING !!] OpenAI API Bad Request (BadRequestError): {e}")
        print("This might be because the model (e.g., gpt-4o-mini) does not support response_format={type: 'json_object'}.")
        print("Attempting fallback without 'response_format'...")
        # --- Fallback Logic ---
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates JSON commands in the specified format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                seed=42
            )
            return response.choices[0].message.content
        except Exception as fallback_e:
            print(f"\n[!! ERROR !!] Fallback LLM call also failed: {fallback_e}")
            return f"{{'plan': 'Error in LLM fallback call', 'command': {{'name': 'error', 'error_message': '{str(fallback_e)}'}}}}"
    except Exception as e:
        print(f"\n[!! ERROR !!] Error calling get_my_llm_response: {e}")
        return f"{{'plan': 'Error in LLM call', 'command': {{'name': 'error', 'error_message': '{str(e)}'}}}}"

# --- (C) Helper Functions ---

def parse_llm_output(llm_text):
    """
    Extract and parse JSON object from LLM returned text.
    """
    # Try direct parse first, as "json_object" mode might return pure JSON without ```
    try:
        return json.loads(llm_text)
    except json.JSONDecodeError:
        # If failed, fallback to regex search
        match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_text)
        
        if match:
            json_str = match.group(1).strip()
        else:
            # If still failed, fallback to finding first and last curly braces
            start_idx = llm_text.find('{')
            end_idx = llm_text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = llm_text[start_idx:end_idx+1]
            else:
                json_str = llm_text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"\n[!! ERROR !!] JSON parsing failed: {e}")
        print(f"Original LLM Output: {llm_text}")
        # Ensure error matches format
        return {"plan": "Error: Failed to parse LLM output.", "command": {"name": "error", "error_message": llm_text}}

def load_qa_data(qa_file_path):
    """Load GRBENCH QA File (one JSON object per line)"""
    qa_pairs = []
    try:
        with open(qa_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                qa_pairs.append(json.loads(line))
        return qa_pairs
    except FileNotFoundError:
        print(f"Error: QA file not found: {qa_file_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Failed to parse QA JSONL file: {qa_file_path}")
        raise

# --- (D) Core ToG 2.0 Reasoning Loop ---

def process_question(question_data, kg, retriever, api_key, model_name, max_steps):
    """
    Run ToG 2.0 mixed reasoning loop on GRBENCH KG.
    """
    question = question_data['question']
    ground_truth = question_data['answer']
    qid = question_data['qid']
    
    full_log = {
        "qid": qid,
        "question": question,
        "ground_truth": ground_truth,
        "steps": [],
        "final_answer": None,
        "correct": False
    }
    
    # --- Step 0: Initial Entity Linking ---
    print(f"  [QID: {qid} | Step: 0: SBERT Entity Linking]")
    try:
        # SBERT Retriever returns (node_id, title)
        initial_node_id, initial_title = retriever.search_entity(question, k=1)
        print(f"    - SBERT found: '{initial_title}' (ID: {initial_node_id})")
        
        # Set initial knowledge for LLM's first step
        knowledge_log = f"Knowledge Gained So Far:\n- Initial entity linking (SBERT) found a potential match for the question:\n  - Node ID: '{initial_node_id}'\n  - Node Title: '{initial_title}'"
        
        full_log["steps"].append({
            "step": 0,
            "command": "SBERT_search",
            "query": question,
            "result": {"node_id": initial_node_id, "title": initial_title}
        })

    except Exception as e:
        print(f"    - [!! ERROR !!] SBERT Entity Linking failed: {e}")
        knowledge_log = "Knowledge Gained So Far:\n- Error: Initial SBERT entity linking failed."
        full_log["steps"].append({"step": 0, "command": "SBERT_search", "error": str(e)})
        # Even if SBERT fails, let LLM try (though it has no search tools, maybe question is simple)
        pass

    step = 1
    
    while step <= max_steps:
        print(f"  [QID: {qid} | Step: {step}/{max_steps}] (LLM Reasoning)")

        # 1. Prepare Prompt (Using V5 Prompt)
        prompt = REASONING_PROMPT.format(
            question=question,
            tools_description=TOOLS_DESCRIPTION,
            knowledge=knowledge_log
        )

        # If we have reached the maximum steps, force the LLM to answer.
        if step == max_steps:
            print(f"    - [WARNING] Max steps reached. Forcing final answer.")
            prompt += "\n\n[SYSTEM WARNING]: This is your LAST step. You have run out of steps. You MUST use the 'finish' command immediately to provide your best guess for the answer based on the current knowledge. Do not use any other commands."

        
        # 2. Get LLM Reasoning
        print(f"    - Calling OpenAI ({model_name})...")
        llm_output_text = get_my_llm_response(prompt, api_key=api_key, model_name=model_name)
        
        # 3. Parse LLM Output
        action = parse_llm_output(llm_output_text)
        
        command_obj = action.get('command', {}) 
        if not isinstance(command_obj, dict):
            command_obj = {"name": "error", "error_message": f"Command field is not a dict: {command_obj}"}
        
        command_name = command_obj.get('name')

        print(f"    - LLM Plan: {action.get('plan')}")
        print(f"    - LLM Command Name: {command_name}")

        step_log = {
            "step": step,
            "prompt": prompt,
            "llm_output_raw": llm_output_text,
            "llm_output_parsed": action,
            "execution_result": None
        }

        command = command_name
        
        # 4. Execute Command on GRBench KG
        execution_result = None
        try:
            if command == 'get_neighbors_by_relation':
                node_id = command_obj.get('node_id')
                relation = command_obj.get('relation_type')
                if node_id is None or relation is None:
                    raise ValueError("get_neighbors_by_relation: Missing 'node_id' or 'relation_type'")
                execution_result = kg.get_neighbors_by_relation(node_id, relation)

            elif command == 'get_node_features':
                node_id = command_obj.get('node_id')
                if node_id is None:
                    raise ValueError("get_node_features: Missing 'node_id'")
                execution_result = kg.get_node_features(node_id)
            
            elif command == 'get_all_relations':
                node_id = command_obj.get('node_id')
                if node_id is None:
                    raise ValueError("get_all_relations: Missing 'node_id'")
                execution_result = kg.get_all_relations(node_id)

            elif command == 'finish':
                final_answer = command_obj.get('answer')
                print(f"    - [QID {qid} FINISHED]")
                print(f"    - Final Answer: {final_answer}")
                print(f"    - Ground Truth: {ground_truth}")
                is_correct = (str(final_answer).strip().lower() == str(ground_truth).strip().lower())
                print(f"    - Correct: {is_correct}")
                step_log["execution_result"] = "finish"
                full_log["steps"].append(step_log)
                full_log["final_answer"] = final_answer
                full_log["correct"] = is_correct
                return full_log

            elif command == 'error':
                execution_result = f"LLM output error: {command_obj.get('error_message')}"

            else:
                execution_result = f"Error: Unknown command name '{command}'"

            print(f"    - Execution Result: {execution_result}")
            step_log["execution_result"] = execution_result
            full_log["steps"].append(step_log)

            # 5. Update Knowledge
            knowledge_log += f"\n- Step {step} Command: {json.dumps(command_obj)}"
            knowledge_log += f"\n- Step {step} Result: {json.dumps(execution_result)}"
            
            step += 1

        except Exception as e:
            print(f"    - [!! ERROR !!] Error executing command: {e}")
            import traceback
            traceback.print_exc()
            execution_result = f"Execution Error: {str(e)}"
            step_log["execution_result"] = execution_result
            full_log["steps"].append(step_log)
            knowledge_log += f"\n- Step {step} Error: {str(e)}"
            step += 1
            
    if step > max_steps:
        # If we exit the loop and still haven't returned (meaning the LLM ignored the forced finish)
        print(f"    - [QID {qid} FAILED: Max steps exceeded and no answer provided]")
        full_log["final_answer"] = "Error: Max steps reached without answer"
        full_log["correct"] = False
        full_log["steps"].append({"step": step, "error": "Max steps reached without answer"})
        return full_log

# --- (E) Main Entry Point --

def main():
    parser = argparse.ArgumentParser(description="Run ToG 2.0 Experiments on GRBENCH")
    
    # --- File Paths ---
    parser.add_argument('--graph_file', type=str, default="/shared/data3/xx19/GraphAgent/data/processed_data/amazon/graph.json", help="Full path to GRBENCH Knowledge Graph (graph.json).")
    parser.add_argument('--qa_file', type=str, default="/shared/data3/xx19/GraphAgent/data/processed_data/amazon/data.json", help="Full path to GRBENCH QA dataset (qa_data.jsonl).")
    parser.add_argument('--output_file', type=str, default="tog2_grbench_amazon_results.jsonl", help="Path for output .jsonl log file.")
    
    # --- [!! New !!] SBERT Index Paths ---
    parser.add_argument('--faiss_index', type=str, default="grbench.faiss", help="Path to precomputed FAISS index (generated by build_sbert_index.py).")
    parser.add_argument('--id_map', type=str, default="grbench_id_map.pkl", help="Path to precomputed ID-Title map (generated by build_sbert_index.py).")

    # --- OpenAI API Parameters ---
    parser.add_argument('--api_key', type=str, default="", help="Your OpenAI API Key (e.g., 'sk-...').")
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini", help="OpenAI model name to use.")
    
    # --- Experiment Control Parameters ---
    parser.add_argument('--max_steps', type=int, default=10, help="Maximum allowed LLM reasoning steps per question.")
    parser.add_argument('--run_first_n', type=int, default=None, help="(Optional) Only run the first N questions for quick testing.")

    args = parser.parse_args()

    # 2. Run Experiment
    try:
        # 1. Initialize KG and Retriever
        print(f"--- 1. Loading GRBench KG... ---")
        kg = GRBenchKG(args.graph_file)
        
        print(f"--- 2. Loading SBERT Retriever... ---")
        retriever = SBERTRetriever(index_path=args.faiss_index, map_path=args.id_map)

        print(f"--- 3. Loading GRBench QA Data... ---")
        qa_data = load_qa_data(args.qa_file)
        
        if args.run_first_n:
            qa_data = qa_data[:args.run_first_n]
            print(f"--- [!! NOTICE !!] Configured to run only the first {args.run_first_n} questions ---")

        # 4. Run Full Experiment Loop
        print(f"--- 4. Experiment Start ---")
        print(f"Total questions: {len(qa_data)}")
        print(f"Using model: {args.model_name}")
        print(f"Results will be saved to: {args.output_file}")
        
        total_correct = 0
        
        with open(args.output_file, 'w', encoding='utf-8') as f_out:
            for qa_item in tqdm(qa_data, desc="Processing QA"):
                
                result_log = process_question(qa_item, 
                                            kg,
                                            retriever, 
                                            api_key=args.api_key,
                                            model_name=args.model_name,
                                            max_steps=args.max_steps)
                
                f_out.write(json.dumps(result_log) + '\n')
                
                if result_log["correct"]:
                    total_correct += 1

        print(f"--- 5. Experiment Finished ---")
        accuracy = (total_correct / len(qa_data)) * 100
        print(f"Total Accuracy: {total_correct} / {len(qa_data)} = {accuracy:.2f}%")
        print(f"Detailed logs saved to: {args.output_file}")

    except FileNotFoundError as e:
        print(f"\n!!! CRITICAL ERROR: File not found. !!!")
        print(e)
        print("Please check your --graph_file, --qa_file, --faiss_index, --id_map path arguments.")
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
