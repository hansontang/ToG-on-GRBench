# utils.py

import json
import jsonlines
import re
import time
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

def call_llm(prompt: str, llm_config: dict, temperature: float = 0.0) -> str:
    """
    A centralized, robust function for calling an OpenAI-compatible LLM API.
    Includes error handling and exponential backoff for retries.
    """
    client = OpenAI(api_key=llm_config['api_key'])
    retries = 0
    backoff = 1
    max_retries = llm_config.get('max_retries', 5)
    
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=llm_config['model_name'],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=llm_config.get('max_tokens', 512),
            )
            return response.choices[0].message.content
        except (APITimeoutError, RateLimitError, APIError) as e:
            retries += 1
            print(f"API Error: {e}. Retrying in {backoff} seconds... ({retries}/{max_retries})")
            time.sleep(backoff)
            backoff *= 2
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return f"Error: {e}"

    print("Max retries reached. LLM call failed.")
    return "Error: Max retries reached."

def save_result_to_jsonl(data: dict, output_path: str):
    """Appends a single result dictionary to a .jsonl file."""
    with jsonlines.open(output_path, mode='a') as writer:
        writer.write(data)

def parse_llm_json_output(response: str) -> dict:
    """
    Robustly parses a JSON object from an LLM's text response,
    handling markdown code blocks and surrounding text.
    """
    match = re.search(r"```json\s*([\s\S]+?)\s*```", response)
    if match:
        response = match.group(1)
    
    try:
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1:
            json_str = response[start:end+1]
            return json.loads(json_str)
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Failed to decode LLM JSON response: {response}")
        return {}

def parse_llm_list_output(response: str) -> list:
    """
    Robustly parses a list from an LLM's text response,
    handling surrounding text.
    """
    match = re.search(r"\[.*\]", response)
    if not match:
        print(f"Warning: Could not find a list in LLM response: {response}")
        return []
        
    list_str = match.group(0)
    try:
        return json.loads(list_str)
    except json.JSONDecodeError:
        print(f"Warning: Failed to decode LLM list response: {list_str}")
        return []