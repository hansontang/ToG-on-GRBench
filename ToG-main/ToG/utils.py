import json
import time
import openai
import re
from prompt_list import *

def retrieve_top_docs(query, docs, model, width=3):
    """
    Retrieve the topn most relevant documents for the given query.
    """
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)
    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]
    return top_docs, top_scores

def compute_bm25_similarity(query, corpus, width=3):
    """
    Computes BM25 similarity and returns the topn relations and scores.
    """
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    relations = bm25.get_top_n(tokenized_query, corpus, n=width)
    doc_scores = sorted(doc_scores, reverse=True)[:width]
    return relations, doc_scores

def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations

def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)

def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    i=0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        i+=1
    return True, relations

def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    # Set the API key for the OpenAI library
    openai.api_key = opeani_api_keys

    messages = [{"role":"system", "content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user", "content":prompt}
    messages.append(message_prompt)
    
    retries = 3
    for i in range(retries):
        try:
            response = openai.ChatCompletion.create(
                    model=engine,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0
            )
            return response["choices"][0]['message']['content']
        except Exception as e:
            print(f"OpenAI API error: {e}. Retrying in 2 seconds...")
            time.sleep(2)
    
    print("All OpenAI API retries failed.")
    return "Error: Could not get a response from the language model."
    
def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)

def del_unknown_entity(entity_candidates):
    if len(entity_candidates) == 1 and entity_candidates[0] == "UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates

def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)
    
def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name):
    data_dict = {"question": question, "results": answer, "reasoning_chains": cluster_chain_of_entities}
    with open(f"ToG_{file_name}.jsonl", "a") as outfile:
        json_str = json.dumps(data_dict)
        outfile.write(json_str + "\n")
    
def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""
    
def if_true(prompt):
    return prompt.lower().strip().replace(" ", "") == "yes"

def generate_without_explored_paths(question, args):
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return response

def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst

def prepare_dataset(dataset_name):
    # This function might need to be adjusted based on your file naming conventions
    file_map = {
        'cwq': '../data/cwq.json',
        'webqsp': '../data/WebQSP.json',
        'grailqa': '../data/grailqa.json',
        # Add other datasets here
    }
    question_map = {
        'cwq': 'question',
        'webqsp': 'RawQuestion',
        'grailqa': 'question',
        # Add other datasets here
    }
    
    if dataset_name in file_map:
        with open(file_map[dataset_name], encoding='utf-8') as f:
            datas = json.load(f)
        return datas, question_map[dataset_name]
    else:
        print(f"Dataset '{dataset_name}' not found in prepare_dataset function.")
        exit(-1)

# You will need to move these from freebase_func.py or prompt_list.py if they are there
def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    return extract_relation_prompt % (args.width, args.width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations) + "\nA: "
        
def construct_entity_score_prompt(question, relation, entity_candidates):
    return score_entity_candidates_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '