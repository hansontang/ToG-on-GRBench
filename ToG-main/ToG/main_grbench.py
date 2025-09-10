import argparse
import json
import random
import re
from tqdm import tqdm
import graph_tool.all as gt
import jsonlines
from grbench_func import *
from utils import *

def load_grbench_data_for_ToG(graph_path, entity_name_path, relation_name_path, entity_vertex_path):
    # This function is correct
    print("Loading GRBench graph data for ToG...")
    g = gt.load_graph(graph_path)
    with open(entity_name_path, 'r', encoding='utf-8') as f:
        id2entity_map = json.load(f)
    with open(relation_name_path, 'r', encoding='utf-8') as f:
        id2relation_map = json.load(f)
        relation2id_map = {name: int(id) for id, name in id2relation_map.items()}
    with open(entity_vertex_path, 'r', encoding='utf-8') as f:
        id2vertex_map = json.load(f)
    print(f"Graph loaded: {g.num_vertices()} vertices, {g.num_edges()} edges.")
    return g, id2entity_map, id2relation_map, relation2id_map, id2vertex_map

# --- FINAL, MORE ROBUST EXTRACTION FUNCTION ---
def extract_topic_entity_from_question(question):
    """
    Uses a series of regular expressions to find the topic entity in various question formats.
    """
    # Define a list of patterns from most specific to most general
    patterns = [
        # For questions like: "...item 'ENTITY_NAME'?"
        r"item '(.*?)'",
        # For questions like: "...brand of ENTITY_NAME?" or "...item ENTITY_NAME have..."
        r"(?:brand of|item|with|query:)\s+((?:[\w\s'-:]+(?:\s\(.+?\))?)+?)(?:\?|\s+have|\s+cost|\s+fall|\s+classified)",
        # For recommendation questions: "based on his history: ["ENTITY_NAME", ...]"
        r'history:\s*\[\s*"(.*?)"'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            # Return the first captured group, stripped of whitespace
            return match.group(1).strip()
            
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ToG method on GRBench dataset.")
    # (Parser arguments remain the same as before)
    parser.add_argument("--dataset", type=str, default="grbench", help="Dataset name.")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--temperature_exploration", type=float, default=0.4)
    parser.add_argument("--temperature_reasoning", type=float, default=0)
    parser.add_argument("--width", type=int, default=3, help="Search width (beam size) of ToG.")
    parser.add_argument("--depth", type=int, default=3, help="Search depth of ToG.")
    parser.add_argument("--LLM_type", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--opeani_api_keys", type=str, default="", help="Your OpenAI API key.")
    parser.add_argument("--prune_tools", type=str, default="llm", help="Pruning tool for ToG: llm, bm25, or sentencebert.")
    parser.add_argument("--num_retain_entity", type=int, default=5, help="Number of entities to sample before scoring.")
    parser.add_argument("--graph_path", type=str, default="/shared/data3/hansont2/GRbench/processed/amazon/graph.gt", help="Path to GRBench graph.gt file.")
    # UPDATED argument names for clarity
    parser.add_argument("--entity_name_path", default="/shared/data3/hansont2/GRbench/processed/amazon/entity_id_to_name.json",  help="Path to entity_id_to_name.json file.")
    parser.add_argument("--relation_name_path", default="/shared/data3/hansont2/GRbench/processed/amazon/relation_id_to_name.json",  help="Path to relation_id_to_name.json file.")
    # *** NEW: Argument for the new map ***
    parser.add_argument("--entity_vertex_path", default="/shared/data3/hansont2/GRbench/processed/amazon/entity_id_to_vertex_index.json",  help="Path to entity_id_to_vertex_index.json file.")
    parser.add_argument("--qa_file_path", default="/shared/data3/hansont2/GRbench/QA/amazon/data.json",  help="Path to QA data JSON file.")
    args = parser.parse_args()

    # Load all data
    g, id2entity, id2relation, relation2id, id2vertex = load_grbench_data_for_ToG(
        args.graph_path, args.entity_name_path, args.relation_name_path, args.entity_vertex_path
    )

    with open(args.qa_file_path, 'r', encoding='utf-8') as f:
        datas = [item for item in jsonlines.Reader(f)]

    print("Creating name-to-ID lookup map for topic entities...")
    name2id = {name: id for id, name in id2entity.items()}
    
    for data in tqdm(datas):
        question = data['question']
        
        topic_entity_name = extract_topic_entity_from_question(question)
        topic_entity = {}
        if topic_entity_name:
            topic_entity_id = name2id.get(topic_entity_name)
            if topic_entity_id:
                topic_entity = {topic_entity_id: topic_entity_name}
            else:
                print(f"Warning: Could not find ID for extracted entity '{topic_entity_name}'")
        
        if not topic_entity:
            print(f"Warning: Could not extract a valid topic entity from question: '{question}'. Skipping.")
            results = generate_without_explored_paths(question, args)
            save_2_jsonl(question, results, [], file_name=args.dataset)
            continue

        cluster_chain_of_entities = []
        pre_relations = []
        pre_heads = [-1] * len(topic_entity)
        flag_printed = False
        
        # Main ToG loop
        for depth in range(1, args.depth + 1):
            current_entity_relations_list = []
            i = 0
            for entity_id, entity_name in topic_entity.items():
                if entity_id != "[FINISH_ID]":
                    relations_with_scores = relation_search_prune(
                        entity_id, entity_name, pre_relations, pre_heads[i], question, args, 
                        g, id2relation, id2vertex
                    )
                    current_entity_relations_list.extend(relations_with_scores)
                i += 1
            
            if not current_entity_relations_list:
                half_stop(question, cluster_chain_of_entities, depth, args)
                flag_printed = True
                break

            total_candidates, total_scores, total_relations = [], [], []
            total_entities_id, total_topic_entities, total_head = [], [], []

            for entity_relation in current_entity_relations_list:
                entity_candidates_id = entity_search(
                    entity_relation['entity'], entity_relation['relation'], entity_relation['head'], 
                    g, relation2id, id2vertex
                )
                
                if not entity_candidates_id: continue
                if args.prune_tools == "llm" and len(entity_candidates_id) > 20:
                    entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)

                scores, entity_candidates_names, entity_candidates_id = entity_score(
                    question, entity_candidates_id, entity_relation['score'], entity_relation['relation'], args, id2entity
                )
                
                total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(
                    entity_candidates_names, entity_relation, scores, entity_candidates_id, total_candidates, total_scores, 
                    total_relations, total_entities_id, total_topic_entities, total_head
                )
            
            if not total_candidates:
                half_stop(question, cluster_chain_of_entities, depth, args)
                flag_printed = True
                break
                
            flag, chain, entities_id, pre_relations, pre_heads = entity_prune(
                total_entities_id, total_relations, total_candidates, total_topic_entities, 
                total_head, total_scores, args, id2entity
            )
            
            if not flag:
                half_stop(question, cluster_chain_of_entities, depth, args)
                flag_printed = True
                break

            cluster_chain_of_entities.append(chain)
            
            stop, results = reasoning(question, cluster_chain_of_entities, args)
            if stop:
                save_2_jsonl(question, results, cluster_chain_of_entities, file_name=args.dataset)
                flag_printed = True
                break
            else:
                flag_finish, entities_id = if_finish_list(entities_id)
                if flag_finish:
                    half_stop(question, cluster_chain_of_entities, depth, args)
                    flag_printed = True
                    break
                else:
                    topic_entity = {eid: id2entity_name_or_type(eid, id2entity) for eid in entities_id}
        
        if not flag_printed:
            results = generate_answer(question, cluster_chain_of_entities, args)
            save_2_jsonl(question, results, cluster_chain_of_entities, file_name=args.dataset)