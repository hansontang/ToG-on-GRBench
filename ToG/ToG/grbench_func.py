import random
from utils import *
from prompt_list import *

def id2entity_name_or_type(entity_id, id2entity_map):
    # The keys in the map might be strings, so we ensure we look up with a string
    return id2entity_map.get(str(entity_id), "Unknown_Entity")

def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args, graph, id2relation_map, id2vertex_map):
    vertex_index = id2vertex_map.get(entity_id)
    if vertex_index is None:
        print(f"Warning: Entity ID {entity_id} not found in vertex map.")
        return []
    
    vertex = graph.vertex(vertex_index)
    
    head_relations, tail_relations = set(), set()
    relation_property = graph.edge_properties.get("relation")
    if not relation_property: return []

    for edge in vertex.out_edges():
        head_relations.add(id2relation_map.get(str(relation_property[edge])))
    for edge in vertex.in_edges():
        tail_relations.add(id2relation_map.get(str(relation_property[edge])))

    if pre_relations:
        if pre_head is not None and pre_head:
            tail_relations = tail_relations - set(pre_relations)
        else:
            head_relations = head_relations - set(pre_relations)

    total_relations = sorted([r for r in list(head_relations | tail_relations) if r is not None])
    if not total_relations: return []
    
    flag, retrieve_relations_with_scores = False, []
    if args.prune_tools == "llm":
        prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)
        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations)
    # ... (other prune tools logic can be added here)

    return retrieve_relations_with_scores if flag else []

def entity_search(entity_id, relation_name, is_head, graph, relation2id_map, id2vertex_map):
    candidate_entity_ids = []
    relation_id = relation2id_map.get(relation_name)
    if relation_id is None: return []

    vertex_index = id2vertex_map.get(entity_id)
    if vertex_index is None: return []
    
    vertex = graph.vertex(vertex_index)
    
    relation_property = graph.edge_properties["relation"]
    vertex2id_map = {v: k for k, v in id2vertex_map.items()}

    if is_head:
        for edge in vertex.out_edges():
            if relation_property[edge] == relation_id:
                candidate_entity_ids.append(vertex2id_map.get(int(edge.target())))
    else:
        for edge in vertex.in_edges():
            if relation_property[edge] == relation_id:
                candidate_entity_ids.append(vertex2id_map.get(int(edge.source())))

    return [eid for eid in candidate_entity_ids if eid is not None]

# --- NEW: Correct, self-contained entity_score function for GRBench ---
def entity_score(question, entity_candidates_id, base_score, relation, args, id2entity_map):
    """
    Scores a list of candidate entities based on their relevance to the question.
    This version correctly accepts the id2entity_map.
    """
    entity_candidates_names = [id2entity_name_or_type(eid, id2entity_map) for eid in entity_candidates_id]
    
    if all_unknown_entity(entity_candidates_names):
        if not entity_candidates_names: return [], [], []
        return [1.0/len(entity_candidates_names) * base_score] * len(entity_candidates_names), entity_candidates_names, entity_candidates_id
    
    valid_pairs = [(name, eid) for name, eid in zip(entity_candidates_names, entity_candidates_id) if name != "Unknown_Entity"]
    if not valid_pairs: return [], [], []
    entity_candidates_names, entity_candidates_id = map(list, zip(*valid_pairs))

    if len(entity_candidates_names) == 1:
        return [base_score], entity_candidates_names, entity_candidates_id

    zipped = sorted(zip(entity_candidates_names, entity_candidates_id))
    entity_candidates_names, entity_candidates_id = map(list, zip(*zipped))
    
    if args.prune_tools == "llm":
        prompt = construct_entity_score_prompt(question, relation, entity_candidates_names)
        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        scores = [float(x) * base_score for x in clean_scores(result, entity_candidates_names)]
        return scores, entity_candidates_names, entity_candidates_id
    else:
        # Fallback for other pruning tools
        scores = [base_score / len(entity_candidates_names)] * len(entity_candidates_names)
        return scores, entity_candidates_names, entity_candidates_id

# --- Self-contained helper functions (previously imported) ---
def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args, id2entity_map):
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    if not sorted_zipped: return False, [], [], [], []

    width = min(args.width, len(sorted_zipped))
    top_items = sorted_zipped[:width]
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in top_items if score > 0]
    if not filtered_list: return False, [], [], [], []
        
    entities_id, relations, candidates, tops, heads, _ = map(list, zip(*filtered_list))
    tops_names = [id2entity_name_or_type(entity_id, id2entity_map) for entity_id in tops]
    cluster_chain_of_entities = [[(tops_names[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    
    return True, cluster_chain_of_entities, entities_id, relations, heads

def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head):
    if len(entity_candidates) == 0:
        return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head
        
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head

def half_stop(question, cluster_chain_of_entities, depth, args):
    print(f"No new knowledge added during search depth {depth}, stop searching.")
    answer = generate_answer(question, cluster_chain_of_entities, args)
    save_2_jsonl(question, answer, cluster_chain_of_entities, file_name=args.dataset)

def generate_answer(question, cluster_chain_of_entities, args): 
    prompt = answer_prompt + question + '\n'
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return result

def reasoning(question, cluster_chain_of_entities, args):
    prompt = prompt_evaluate + question
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    result = extract_answer(response)
    return if_true(result), response