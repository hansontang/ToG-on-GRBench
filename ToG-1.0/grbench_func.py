# grbench_func.py

from utils import *
from prompt_list import *
import graph_tool.all as gt

# --- Graph Traversal Functions ---
def search_relations(graph, vertex_index, id2relation_map):
    """Finds all relation names connected to a node."""
    relations = set()
    vertex = graph.vertex(vertex_index)
    relation_prop = graph.edge_properties["relation"]
    for edge in vertex.out_edges():
        relations.add(id2relation_map.get(str(relation_prop[edge])))
    for edge in vertex.in_edges():
        relations.add(f"{id2relation_map.get(str(relation_prop[edge]))}_of")
    return sorted([r for r in list(relations) if r is not None])

def search_entities(graph, vertex_index, relation_name, id2entity_map, relation2id_map, id2vertex_map):
    """Finds all neighbor entities along a specific relation."""
    entities = []
    vertex = graph.vertex(vertex_index)
    relation_prop = graph.edge_properties["relation"]
    is_reverse = relation_name.endswith("_of")
    base_relation_name = relation_name.replace("_of", "")
    
    relation_id = relation2id_map.get(base_relation_name)
    if relation_id is None: return []

    edges = vertex.in_edges() if is_reverse else vertex.out_edges()
    vertex2id_map = {v: k for k, v in id2vertex_map.items()}

    for edge in edges:
        if relation_prop[edge] == relation_id:
            neighbor_vertex = edge.source() if is_reverse else edge.target()
            neighbor_id = vertex2id_map.get(int(neighbor_vertex))
            if neighbor_id:
                entities.append({"id": neighbor_id, "name": id2entity_map.get(neighbor_id, "Unknown")})
    return entities

# --- LLM Logic Functions ---
def prune_relations(question, topic_entity_name, relations, llm_config):
    """Uses an LLM to prune relations."""
    if not relations: return []
    prompt = RELATION_PRUNE_PROMPT.format(question=question, topic_entity=topic_entity_name, relations=str(relations))
    response = call_llm(prompt, llm_config)
    scores = parse_llm_list_output(response)
    if len(scores) != len(relations): 
        print(f"Warning: Relation score count mismatch. Expected {len(relations)}, got {len(scores)}.")
        return []
    return sorted(zip(relations, scores), key=lambda x: x[1], reverse=True)

def prune_entities(question, current_path_str, entities, llm_config):
    """Uses an LLM to prune entities."""
    if not entities: return []
    entity_names = [e['name'] for e in entities]
    prompt = ENTITY_PRUNE_PROMPT.format(question=question, current_path=current_path_str, entities=str(entity_names))
    response = call_llm(prompt, llm_config)
    scores = parse_llm_list_output(response)
    if len(scores) != len(entities): 
        print(f"Warning: Entity score count mismatch. Expected {len(entities)}, got {len(scores)}.")
        return []
    return sorted(zip(entities, scores), key=lambda x: x[1], reverse=True)

def reason_and_decide(question, paths, llm_config):
    """Uses an LLM to decide if information is sufficient."""
    path_strs = [" -> ".join(p) for p in paths]
    prompt = REASONING_PROMPT.format(question=question, knowledge_paths="\n".join(f"- {s}" for s in path_strs))
    response = call_llm(prompt, llm_config)
    return parse_llm_json_output(response).get("sufficient", False)

def generate_final_answer(question, paths, llm_config):
    """Uses an LLM to generate the final answer."""
    path_strs = [" -> ".join(p) for p in paths]
    prompt = ANSWER_GENERATION_PROMPT.format(question=question, knowledge_paths="\n".join(f"- {s}" for s in path_strs))
    return call_llm(prompt, llm_config)