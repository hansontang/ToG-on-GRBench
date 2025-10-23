# main_grbench.py

import argparse
import json
from tqdm import tqdm
import graph_tool.all as gt
import jsonlines
from grbench_func import *
from utils import *

def load_data(args):
    """加载所有预处理好的数据文件。"""
    print("--- Loading preprocessed data... ---")
    g = gt.load_graph(args.graph_path)
    with open(args.entity_name_path, 'r', encoding='utf-8') as f:
        id2entity = json.load(f)
    with open(args.relation_name_path, 'r', encoding='utf-8') as f:
        id2relation = json.load(f)
        relation2id = {name: int(id_str) for id_str, name in id2relation.items()}
    with open(args.entity_vertex_path, 'r', encoding='utf-8') as f:
        id2vertex = json.load(f)
    print("All data loaded successfully.")
    return g, id2entity, id2relation, relation2id, id2vertex

def run_tog(args):
    """主函数，运行 ToG 算法。"""
    
    g, id2entity, id2relation, relation2id, id2vertex = load_data(args)
    
    # 将所有LLM相关配置集中到一个字典中
    llm_config = {
        "api_key": args.api_key,
        "model_name": args.llm_model
    }

    with jsonlines.open(args.dataset) as reader:
        for item in tqdm(reader, desc="Processing QA pairs"):
            question = item['question']
            ground_truth = item.get('answer', item.get('answers'))
            topic_entity_id = item.get('topic_entity_id')

            if not topic_entity_id or topic_entity_id not in id2entity:
                print(f"Skipping question due to missing/invalid topic entity: {question}")
                continue

            topic_entity_name = id2entity[topic_entity_id]
            
            beam = [([topic_entity_name], topic_entity_id)] 
            all_generated_paths = {tuple([topic_entity_name])}
            final_paths_for_generation = []

            for depth in range(args.depth):
                next_beam = []
                
                for path, current_entity_id in beam:
                    current_vertex_idx = id2vertex.get(current_entity_id)
                    if current_vertex_idx is None: continue

                    relations = search_relations(g, current_vertex_idx, id2relation)
                    scored_relations = prune_relations(question, path[-1], relations, llm_config)
                    
                    for rel_name, rel_score in scored_relations[:args.width]:
                        if rel_score < 0.1: continue
                        
                        new_path_str = " -> ".join(path + [rel_name, "?"])
                        entities = search_entities(g, current_vertex_idx, rel_name, id2entity, relation2id, id2vertex)
                        scored_entities = prune_entities(question, new_path_str, entities, llm_config)

                        for entity_info, entity_score in scored_entities[:args.width]:
                            if entity_score < 0.1: continue
                            
                            new_path = tuple(path + [rel_name, entity_info['name']])
                            if new_path not in all_generated_paths:
                                next_beam.append((list(new_path), entity_info['id']))
                                all_generated_paths.add(new_path)
                
                if not next_beam:
                    print(f"Beam is empty at depth {depth+1}, stopping.")
                    break
                
                beam = sorted(next_beam, key=lambda x: len(x[0]), reverse=True)[:args.width] 
                final_paths = [p for p, _ in beam]
                
                if reason_and_decide(question, final_paths, llm_config):
                    print(f"Sufficient info found at depth {depth+1}. Stopping.")
                    final_paths_for_generation = final_paths
                    break
            
            if not final_paths_for_generation:
                 final_paths_for_generation = [p for p, _ in beam]

            model_answer = generate_final_answer(question, final_paths_for_generation, llm_config)
            
            result_data = {
                "question": question,
                "ground_truth": ground_truth,
                "model_answer": model_answer.strip(),
                "topic_entity": f"{topic_entity_name} ({topic_entity_id})",
                "retrieved_paths": [" -> ".join(p) for p in final_paths_for_generation]
            }
            save_result_to_jsonl(result_data, args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ToG on preprocessed and linked GRBench data.")
    # 文件路径参数
    parser.add_argument("--dataset", type=str,default="/shared/data3/hansont2/GRbench/processed/amazon/data.json", help="Path to the linked QA dataset (.jsonl).")
    parser.add_argument("--output_path", type=str, default="/shared/data3/hansont2/GRbench/processed/amazon/result.json", help="Path to save the output results.")
    parser.add_argument("--graph_path", type=str, default="/shared/data3/hansont2/GRbench/processed/amazon/graph.gt", help="Path to the graph.gt file.")
    parser.add_argument("--entity_name_path", default="/shared/data3/hansont2/GRbench/processed/amazon/entity_id_to_name.json",  help="Path to entity_id_to_name.json file.")
    parser.add_argument("--relation_name_path", default="/shared/data3/hansont2/GRbench/processed/amazon/relation_id_to_name.json",  help="Path to relation_id_to_name.json file.")
    parser.add_argument("--entity_vertex_path", default="/shared/data3/hansont2/GRbench/processed/amazon/entity_id_to_vertex_index.json",  help="Path to entity_id_to_vertex_index.json file.")
    
    # ToG 超参数
    parser.add_argument("--depth", type=int, default=3, help="Search depth for ToG.")
    parser.add_argument("--width", type=int, default=3, help="Search width (beam size) for ToG.")

    # LLM 配置
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="Name of the LLM for entity extraction.")
    parser.add_argument("--api_key", type=str, default="", help="Your OpenAI API key.")

    args = parser.parse_args()
    if not args.api_key:
        raise ValueError("API key and base URL must be provided via arguments or environment variables.")
    run_tog(args)
