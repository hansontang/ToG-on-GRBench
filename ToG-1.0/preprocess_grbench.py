import json
import argparse
from tqdm import tqdm
import graph_tool.all as gt
import os
import numpy as np

NODE_NAME_FEATURES = {
    'item': 'title', 'brand': 'name', 'paper': 'title',
    'author': 'name', 'venue': 'name',
}

def preprocess_graph_json(input_path, output_dir):
    print(f"Loading raw graph from: {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    print("Step 1: Creating mappings and graph structure...")
    
    entity_id_to_name = {}
    relation_name_to_id = {}
    # *** NEW: This map is the key to the fix ***
    entity_id_to_vertex_index = {}
    
    g = gt.Graph(directed=True)

    print("  - Pass 1: Adding vertices and building maps...")
    for node_type_key in tqdm(graph_data.keys(), desc="Processing node types"):
        node_type = node_type_key.split('_nodes')[0]
        name_feature = NODE_NAME_FEATURES.get(node_type, 'name')

        for node_id, node_data in graph_data[node_type_key].items():
            if node_id not in entity_id_to_vertex_index:
                v = g.add_vertex()
                vertex_index = int(v)
                entity_id_to_vertex_index[node_id] = vertex_index
                entity_id_to_name[node_id] = node_data['features'].get(name_feature, "Unknown")

    print("  - Pass 2: Collecting and adding edges...")
    edge_list = []
    relation_property_list = []
    relation_id_counter = 0

    for node_type_key in tqdm(graph_data.keys(), desc="Collecting edges"):
        for source_node_id, node_data in graph_data[node_type_key].items():
            source_vertex_index = entity_id_to_vertex_index[source_node_id]
            if 'neighbors' in node_data:
                for relation_name, neighbor_ids in node_data['neighbors'].items():
                    if relation_name not in relation_name_to_id:
                        relation_name_to_id[relation_name] = relation_id_counter
                        relation_id_counter += 1
                    relation_id = relation_name_to_id[relation_name]
                    for target_node_id in neighbor_ids:
                        if target_node_id in entity_id_to_vertex_index:
                            target_vertex_index = entity_id_to_vertex_index[target_node_id]
                            edge_list.append((source_vertex_index, target_vertex_index))
                            relation_property_list.append(relation_id)

    print(f"  - Adding {len(edge_list)} edges to the graph in a single batch...")
    g.add_edge_list(edge_list)

    relation_prop = g.new_edge_property("int")
    relation_prop.a = np.array(relation_property_list)
    g.edge_properties["relation"] = relation_prop
    relation_id_to_name = {v: k for k, v in relation_name_to_id.items()}

    print("Step 2: Saving processed files...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Save all three files ---
    g.save(os.path.join(output_dir, "graph.gt"))
    with open(os.path.join(output_dir, "entity_id_to_name.json"), 'w', encoding='utf-8') as f:
        json.dump(entity_id_to_name, f, indent=4)
    with open(os.path.join(output_dir, "relation_id_to_name.json"), 'w', encoding='utf-8') as f:
        json.dump(relation_id_to_name, f, indent=4)
    # *** NEW: Save the third, crucial mapping file ***
    with open(os.path.join(output_dir, "entity_id_to_vertex_index.json"), 'w', encoding='utf-8') as f:
        json.dump(entity_id_to_vertex_index, f, indent=4)

    print("\nPreprocessing complete! All necessary files have been generated.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GRBench graph.json into optimized formats.")
    parser.add_argument("--input_graph_json", type=str, default="/shared/data3/hansont2/GRbench/graph/amazon/graph.json", help="Path to the raw graph.json file.")
    parser.add_argument("--output_dir", type=str, default="/shared/data3/hansont2/GRbench/processed/amazon", help="Directory to save the processed files (graph.gt, entity_map.json, relation_map.json).")
    args = parser.parse_args()
    
    preprocess_graph_json(args.input_graph_json, args.output_dir)