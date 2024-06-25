
import re
import networkx as nx
from llama_main.llama import Llama, Dialog
import numpy as np
from laser_encoders import LaserEncoderPipeline
import faiss
import json
from tqdm import tqdm
import gc
import torch
import os
import time

def generated():
    generator = Llama.build(
        ckpt_dir="llama_main/llama-2-7b-chat/",
        tokenizer_path="llama_main/tokenizer.model",
        max_seq_len=4096,
        max_batch_size=4,
    )
    return generator

def chat_completion(generator, system_input, user_input):
    dialogs = [
        [
            {"role": "system", "content": system_input},
            {"role": "user", "content": user_input},
        ],
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=None,
        temperature=0,
        top_p=1,
    )
    return results

def extract_and_split_content(text):
    # Regular expression to find content within parentheses
    try:
        pattern = r'\((.*?)\)'
        extracted_content = re.findall(pattern, text)
        # Split each triple by comma and strip whitespace
        split_content = [triple.split(',') for triple in extracted_content]
        # Strip whitespace from each element
        split_content = [[element.strip() for element in triple] for triple in split_content]
        return split_content
    except:
        return []

def extract_triples(results):
    triples = []
    for result in results:
        triple_list = extract_and_split_content(result)

    return triple_list

def load_data_graph(json_file_path):
    # Load data from JSON file
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return nx.node_link_graph(data)

def select_nodes_by_type(G, type_use_content):
    selected_nodes = {}
    for node in G.nodes:
        node_data = G.nodes[node]
        node_type = node_data.get('type')

        if node_type in type_use_content:
            use_content = type_use_content[node_type]
            selected_info = {'type': node_type}
            if use_content:
                selected_info['content'] = node_data.get('content', '')
            else:
                selected_info['data'] = node_data
            selected_nodes[node] = selected_info
    return selected_nodes

def get_embeddings(text, lang_code='eng_Latn', encoder=None):
    encoder = LaserEncoderPipeline(lang=lang_code)
    batch_embeddings = encoder.encode_sentences(text)
    gc.collect()
    torch.cuda.empty_cache()
    return batch_embeddings

def similarity_search(query_embedding, database, k=5, measure='IP'):
    if measure == 'IP':
        index = faiss.IndexFlatIP(1024)  # Index with Inner Product (IP) as the similarity measure
    elif measure == 'L2':
        index = faiss.IndexFlatL2(1024)  # Index with L2 as the similarity measure
    index.add(database)
    D, I = index.search(query_embedding, k)
    return I

def find_and_categorize_predecessors_by_type(G, target_node):
    categorized_nodes = {}
    if target_node in G:
        for pred in G.predecessors(target_node):
            node_type = G.nodes[pred].get('type', 'Unknown')  
            if node_type not in categorized_nodes:
                categorized_nodes[node_type] = []
            categorized_nodes[node_type].append(pred)
    return categorized_nodes

def process_predecessors(G, target_node):
    precis_contents = []  
    section_contents = []  

    if target_node in G:
        for pred in G.predecessors(target_node):
            node_type = G.nodes[pred].get('type')
            
            if node_type == 'precis_intro':
                for precis_node in G.successors(pred):
                    if G.nodes[precis_node].get('type') == 'precis' and G.nodes[precis_node].get('language') == 'English':
                        precis_contents.append(G.nodes[precis_node].get('content', ''))
            
            elif node_type == 'Section' and G.nodes[pred].get('language') == 'en':
                section_contents.append(G.nodes[pred].get('content', ''))

    return precis_contents, section_contents

def merge_categorized_predecessors(categorized_predecessors_list):
    merged_results = {}
    for categorized_nodes in categorized_predecessors_list:
        for node_type, predecessors in categorized_nodes.items():
            if node_type not in merged_results:
                merged_results[node_type] = set()
            merged_results[node_type].update(predecessors)
    return merged_results

def get_first_embedding(G, lang_code='eng_Latn'):
    if not (os.path.exists('data/Processed/embeddings/keyword_embeddings.npy') and os.path.exists('data/Processed/embeddings/conpro_embeddings.npy') and os.path.exists('data/Processed/embeddings/codices_embeddings.npy')):
        selected_types = {
        "ConPro_Category": True,
        "Keyword": False,
        "CODICES_Category": True
        }
        selected_nodes = select_nodes_by_type(G, selected_types)
        keyword_list = []
        conpro_list = []
        codices_list = []
        for node, data in selected_nodes.items():
            node_type = data.get('type') 
            if node_type == "Keyword":
                keyword_list.append(node[8:])
            elif node_type == "ConPro_Category":
                conpro_list.append(node)
            elif node_type == "CODICES_Category":
                codices_list.append(node)
        keyword_embeddings = get_embeddings(keyword_list, 'eng_Latn')
        conpro_embeddings = get_embeddings(conpro_list, 'eng_Latn')
        codices_embeddings = get_embeddings(codices_list, 'eng_Latn')
        np.save('data/Processed/embeddings/keyword_embeddings.npy', keyword_embeddings)
        np.save('data/Processed/embeddings/conpro_embeddings.npy', conpro_embeddings)
        np.save('data/Processed/embeddings/codices_embeddings.npy', codices_embeddings)
        with open('data/Processed/embeddings/keyword_list.json', 'w') as f:
            json.dump(keyword_list, f)
        with open('data/Processed/embeddings/conpro_list.json', 'w') as f:
            json.dump(conpro_list, f)
        with open('data/Processed/embeddings/codices_list.json', 'w') as f:
            json.dump(codices_list, f)
    else:
        keyword_embeddings = np.load('data/Processed/embeddings/keyword_embeddings.npy')
        conpro_embeddings = np.load('data/Processed/embeddings/conpro_embeddings.npy')
        codices_embeddings = np.load('data/Processed/embeddings/codices_embeddings.npy')
        with open('data/Processed/embeddings/keyword_list.json', 'r') as f:
            keyword_list = json.load(f)
        with open('data/Processed/embeddings/conpro_list.json', 'r') as f:
            conpro_list = json.load(f)
        with open('data/Processed/embeddings/codices_list.json', 'r') as f:
            codices_list = json.load(f)

    return keyword_embeddings, conpro_embeddings, codices_embeddings, keyword_list, conpro_list, codices_list

def get_all_embeddings():
    if not (os.path.exists('data/Processed/embeddings/all_metadata.json') and os.path.exists('data/Processed/embeddings/all_embeddings.npy')):
        print("Failed to find embeddings, generating new embeddings...")
    else:
        print("Found embeddings, loading...")
        with open('data/Processed/embeddings/all_metadata.json', 'r') as f:
            all_metadata = json.load(f)
        all_embeddings = np.load('data/Processed/embeddings/all_embeddings.npy')
        return all_metadata, all_embeddings

def get_content_embedding(merged_results, all_metadata, all_embeddings):
        precis_selected_embeddings = []
        section_selected_embeddings = []
        precis_nodes = []
        section_nodes = []
        for node_type, predecessors in merged_results.items():
            if node_type == "precis_intro":
                for predecessor in predecessors:
                    for idx, doc_info in enumerate(all_metadata):
                        if doc_info[0] == (predecessor + "_CODICES") and doc_info[1] == "English":
                            precis_nodes.append(predecessor + "_CODICES")
                            precis_selected_embeddings.append(all_embeddings[idx])
            elif node_type == "Section":
                for predecessor in predecessors:
                    for idx, doc_info in enumerate(all_metadata):
                        if doc_info[0] == predecessor and doc_info[1] == "en":
                            section_nodes.append(predecessor)
                            section_selected_embeddings.append(all_embeddings[idx])
        precis_selected_embeddings_list = np.array(precis_selected_embeddings)
        section_selected_embeddings_list = np.array(section_selected_embeddings)

        print(precis_selected_embeddings_list, section_selected_embeddings_list)
        return precis_selected_embeddings_list, section_selected_embeddings_list, precis_nodes, section_nodes

def chose_content(G, precis_set, section_set, query_embedding_text):
    precis_reference_articles = []
    section_reference_articles = []
    for node_id in precis_set:
        if node_id in G:
            node_content = G.nodes[node_id]['content']
            precis_reference_articles.extend(node_content.split('\n'))
    for node_id in section_set:
        if node_id in G:
            node_content = G.nodes[node_id]['content']
            section_reference_articles.extend(node_content.split('\n'))
    precis_embedding_part = get_embeddings(precis_reference_articles, 'eng_Latn')
    section_embedding_part = get_embeddings(section_reference_articles, 'eng_Latn')
    
    return precis_reference_articles[similarity_search(query_embedding_text, precis_embedding_part)[0][0]], section_reference_articles[similarity_search(query_embedding_text, section_embedding_part)[0][0]]

def main(generator, task_sen, G, keyword_embeddings, conpro_embeddings, codices_embeddings, keyword_list, conpro_list, codices_list, all_metadata, all_embeddings):
    
    print("Running step 1...")

    user_input = "Task sentence: " + "\"" + task_sen + "\""
    results_step1 = chat_completion(generator, system_input, user_input)
    triples_list = extract_triples(results_step1)
    if len(triples_list) == 0:
        triples_list = [task_sen]
    flattened_list = [item for sublist in triples_list for item in sublist]

    print(results_step1)
    print(flattened_list)
    print("\n==================================\n")

    print("Running step 2...")
    # print("Loading data graph...")
    # G = load_data_graph("data/Processed/constitute_ontology.json")
    print("Getting embeddings...")


    query_embedding_triple = get_embeddings(flattened_list, 'eng_Latn')
    query_embedding_text = get_embeddings([user_input], 'zho_Hant')

    print("Getting similarities...")
    triple_keyword_similarities = similarity_search(query_embedding_triple, keyword_embeddings)
    text_keyword_similarities = similarity_search(query_embedding_text, keyword_embeddings)
    triple_conpro_similarities = similarity_search(query_embedding_triple, conpro_embeddings)
    text_conpro_similarities = similarity_search(query_embedding_text, conpro_embeddings)
    triple_codices_similarities = similarity_search(query_embedding_triple, codices_embeddings)
    text_codices_similarities = similarity_search(query_embedding_text, codices_embeddings)

    categorized_predecessors_results = []
    for i in range(len(flattened_list)):
        # print(f"Triple {i+1} matches: {keyword_list[triple_keyword_similarities[i][0]]}")
        result = find_and_categorize_predecessors_by_type(G, "keyword_"+keyword_list[triple_keyword_similarities[i][0]])
        categorized_predecessors_results.append(result)
        # print("=====================================")
        # print(f"Triple {i+1} matches: {conpro_list[triple_conpro_similarities[i][0]]}")
        result = find_and_categorize_predecessors_by_type(G, conpro_list[triple_conpro_similarities[i][0]])
        categorized_predecessors_results.append(result)
        # print("=====================================")
        # print(f"Triple {i+1} matches: {codices_list[triple_codices_similarities[i][0]]}")
        result = find_and_categorize_predecessors_by_type(G, codices_list[triple_codices_similarities[i][0]])
        categorized_predecessors_results.append(result)
    for i in range(len([user_input])):    
        # print(f"Text {i+1} matches: {keyword_list[text_keyword_similarities[i][0]]}")
        result = find_and_categorize_predecessors_by_type(G, "keyword_"+keyword_list[text_keyword_similarities[i][0]])
        categorized_predecessors_results.append(result)
        # print("=====================================")
        # print(f"Text {i+1} matches: {conpro_list[text_conpro_similarities[i][0]]}")
        result = find_and_categorize_predecessors_by_type(G, conpro_list[text_conpro_similarities[i][0]])
        categorized_predecessors_results.append(result)
        # print("=====================================")
        # print(f"Text {i+1} matches: {codices_list[text_codices_similarities[i][0]]}")
        result = find_and_categorize_predecessors_by_type(G, codices_list[text_codices_similarities[i][0]])
        categorized_predecessors_results.append(result)


    print("=====================================")
    print("Merging results...")
    merged_results = merge_categorized_predecessors(categorized_predecessors_results)

    print("Running step 3...")

    print("Getting embeddings...")
    precis_embedding, section_embedding, precis_nodes, section_nodes = get_content_embedding(merged_results, all_metadata, all_embeddings)
    
    print("Getting similarities...")
    triple_precis_similarities = similarity_search(query_embedding_triple, precis_embedding)
    text_precis_similarities = similarity_search(query_embedding_text, precis_embedding)
    triple_section_similarities = similarity_search(query_embedding_triple, section_embedding)
    text_section_similarities = similarity_search(query_embedding_text, section_embedding)
    print("Finish getting embeddings")
    # print(triple_precis_similarities, text_precis_similarities, triple_section_similarities, text_section_similarities)
    print("=====================================")

    print("Running step 4...")


    precis_set = set()
    section_set = set()
    for i in range(len(flattened_list)):
        precis_set.add(precis_nodes[triple_precis_similarities[i][0]])
        section_set.add(section_nodes[triple_section_similarities[i][0]])
    for i in range(len([user_input])):
        precis_set.add(precis_nodes[text_precis_similarities[i][0]])
        section_set.add(section_nodes[text_section_similarities[i][0]])


    ref1, ref2 = chose_content(G, precis_set, section_set, query_embedding_text)
 
    end = time.time()
    print("Time taken: ", end - start)
    print("Running step 5...")
    reference_articles = "Reference Articles: \n" + ref1
    final_input = user_input + "\n" + reference_articles
    print(final_input)
    results_ans1 = chat_completion(generator, system_input, final_input)
    print(results_ans1)

    reference_articles = "Reference Articles: \n" + ref2
    final_input = user_input + reference_articles
    print(final_input)
    results_ans2 = chat_completion(generator, system_input, final_input)
    print(results_ans2)

    return results_step1, results_ans1, results_ans2


if __name__ == "__main__":
    start = time.time()

    lans = ["8_zh_tw"]
    addis = ["train", "test", "dev"]
    print("Loading inital data graph...")
    G = load_data_graph("data/Processed/constitute_ontology.json")
    print("Getting initial embeddings...")
    keyword_embeddings, conpro_embeddings, codices_embeddings, keyword_list, conpro_list, codices_list = get_first_embedding(G, 'eng_Latn')
    all_metadata, all_embeddings = get_all_embeddings()
    for lan in lans:
        for addi in addis:
            folder_path = f'./output/{lan}/{addi}'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            result_counter = 0

            with open(f'./data/ner/{lan}/{addi}_words.txt', 'r') as file:
                lines = file.readlines()

            generator = generated()
            for line in lines:
                result_counter = result_counter + 1
                try:
                    result1, result2, result3 = main(generator, line, G, keyword_embeddings, conpro_embeddings, codices_embeddings, keyword_list, conpro_list, codices_list, all_metadata, all_embeddings)
                    # print(type(result1), type(result2), type(result3))
                    
                    file_name = f"{lan}_{addi}_{result_counter:05}.txt"
                    with open(os.path.join(folder_path, file_name), 'a+') as file:
                        file.write('Input:')
                        file.write("\n")
                        file.write(line)
                        file.write("\n")
                        file.write('Result1:')
                        file.write("\n")
                        file.write(result1[0]['generation']['content'])
                        file.write("\n")
                        file.write('Result2:')
                        file.write("\n")
                        file.write(result2[0]['generation']['content'])
                        file.write("\n")
                        file.write('Result3:')
                        file.write("\n")
                        file.write(result3[0]['generation']['content'])
                        file.write("\n")
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    continue
