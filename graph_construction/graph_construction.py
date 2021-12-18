import stanza
import os
import json
from tqdm import tqdm
import sys,os
import pickle
import numpy as np
import sys,os
sys.path.append('/data/data/hujingpeng/sumwithGraph')

nlp = stanza.Pipeline('en', package='mimic', processors={'ner': 'radiology'})
nlp2 = stanza.Pipeline('en', package='mimic', processors='tokenize')

EMB_INIT_RANGE = 1.0


def get_single_entity_graph(document,impression,entity_modified=True,entity_interval=True,entity_deparser=True):
    doc = nlp(document)
    imp = nlp2(impression)
    fingings_list = []
    impression_list = []
    current_senquence_num = 0

    edges = []
    edge_words = []
    edges_type = dict()
    edges_type['deparser'] = []
    edges_type['modified'] = []
    edges_type['interval'] = []
    for sentence in imp.sentences:
        for i in range(len(sentence.words)):
            impression_list.append(sentence.words[i].text)

    for sentence in doc.sentences:
        for i in range(len(sentence.words)):
            fingings_list.append(sentence.words[i].text)

        entities_sentence = []
        entities_id_sentence = []
        words_entities_sentence = []
        words_id_entities_sentence = []
        entities_type = []
        edges_sentence = []
        edges_word_sentence = []
        entities_type_dict = dict()
        edges_type_sentence = dict()
        edges_type_sentence['deparser'] = []
        edges_type_sentence['modified'] = []

        edges_type_sentence['interval'] = []

        for i in range(len(sentence.tokens)):


            token = sentence.tokens[i]
            ent_token = token.ner
            text_token = token.text
            id_token = token.id[0]

            if ent_token!='O':
                ent_index = ent_token.split('-')[0]
                words_entities_sentence.append(text_token)
                words_id_entities_sentence.append(id_token)
                current_ent_type = ent_token.split('-')[-1]
                if ent_index == 'S':
                    entities_sentence.append([text_token])
                    entities_id_sentence.append([id_token])
                    entities_type.append(current_ent_type)

                    if current_ent_type not in entities_type_dict:
                        entities_type_dict[current_ent_type] = [id_token]
                    else:
                        entities_type_dict[current_ent_type].append(id_token)


                elif ent_index == 'B':
                    entities_sentence.append([text_token])
                    entities_id_sentence.append([id_token])

                elif ent_index == 'I':
                    try:
                        entities_sentence[-1].append(text_token)
                        entities_id_sentence[-1].append(id_token)
                    except:
                        entities_sentence.append([text_token])
                        entities_id_sentence.append([id_token])


                elif ent_index == 'E':
                    entities_sentence[-1].append(text_token)
                    entities_id_sentence[-1].append(id_token)
                    entities_type.append(ent_token.split('-')[-1])

                    if current_ent_type not in entities_type_dict:
                        entities_type_dict[current_ent_type] = entities_id_sentence[-1]
                    else:
                        entities_type_dict[current_ent_type] = entities_type_dict[current_ent_type]+entities_id_sentence[-1]


        if entity_deparser:
            if 'deparser' not in edges_type_sentence.keys():
                edges_type_sentence['deparser'] = []


            for word in sentence.words:
                word_id = word.id
                word_head = word.head

                if word_id in words_id_entities_sentence and word_head in words_id_entities_sentence:
                    word_doc_id = word_id+current_senquence_num-1
                    word_doc_head = word_head+current_senquence_num-1
                    if word_doc_id>=0 and word_doc_head>=0 and [word_doc_id,word_doc_head] not in edges_sentence\
                            and [word_doc_head,word_doc_id] not in edges_sentence:

                        edges_sentence.append([word_doc_id,word_doc_head])
                        edges_word_sentence.append([sentence.words[word_id-1].text,sentence.words[word_head-1].text])

                    if word_doc_id>=0 and word_doc_head>=0:
                        edges_type_sentence['deparser'].append([word_doc_id,word_doc_head])
                    # print('parser', sentence.words[word_id-1].text,sentence.words[word_head-1].text)

        if entity_modified:
            if 'modified' not in edges_type_sentence.keys():
                edges_type_sentence['modified'] = []



            if 'ANATOMY' in entities_type_dict and 'ANATOMY_MODIFIER' in entities_type_dict:
                anatomy_ids = entities_type_dict['ANATOMY']
                anatomy_modifier_ids = entities_type_dict['ANATOMY_MODIFIER']
                for anatomy_id in anatomy_ids:
                    for anatomy_modifier_id in anatomy_modifier_ids:
                        anatomy_doc_id = anatomy_id + current_senquence_num - 1

                        anatomy_modifier_doc_id = anatomy_modifier_id + current_senquence_num - 1
                        if anatomy_doc_id >= 0 and anatomy_modifier_doc_id >= 0 and [anatomy_doc_id,anatomy_modifier_doc_id] not in edges_sentence\
                               and [anatomy_modifier_doc_id,anatomy_doc_id] not in edges_sentence:
                            edges_sentence.append([anatomy_doc_id, anatomy_modifier_doc_id])
                            edges_word_sentence.append(
                                [sentence.words[anatomy_id - 1].text, sentence.words[anatomy_modifier_id - 1].text])
                        if anatomy_doc_id >= 0 and anatomy_modifier_doc_id >= 0:
                            edges_type_sentence['modified'].append([anatomy_doc_id, anatomy_modifier_doc_id])
                            # print('anatomy', sentence.words[anatomy_id - 1].text,
                            #       sentence.words[anatomy_modifier_id - 1].text)

            if 'OBSERVATION' in entities_type_dict and 'OBSERVATION_MODIFIER' in entities_type_dict:
                observation_ids = entities_type_dict['OBSERVATION']
                observation_modifier_ids = entities_type_dict['OBSERVATION_MODIFIER']
                for observation_id in observation_ids:
                    for observation_modifier_id in observation_modifier_ids:
                        observation_doc_id = observation_id + current_senquence_num - 1
                        observation_modifier_doc_id = observation_modifier_id + current_senquence_num - 1
                        if observation_doc_id >= 0 and observation_modifier_doc_id >= 0 and \
                                [observation_doc_id, observation_modifier_doc_id] not in edges_sentence\
                                and [observation_modifier_doc_id, observation_doc_id] not in edges_sentence:
                            edges_sentence.append([observation_doc_id, observation_modifier_doc_id])
                            edges_word_sentence.append(
                                [sentence.words[observation_id - 1].text, sentence.words[observation_modifier_id - 1].text])
                        if  observation_doc_id >= 0 and observation_modifier_doc_id >= 0:
                            edges_type_sentence['modified'].append([observation_doc_id, observation_modifier_doc_id])
                            # print('observation_id', sentence.words[observation_id - 1].text,
                            #       sentence.words[observation_modifier_id - 1].text)

        if entity_interval:
            if 'interval' not in edges_type_sentence.keys():
                edges_type_sentence['interval'] = []

            for m in range(len(entities_id_sentence)):
                entity_length = len(entities_id_sentence[m])
                if entity_length>1:
                    for n in range(entity_length-1):
                        current_id = entities_id_sentence[m][n]
                        current_tag_id = entities_id_sentence[m][n+1]
                        current_doc_id = current_id+current_senquence_num-1
                        current_doc_tag_id = current_tag_id +current_senquence_num-1
                        if current_doc_id>=0 and current_doc_tag_id>=0 and [current_doc_id,current_doc_tag_id] not in edges_sentence \
                               and [current_doc_tag_id, current_doc_id] not in edges_sentence  :
                            edges_sentence.append([current_doc_id, current_doc_tag_id])
                            edges_word_sentence.append([sentence.words[current_id-1].text,sentence.words[current_tag_id-1].text])

                        if current_doc_id>=0 and current_doc_tag_id>=0:
                            edges_type_sentence['interval'].append([current_doc_id, current_doc_tag_id])
                            # print('interval entity', sentence.words[current_doc_id - 1].text,
                            #       sentence.words[current_tag_id - 1].text)

        current_senquence_num = current_senquence_num + len(sentence.words)

        edges_type['deparser']  = edges_type['deparser'] + edges_type_sentence['deparser']
        edges_type['modified']  = edges_type['modified'] + edges_type_sentence['modified']
        edges_type['interval']  = edges_type['interval'] + edges_type_sentence['interval']

        edges = edges + edges_sentence
        edge_words = edge_words+edges_word_sentence

    pyg_edges_document = []
    src_index = []
    tag_index = []
    for edge_item in edges:
        src_index.append(edge_item[0])
        tag_index.append(edge_item[1])
    pyg_edges_document.append(src_index)
    pyg_edges_document.append(tag_index)


    return pyg_edges_document,edge_words,fingings_list,impression_list,edges_type


def build_entity_graph(data_path,entity_modified=True,entity_interval=True,entity_deparser=True):
    file = open(data_path, 'r', encoding='utf-8')
    lines = file.readlines()
    num_line = len(lines)
    new_json_path = data_path.replace('.jsonl', '')
    name_type = '_with_entity'
    if entity_modified:
        name_type = name_type + '_modified'

    if entity_interval:
        name_type = name_type + '_interval'

    if entity_deparser:
        name_type = name_type + '_deparser'
    new_json_path = new_json_path + name_type + '.jsonl'
    if (os.path.exists(new_json_path)):
        print('there are already exist ' + new_json_path)
        return new_json_path
    else:
        new_json_file = open(new_json_path, 'w', encoding='utf-8')
        for i in tqdm(range(num_line)):
            dic_items = json.loads(lines[i])
            findings_list = dic_items['findings']
            findings = ' '.join(findings_list)
            impression_list = dic_items['impression']
            impression = ' '.join(impression_list)
            edges_with_nodeid = []

            edges,edge_words,fingings_list,impression_list,edges_type_sentence = get_single_entity_graph(findings,impression,entity_modified=entity_modified,
                                                                                     entity_interval=entity_interval,
                                                                                     entity_deparser=entity_deparser)

            dic_items['pyg_edges_document'] = edges
            dic_items['findings'] = fingings_list
            dic_items['impression'] = impression_list
            dic_items['edge_words'] = edge_words

            nodes = []

            finding_list = dic_items['findings']
            edges = dic_items['pyg_edges_document']
            src_index = edges[0]
            tag_index = edges[1]

            src_node_index = []
            tag_node_index = []
            word_dict = dict()
            for k in range(len(tag_index)):
                src_word = finding_list[src_index[k]]
                tag_word = finding_list[tag_index[k]]
                if src_word not in word_dict:
                    word_dict[src_word] = len(word_dict)
                    nodes.append(src_word)
                if tag_word not in word_dict:
                    word_dict[tag_word] = len(word_dict)
                    nodes.append(tag_word)

                src_node_index.append(word_dict[src_word])
                tag_node_index.append(word_dict[tag_word])

            edges_with_nodeid.append(src_node_index)
            edges_with_nodeid.append(tag_node_index)



            edges_modified = edges_type_sentence['modified']
            edges_deparser = edges_type_sentence['deparser']
            edges_interval = edges_type_sentence['interval']

            edges_modified_with_nodeid = []
            edges_deparser_with_nodeid = []
            edges_interval_with_nodeid = []


            if len(edges_modified)>0:
                modified_src_node_index = []
                modified_tag_node_index = []
                for k in range(len(edges_modified)):
                    src_word = finding_list[edges_modified[k][0]]
                    tag_word = finding_list[edges_modified[k][1]]
                    modified_src_node_index.append(word_dict[src_word])
                    modified_tag_node_index.append(word_dict[tag_word])

                edges_modified_with_nodeid.append(modified_src_node_index)
                edges_modified_with_nodeid.append(modified_tag_node_index)

            if len(edges_deparser)>0:

                deparser_src_node_index = []
                deparser_tag_node_index = []
                for k in range(len(edges_deparser)):
                    src_word = finding_list[edges_deparser[k][0]]
                    tag_word = finding_list[edges_deparser[k][1]]
                    deparser_src_node_index.append(word_dict[src_word])
                    deparser_tag_node_index.append(word_dict[tag_word])
                edges_deparser_with_nodeid.append(deparser_src_node_index)
                edges_deparser_with_nodeid.append(deparser_tag_node_index)

            if len(edges_interval)>0:

                interval_src_node_index = []
                interval_tag_node_index = []
                for k in range(len(edges_interval)):
                    src_word = finding_list[edges_interval[k][0]]
                    tag_word = finding_list[edges_interval[k][1]]
                    interval_src_node_index.append(word_dict[src_word])
                    interval_tag_node_index.append(word_dict[tag_word])
                edges_interval_with_nodeid.append(interval_src_node_index)
                edges_interval_with_nodeid.append(interval_tag_node_index)

            dic_items['nodes'] = nodes
            dic_items['edges_with_nodeid'] = edges_with_nodeid

            dic_items['edges_interval_with_nodeid'] = edges_interval_with_nodeid
            dic_items['edges_modified_with_nodeid'] = edges_modified_with_nodeid
            dic_items['edges_deparser_with_nodeid'] = edges_deparser_with_nodeid


            if len(fingings_list)>10 and len(impression_list)>3:
                print(json.dumps(dic_items), file=new_json_file)


# radiology

def add_edge_words(data_path):

    file = open(data_path, 'r', encoding='utf-8')
    lines = file.readlines()
    num_line = len(lines)
    new_json_path = data_path.replace('.jsonl', '')
    new_json_path = new_json_path + '_with_entity_graph_node' + '.jsonl'
    if (os.path.exists(new_json_path)):
        print('there are already exist ' + new_json_path)
        return new_json_path
    else:
        new_json_file = open(new_json_path, 'w', encoding='utf-8')
        for i in tqdm(range(num_line)):
            dic_items = json.loads(lines[i])
            edge_words = []
            edges_with_nodeid = []
            nodes = []

            finding_list = dic_items['findings']
            edges = dic_items['pyg_edges_document']
            src_index = edges[0]
            tag_index = edges[1]

            src_node_index = []
            tag_node_index = []
            word_dict = dict()
            for k in range(len(tag_index)):
                src_word = finding_list[src_index[k]]
                tag_word = finding_list[tag_index[k]]
                if src_word not in word_dict:
                    word_dict[src_word] = len(word_dict)
                    nodes.append(src_word)
                if tag_word not in word_dict:
                    word_dict[tag_word] = len(word_dict)
                    nodes.append(tag_word)

                edge_words.append([src_word,tag_word])

                src_node_index.append(word_dict[src_word])
                tag_node_index.append(word_dict[tag_word])

            edges_with_nodeid.append(src_node_index)
            edges_with_nodeid.append(tag_node_index)

            dic_items['edge_words'] = edge_words
            dic_items['nodes'] = nodes
            dic_items['edges_with_nodeid'] = edges_with_nodeid

            print(json.dumps(dic_items), file=new_json_file)



def obtain_word_pair_for(data_path):
    file = open(data_path, 'r', encoding='utf-8')
    lines = file.readlines()
    num_line = len(lines)
    new_json_path = data_path.replace('.jsonl', '')
    new_json_path = new_json_path + '_words_pair' + '.jsonl'
    if (os.path.exists(new_json_path)):
        print('there are already exist ' + new_json_path)
        return new_json_path
    else:
        new_json_file = open(new_json_path, 'w', encoding='utf-8')
        for i in tqdm(range(num_line)):
            dic_items = json.loads(lines[i])
            edge_words = []
            edges_with_nodeid = []
            nodes = []

            finding_list = dic_items['findings']
            edges = dic_items['pyg_edges_document']
            edges_interval_with_nodeid = dic_items['edges_interval_with_nodeid']
            edges_modified_with_nodeid = dic_items['edges_modified_with_nodeid']
            edges_deparser_with_nodeid = dic_items['edges_deparser_with_nodeid']
            edges_word = dic_items['edge_words']

            # if len(edges_interval_with_nodeid) == 0:
            #     edges_interval_with_nodeid = [[0],[0]]
            # if len(edges_modified_with_nodeid) == 0:
            #     edges_modified_with_nodeid = [[0], [0]]
            # if len(edges_deparser_with_nodeid) == 0:
            #     edges_deparser_with_nodeid = [[0], [0]]

            src_index = edges[0]
            tag_index = edges[1]
            node_word = dic_items['nodes']
            w2i = dict()
            i2w = dict()

            for i in range(len(node_word)):
                w2i[node_word[i]] = len(w2i)
                i2w[len(i2w)] = node_word[i]

            if len(edges_interval_with_nodeid) != 0 :
                edges_word_interval = []
                src_index_edges_interval = edges_interval_with_nodeid[0]
                tag_index_edges_interval = edges_interval_with_nodeid[1]
                for i in range(len(src_index_edges_interval)):
                    try:
                        pair = [i2w[src_index_edges_interval[i]],i2w[tag_index_edges_interval[i]]]
                        pair_ = [pair[1],pair[0]]
                    except:
                        import pdb
                        pdb.set_trace()
                    if pair not in edges_word and pair_ not in edges_word:
                        print('error')
                    edges_word_interval.append(pair)
            else:
                edges_word_interval = []

            if len(edges_modified_with_nodeid) != 0:
                edges_word_modified = []
                src_index_edges_modified = edges_modified_with_nodeid[0]
                tag_index_edges_modified = edges_modified_with_nodeid[1]

                for i in range(len(src_index_edges_modified)):
                    pair = [i2w[src_index_edges_modified[i]], i2w[tag_index_edges_modified[i]]]
                    pair_ = [pair[1], pair[0]]
                    if pair not in edges_word and pair_ not in edges_word:
                        print('error')
                    edges_word_modified.append(pair)
            else:
                edges_word_modified = []

            if len(edges_deparser_with_nodeid) != 0:
                edges_word_deparser = []
                src_index_edges_deparser = edges_deparser_with_nodeid[0]
                tag_index_edges_deparser = edges_deparser_with_nodeid[1]

                for i in range(len(src_index_edges_deparser)):
                    pair = [i2w[src_index_edges_deparser[i]], i2w[tag_index_edges_deparser[i]]]
                    pair_ = [pair[1], pair[0]]
                    if pair not in edges_word and pair_ not in edges_word:
                        print('error')
                    edges_word_deparser.append(pair)
            else:
                edges_word_deparser = []

            dic_items['edges_word_deparser'] = edges_word_deparser
            dic_items['edges_word_modified'] = edges_word_modified
            dic_items['edges_word_interval'] = edges_word_interval
            print(json.dumps(dic_items), file=new_json_file)


if __name__ == '__main__':
    build_entity_graph('example.jsonl')


