import json
import os
import numpy as np
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from chatgpt_lib.copy_detect import find_largest_number
from chatgpt_lib.log import log
import torch
from chatgpt_lib import WORK_DIR
import configparser
import re

config = configparser.ConfigParser()
config_path = os.path.join(WORK_DIR, 'config', 'config.ini')
config.read(config_path)

model_dirs = ['i172', 'i173', 'i174', 'i178']
models = {}
tokenizers = {}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

for model_dir in model_dirs:
    model_name = f"model_{model_dir}"
    tokenizer_name = f"tokenizer_{model_dir}"
    model_path = os.path.join(WORK_DIR, 'model', model_dir)
    tokenizers[tokenizer_name] = XLNetTokenizer.from_pretrained(model_path)
    models[model_name] = XLNetForSequenceClassification.from_pretrained(model_path)
    models[model_name].to(device)
    models[model_name].eval()

chgpt_model = config.get('openai-key', 'model')
CER_threshold = float(config.get('threshold', 'CER_threshold'))
POE_threshold = float(config.get('threshold', 'POE_threshold'))
POC_threshold = float(config.get('threshold', 'POC_threshold'))
data_filename = os.path.join(WORK_DIR, 'data', 'metadata')
file_list = os.listdir(data_filename)

file_data = {}
for file in file_list:
    var_name = file[:4]
    with open(os.path.join(data_filename, file), 'r', encoding='utf-8') as f:
        file_data[file[:4]] = json.loads(f.read())


# evidence_filename = os.path.join(WORK_DIR, 'data', 'evidence.json')
# keyword_filename = os.path.join(WORK_DIR, 'data', 'key_word.json')
# with open(evidence_filename, 'r', encoding='utf-8') as f:
#     evi = f.read()
# evidence_json = json.loads(evi)
#
# with open(keyword_filename, 'r') as f:
#     key = f.read()
# keyword_json = json.loads(key)

model = SentenceTransformer('bert-base-nli-mean-tokens')
model1 = SentenceTransformer("paraphrase-MiniLM-L3-v2")
model.to(device)
model1.to(device)
model.eval()
model1.eval()


def predict(content, lesson_id, threshold=CER_threshold):  # threshold .5 for CER model

    # forward pass
    lesson_mapping = {
        "i172": (tokenizers['tokenizer_i172'], models['model_i172']),
        "i173": (tokenizers['tokenizer_i173'], models['model_i173']),
        "i174": (tokenizers['tokenizer_i174'], models['model_i174']),
        "i178": (tokenizers['tokenizer_i178'], models['model_i178'])
    }

    # Check if lesson_id is valid
    if lesson_id in lesson_mapping:
        tokenizer, model = lesson_mapping[lesson_id]
        inputs = tokenizer(content, max_length=50,
                           # return_offsets_mapping=True,
                           padding='max_length',
                           truncation=True, return_tensors="pt")

        # move to gpu
        ids = inputs["input_ids"].to(device)
        idt = inputs["token_type_ids"].to(device)
        mask = inputs["attention_mask"].to(device)
        outputs = model(ids, token_type_ids=idt, attention_mask=mask)
    else:
        log('error', "wrong lesson ID")
        outputs = [0]
    logits = outputs[0]
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))

    flattened_predictions = probs.cpu().detach().numpy()[0]
    label = []
    for i in range(len(flattened_predictions)):
        if flattened_predictions[i] >= threshold:
            label.append(i)

    return flattened_predictions, [3] if label == [] else label


def keyword_match(sentence, support, support_fact, mapping, cer_result):
    if 1 in cer_result[1]:
        supports = []
        for item in support:
            supports.extend(item)
        supports = [s.lower() for s in supports]
        keys = re.findall('|'.join(supports), sentence.lower())

        index = 254
        for i in range(len(support)):
            if set([s.lower() for s in support[i]]) <= set(keys):
                if type(index) == int:
                    index = [i]
                else:
                    index.append(i)

        if index != 254:
            result = [i for i in mapping if any(x in mapping[i] for x in index)]

            return [support_fact[i] for i in result]
        else:
            return 254
    else:
        return 255


def map_keyword(relevant_fact, id="i174"):
    y = relevant_fact[id]['keys']
    support = [k for i in y for k in i['key_word']]  # all sentences support yes or no
    support_fact = [i['describe'] for i in y]  # all categories support yes or no
    support_l = [i['key_word'] for i in y]
    mapping = {v: [support.index(i) for i in k] for v, k in enumerate(support_l)}  # {0:[0,1],1:[2,3],2:[4,5,6]}

    return support, support_fact, mapping


def map_value(relevant_fact):  # mapping sentence in json with each category(target sentence)
    y = relevant_fact['relevant_facts']
    support = [k for i in y for k in i['sentences']]  # all target sentences support yes or no
    sub_categ = [i['describe'] for i in y]  # all categories support yes or no
    support_l = [i['sentences'] for i in y]
    mapping = {v: [support.index(i) for i in k] for v, k in enumerate(support_l)}  # {0:[0,1],1:[2,3],2:[4,5,6]}

    return support, sub_categ, mapping


def similar_cal(sen_list1, sen_list2, lesson_id, sub_categ, cer_result, claim=1, mapping=None, threshold=POE_threshold):
    if mapping is None:
        mapping = {0: [i for i in range(len(sen_list2))]}
    embeddings1 = model.encode(sen_list1, convert_to_tensor=True)
    embeddings2 = model.encode(sen_list2, convert_to_tensor=True)
    cosine_scores1 = util.cos_sim(embeddings1, embeddings2)
    embeddings3 = model1.encode(sen_list1, convert_to_tensor=True)
    embeddings4 = model1.encode(sen_list2, convert_to_tensor=True)
    cosine_scores2 = util.cos_sim(embeddings3, embeddings4)
    cosine_scores = np.array(cosine_scores1.cpu() * .5 + cosine_scores2.cpu() * .5)
    # cosine_scores = np.array(cosine_scores1.cpu())
    max_score = [max(i) for i in cosine_scores]
    num_match = [sum(1 for num in row if num > threshold) for row in cosine_scores]
    tmp = [j for j, _ in
           np.argwhere(
               np.array(
                   cosine_scores) > threshold)]  # which sentence has score higher than threshold, give sentence index

    length = cosine_scores.shape[0]  # how many sentences
    sentence_index = [i for n, i in enumerate(tmp) if i not in tmp[:n]]  # drop duplicate
    index = [int(np.argmax(cosine_scores, axis=1)[i]) for i in
             sentence_index]  # sentence relate to which target sentence, give sentence index sentence index to
    # category index
    sentence_label = [
        255 if i not in sentence_index else [j for j in mapping if index[sentence_index.index(i)] in mapping[j]][0] for
        i in range(
            length)]  # Give a list which show each sentence category. If one sentence has no category, it will show
    # 255. otherwise will show index of category. EX: [255,255,1,255,0]

    fact = []
    evi_key_match = [0] * len(sentence_label)
    if claim != 1:  # only for evidence
        support, support_fact, mapping = map_keyword(file_data, lesson_id)
        for i in range(len(sentence_label)):
            if sentence_label[i] == 255:  # no similarity match
                label = keyword_match(sen_list1[i], support, support_fact, mapping, cer_result[i])

                if type(label) == list:
                    for la in label:
                        if la in sub_categ:
                            fact.append(la)
                            ind = sub_categ.index(la)
                            sentence_label[i] = ind
                            evi_key_match[i] = 1  # the sentence fail the similarity match, has cer evidence score
                            # higher than .5 and match keyword
                        else:
                            sentence_label[i] = la if type(la) != str else 254  # change label to 254 to know we have
                            # cer score higher than .5 but no match keys

    num = list(filter(lambda x: x != 255 and x != 254, sentence_label))
    return index, num, sentence_label, fact, max_score, num_match, evi_key_match


#  match sentence index to category index
def match(value, mapping, support_fact, fact):
    fact_lack = support_fact * 1
    fact_present = []
    for item in value:
        index = [i for i in mapping if item in mapping[i]][0]
        if support_fact[index] not in fact_present:
            fact_present.append(support_fact[index])
            fact_lack.remove(support_fact[index])
    for item in fact:
        if item in fact_lack:
            fact_lack.remove(item)
            fact_present.append(item)
    return fact_present, fact_lack


def cal(lesson_id, sentences):  # match the subcat of evidences
    cer_result = [predict(sentence, lesson_id) for sentence in sentences]
    lesson_type = file_data[lesson_id]['ex']['prompt_type']
    if lesson_type == 1:

        relevant_fact_y = file_data[lesson_id]['ex']['support_yes']
        support_y, sub_categ_y, mapping_y = map_value(relevant_fact_y)

        value_y, sentence_index_y, sentences_label_y, fact_y, max_score_y, num_match_y, key_match_y = similar_cal(
            sentences, support_y, lesson_id, sub_categ_y, cer_result, 0,
            mapping_y)
        fact_present_y, fact_lack_y = match(value_y, mapping_y, sub_categ_y, fact_y)

        relevant_fact_n = file_data[lesson_id]['ex']['support_no']
        support_n, sub_categ_n, mapping_n = map_value(relevant_fact_n)
        value_n, sentence_index_n, sentences_label_n, fact_n, max_score_n, num_match_n, key_match_n = similar_cal(
            sentences, support_n, lesson_id, sub_categ_n, cer_result, 0,
            mapping_n)
        key_match_final = ["a" if pair[1] == 1 else "b" if pair[0] == 1 else "none" for pair in
                           zip(key_match_n, key_match_y)]
        fact_present_n, fact_lack_n = match(value_n, mapping_n, sub_categ_n, fact_n)
        result = {'num_support_a': len(sentence_index_y), 'fact_present_a': fact_present_y,
                  'fact_lack_a': fact_lack_y, 'fact_all_a': sub_categ_y, 'sentence_label_a': sentences_label_y,
                  'num_support_b': len(sentence_index_n), 'fact_present_b': fact_present_n,
                  'fact_lack_b': fact_lack_n, 'fact_all_b': sub_categ_n, 'sentence_label_b': sentences_label_n,
                  'number_general_info': 0, 'sentences_label_ge': [255] * len(sentences), 'number_solution': 0,
                  'sentences_label_sol': [255] * len(sentences), 'claim_label': [255] * len(sentences),
                  'copy': 0, 'reasoning': 0, 'cer_score': [], 'table': [], 'lesson_type': lesson_type}

        general_info = file_data[lesson_id]['ex']['general_info']
        solution = file_data[lesson_id]['ex']['solution']
        y_claim = file_data[lesson_id]['ex']['y_claim']
        n_claim = file_data[lesson_id]['ex']['n_claim']
        if general_info:
            _, sentence_index_ge, sentences_label_ge, fact_g, _, _, _ = similar_cal(sentences, general_info, lesson_id,
                                                                                    ["general_info"], cer_result, 0)
            result['number_general_info'] = len(sentence_index_ge)
            result['sentences_label_ge'] = sentences_label_ge
        if solution:
            _, sentence_index_sol, sentences_label_sol, fact_s, _, _, _ = similar_cal(sentences, solution, lesson_id,
                                                                                      ["solution"], cer_result, 0)
            result['number_solution'] = len(sentence_index_sol)
            result['sentences_label_sol'] = sentences_label_sol
        sentences_label_claim = [255] * len(sentences)
        sentence_new = [i if 0 in predict(i,lesson_id)[1] else " " for i in sentences]  # make non-claim sentence to ' '
        if y_claim or n_claim:
            value_claim, sentence_claim, sentences_label_claim, fact_claim, claim_best_score, _, _ = similar_cal(sentence_new,
                                                                                                  y_claim + n_claim,
                                                                                                  lesson_id,
                                                                                                  ["y_claim",
                                                                                                   "n_claim"],
                                                                                                  cer_result, 1, {
                                                                                                      0: list(
                                                                                                          range(0, len(
                                                                                                              y_claim))),
                                                                                                      1: list(
                                                                                                          range(len(
                                                                                                              y_claim),
                                                                                                              len(
                                                                                                                  y_claim + n_claim)))},
                                                                                                  POC_threshold)

            result['claim'] = [{"claim_id": 'a', "claim": y_claim[0]}, {"claim_id": 'b', "claim": n_claim[0]}]
            result['claim_label'] = sentences_label_claim


        combined = []
        table = []
        for i in range(len(sentences)):
            cer_result_sent = cer_result[i][0]
            if cer_result_sent[2] >= .7:
                result['reasoning'] = 1
            sentence_data = {
                "sentence": sentences[i],
                "claim_score": round(cer_result_sent[0], 3),
                "evidence_score": round(cer_result_sent[1], 3),
                "claim_TRE": "a" if sentences_label_claim[i] == 0 else "b" if sentences_label_claim[i] == 1 else "none",
                "claim_best_score": round(claim_best_score[i], 3),
                "evid_max_score_a": round(max_score_y[i], 3),
                "evid_num_match_a": num_match_y[i],
                "evid_max_score_b": round(max_score_n[i], 3),
                "evid_num_match_b": num_match_n[i],
                "key_word": key_match_final[i]
            }
            table.append(sentence_data)
            combined.append('{}{}'.format(sentences[i], cer_result_sent))
        result['table'] = table
        result['cer_score'] = np.array(cer_result)
        log('info', 'CER result\n%s.' % '\n'.join(combined))
        evid_info = find_largest_number(sentences, lesson_id)
        num_evid = evid_info[0]
        if num_evid / len(sentences) >= .8 or num_evid >= 6 or evid_info[1] == 1:
            result['copy'] = 1

        return result
    elif lesson_type == 2:
        relevant_fact_a = file_data[lesson_id]['ex']['support_a']
        support_a, sub_categ_a, mapping_a = map_value(relevant_fact_a)

        value_a, sentence_index_a, sentences_label_a, fact_a, max_score_a, num_match_a, key_match_a = similar_cal(
            sentences, support_a, lesson_id, sub_categ_a, cer_result, 0,
            mapping_a)
        fact_present_a, fact_lack_a = match(value_a, mapping_a, sub_categ_a, fact_a)

        relevant_fact_b = file_data[lesson_id]['ex']['support_b']
        support_b, sub_categ_b, mapping_b = map_value(relevant_fact_b)
        value_b, sentence_index_b, sentences_label_b, fact_b, max_score_b, num_match_b, key_match_b = similar_cal(
            sentences, support_b, lesson_id, sub_categ_b, cer_result, 0,
            mapping_b)
        fact_present_b, fact_lack_b = match(value_b, mapping_b, sub_categ_b, fact_b)
        try:
            relevant_fact_c = file_data[lesson_id]['ex']['support_c']
            support_c, sub_categ_c, mapping_c = map_value(relevant_fact_c)
            value_c, sentence_index_c, sentences_label_c, fact_c, max_score_c, num_match_c, key_match_c = similar_cal(
                sentences, support_c, lesson_id, sub_categ_c, cer_result, 0,
                mapping_c)
            fact_present_c, fact_lack_c = match(value_c, mapping_c, sub_categ_c, fact_c)
        except:
            value_c, sentence_index_c, sentences_label_c, fact_c, max_score_c, num_match_c, key_match_c, fact_present_c, fact_lack_c = [], [], [], [], [], 0,[0]*len(key_match_a), [], ['all']
        try:
            relevant_fact_d = file_data[lesson_id]['ex']['support_d']
            support_d, sub_categ_d, mapping_d = map_value(relevant_fact_d)
            value_d, sentence_index_d, sentences_label_d, fact_d, max_score_d, num_match_d, key_match_d = similar_cal(
                sentences, support_d, lesson_id, sub_categ_d, cer_result, 0,
                mapping_d)
            fact_present_d, fact_lack_d = match(value_d, mapping_d, sub_categ_d, fact_d)
        except:
            value_d, sentence_index_d, sentences_label_d, fact_d, max_score_d, num_match_d, key_match_d, fact_present_d, fact_lack_d = [], [], [], [], [], 0, [0]*len(key_match_a), [], ['all']
        # key_match_final = ["a" if pair[1] == 1 else "b" if pair[0] == 1 else "c" if pair[2] == 1 else "d" if pair[3] == 1 else "none" for pair in
        #                    zip(key_match_b, key_match_a, key_match_c, key_match_d)]
        key_match_final = []
        for pair in zip(key_match_b, key_match_a, key_match_c, key_match_d):
            res = []
            if pair[1] == 1:
                res.append('a')
            if pair[0] == 1:
                res.append('b')
            if pair[2] == 1:
                res.append('c')
            if pair[3] == 1:
                res.append('d')
            if not res:
                res.append('none')
            key_match_final.append(res)

        result = {'num_support_a': len(sentence_index_a), 'fact_present_a': fact_present_a,
                  'sentence_label_a': sentences_label_a, 'fact_lack_a': fact_lack_a,
                  'num_support_b': len(sentence_index_b), 'fact_present_b': fact_present_b,
                  'sentence_label_b': sentences_label_b, 'fact_lack_b': fact_lack_b,
                  'num_support_c': len(sentence_index_c), 'fact_present_c': fact_present_c,
                  'sentence_label_c': sentences_label_c, 'fact_lack_c': fact_lack_c,
                  'num_support_d': len(sentence_index_d), 'fact_present_d': fact_present_d,
                  'sentence_label_d': sentences_label_d, 'fact_lack_d': fact_lack_d,
                  'number_general_info': 0, 'sentences_label_ge': [255] * len(sentences), 'number_solution': 0,
                  'sentences_label_sol': [255] * len(sentences), 'claim_label': [255] * len(sentences),
                  'copy': 0, 'reasoning': 0, 'cer_score': [], 'table': [], 'lesson_type': lesson_type}

        general_info = file_data[lesson_id]['ex']['general_info']
        solution = file_data[lesson_id]['ex']['solution']
        a_claim = file_data[lesson_id]['ex']['a_claim']
        b_claim = file_data[lesson_id]['ex']['b_claim']
        try:
            c_claim = file_data[lesson_id]['ex']['c_claim']
        except:
            c_claim = []
        try:
            d_claim = file_data[lesson_id]['ex']['d_claim']
        except:
            d_claim = []
        if general_info:
            _, sentence_index_ge, sentences_label_ge, fact_g, _, _, _ = similar_cal(sentences, general_info, lesson_id,
                                                                                    ["general_info"], cer_result, 0)
            result['number_general_info'] = len(sentence_index_ge)
            result['sentences_label_ge'] = sentences_label_ge
        if solution:
            _, sentence_index_sol, sentences_label_sol, fact_s, _, _, _ = similar_cal(sentences, solution, lesson_id,
                                                                                      ["solution"], cer_result, 0)
            result['number_solution'] = len(sentence_index_sol)
            result['sentences_label_sol'] = sentences_label_sol
        sentences_label_claim = [255] * len(sentences)
        sentence_new = [i if 0 in predict(i, lesson_id)[1] else " " for i in sentences]  # make non-claim sentence to ' '
        if a_claim or b_claim or c_claim or d_claim:
            value_claim, sentence_claim, sentences_label_claim, fact_claim, _, _, _ = similar_cal(sentence_new,
                                                                                                  a_claim + b_claim + c_claim + d_claim,
                                                                                                  lesson_id,
                                                                                                  ["a_claim",
                                                                                                   "b_claim",
                                                                                                   "c_claim",
                                                                                                   "d_claim"],
                                                                                                  cer_result, 1, {
                                                                                                      0: list(
                                                                                                          range(0, len(
                                                                                                              a_claim))),
                                                                                                      1: list(
                                                                                                          range(len(
                                                                                                              a_claim),
                                                                                                              len(
                                                                                                                  a_claim + b_claim))),
                                                                                                      2: list(
                                                                                                          range(len(
                                                                                                              a_claim + b_claim),
                                                                                                              len(
                                                                                                                  a_claim + b_claim + c_claim))),
                                                                                                      3: list(
                                                                                                          range(len(
                                                                                                              a_claim + b_claim + c_claim),
                                                                                                              len(
                                                                                                                  a_claim + b_claim + c_claim + d_claim)))},
                                                                                                  POC_threshold)

            result['claim'] = [{"claim_id": 'a', "claim": a_claim[0]}, {"claim_id": 'b', "claim": b_claim[0]}, {"claim_id": 'c', "claim": c_claim[0] if c_claim else ''},  {"claim_id": 'd', "claim": d_claim[0] if d_claim else ''} ]
            result['claim_label'] = sentences_label_claim

        combined = []
        table = []
        for i in range(len(sentences)):
            cer_result_sent = cer_result[i][0]
            if cer_result_sent[2] >= .5:
                result['reasoning'] = 1
            sentence_data = {
                "sentence": sentences[i],
                "claim_score": round(cer_result_sent[0], 3),
                "evidence_score": round(cer_result_sent[1], 3),
                "claim_TRE": "a" if sentences_label_claim[i] == 0 else "b" if sentences_label_claim[i] == 1 else "c" if sentences_label_claim[i] == 2 else "d" if sentences_label_claim[i] == 3 else "none",
                "evid_max_score_a": round(max_score_a[i], 3),
                "evid_num_match_a": num_match_a[i],
                "evid_max_score_b": round(max_score_b[i], 3),
                "evid_num_match_b": num_match_b[i],
                "evid_max_score_c": round(max_score_c[i], 3) if max_score_c else 0,
                "evid_num_match_c": num_match_c[i] if num_match_c else 0,
                "evid_max_score_d": round(max_score_d[i], 3) if max_score_d else 0,
                "evid_num_match_d": num_match_d[i] if num_match_d else 0,
                "key_word": ','.join(key_match_final[i])
            }
            table.append(sentence_data)
            combined.append('{}{}'.format(sentences[i], cer_result_sent))
        result['table'] = table
        result['cer_score'] = np.array(cer_result)
        log('info', 'CER result\n%s.' % '\n'.join(combined))
        evid_info = find_largest_number(sentences, lesson_id)
        num_evid = evid_info[0]
        if num_evid / len(sentences) >= .8 or num_evid >= 6 or evid_info[1] == 1:
            result['copy'] = 1

        return result
    else:  # right now we don't have another type, save this for future
        return {'general_info': 0}
