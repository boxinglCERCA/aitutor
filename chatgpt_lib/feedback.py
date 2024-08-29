import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from chatgpt_lib.log import log
from nltk import word_tokenize
import numpy as np
import os
from chatgpt_lib import WORK_DIR
import configparser
import json
import re

# load and set our key
config = configparser.ConfigParser()
config_path = os.path.join(WORK_DIR, 'config', 'config.ini')
config.read(config_path)
openai.api_key = config.get('openai-key', 'key')
chgpt_model = config.get('openai-key', 'model')
port = config.get('port', 'port')
host = config.get('port', 'host')
essay_filename = os.path.join(WORK_DIR, 'data', 'essay.json')

app = Flask(__name__)  ####
CORS(app)


def to_text(label, response, essay, question, content_list):
    if len(content_list) == 1 and len(word_tokenize(response)) > 20:
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "content": "The response appears to contain too little content or punctuation for evaluation, "
                                   "or unusual syntax, punctuation or grammar and should be reviewed and improved for "
                                   "future automated feedback."
                    }
                }
            ]
        }
    if len(content_list) < 3 and len(word_tokenize(response)) <= 15:
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "content": "The response is insufficient for automated evaluation and feedback."
                    }
                }
            ]
        }
    if 0 in label["claim_label"] or 1 in label["claim_label"]:  # 0 is A claim, 1 is B claim
        if 0 in label["claim_label"] and 1 in label["claim_label"]:  # two opposite claims
            claim = "There are two conflicting claims in the response, which makes claim unclear. "
        else:  # one clear claim
            claim = 'There is a solid claim. '
    else:  # no claim
        claim = "There is no clear claim. "
    if label["fact_present_a"] or label["fact_present_b"]:  # there is at least one evidence
        if label['copy'] != 1:  # not Excessive copying
            if 0 in label["claim_label"] and 1 in label["claim_label"]:  # two opposite claims what is the evidence
                if label["fact_present_a"] and label[
                    "fact_present_b"]:  # Both categories of claim and evidence are found
                    evid = 'Your response has evidence and claims for both sides of the issue; be sure your claim is ' \
                           'clear and meets the demands of the prompt. '
                else:
                    evid = 'you need to make sure you provide the right evidences to support you claim'
            elif 0 in label["claim_label"]:  # A claim only case
                if len(label["fact_present_a"]) >= 3:  # 3 more evidences good job
                    return jsonify({"content": "You did a good job of giving a solid claim and providing evidences. ",
                                    "role": "assistant"})
                elif len(label["fact_present_a"]) < len(label["fact_present_b"]):  # wrong evidence
                    evid = "Your response presents a claim but appears to cite evidence or a different claim."
                else:  # right evidences but less than 3
                    evid_lack_list = label["fact_lack_a"]
                    evid = 'You provide ' + ",".join(
                        ["'" + item + "'" for item in
                         label["fact_present_a"]]) + " as evidence point. " + "You can add " \
                                                                              "more evidence " \
                                                                              "from text to " \
                                                                              "support " \
                                                                              "your claim " \
                                                                              "such as " + \
                           ", ".join(
                               ["'" + item + "'" for item in evid_lack_list])
            else:  # B claim only case
                if len(label["fact_present_b"]) >= 3:  # 3 more evidences good job
                    return jsonify({"content": "You did a good job of giving a solid claim and providing evidences. ",
                                    "role": "assistant"})
                elif len(label["fact_present_b"]) < len(label["fact_present_b"]):  # wrong evidence

                    evid = "Your response presents a claim but appears to cite evidence or a different claim."
                else:  # right evidences but less than 3
                    evid_lack_list = label["fact_lack_b"]
                    evid = 'You provide ' + ",".join(["'" + item + "'" for item in label[
                        "fact_present_b"]]) + " as evidence point. " + "You can add more " \
                                                                       "evidence from text to " \
                                                                       "support your claim " \
                                                                       "such as " + ", ".join(
                        ["'" + item + "'" for item in evid_lack_list])
        else:  # Excessive copying
            evid = "Excessive copying of passage content. You need to express with your language and give reasoning " \
                   "why it supports your claim "
    else:  # no evidence
        evid = "There is no evidence. add evidences from text support you claim"

    overall_content = essay + "\n" + "Student response: '" + response + " '\n" + "ThinkCERCA AI: '" + claim + evid + "'\n" + question

    log('info', 'Chatgpt input is: \n %s.' % overall_content)

    completion = openai.ChatCompletion.create(
        model=chgpt_model,  # this is "ChatGPT" $0.002 per 1k tokens
        messages=[{"role": "system", "content": "You are a elementary teacher"},
                  {"role": "user", "content": overall_content}])
    output = completion["choices"][0]["message"]['content']
    log('info', 'Chatgpt output is :\n%s.' % output + "\n-----------------------------------------")
    return completion


def to_feedback(label, response, content_list):
    Response_c = ''
    CER_q = ''
    SOE = ''
    SOC = ''
    evid_lack_list = []
    completion = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "content": ""
                }
            }
        ]
    }

    if len(content_list) == 1 and len(word_tokenize(response)) > 30:
        Response_c += "Found Condition 15, the response appears to contain too little content or " \
                      "punctuation for evaluation, or unusual syntax, punctuation " \
                      "or grammar and should be reviewed and improved for future " \
                      "automated feedback. "
        # completion["choices"][0]["message"][
        #     'content'] = 'Strength of claim: ' + " " + '\n' + 'Response characteristics: ' + Response_c + '\n' + 'CER Qualities: ' + " " + '\n' + 'Sufficiency of Evidence: ' + ""
        # return completion
    if len(content_list) < 3 or len(word_tokenize(response)) <= 15:

        Response_c += "Found Condition 16, the response is insufficient for automated evaluation and " \
                      "feedback. "
        # completion["choices"][0]["message"][
        #     'content'] = 'Strength of claim: ' + " " + '\n' + 'Response characteristics: ' + Response_c + '\n' + 'CER Qualities: ' + " " + '\n' + 'Sufficiency of Evidence: ' + ""
        # return completion
    # feedback for claim for only a, b claim
    max_claim_score = label['cer_score'][:, 0]

    first_column_values = [arr[0] for arr in max_claim_score]
    second_column_values = [arr[1] for arr in max_claim_score]
    max_value = np.max(first_column_values)
    max_evid = np.max(second_column_values)

    if label['lesson_type'] == 1:

        if 0 in label["claim_label"] or 1 in label["claim_label"]:  # 0 is A claim, 1 is B claim
            SOC = 'Found Condition 0, makes a solid claim responsive to the demands of the prompt. '
            if 0 in label["claim_label"] and 1 in label["claim_label"]:  # two opposite claims

                CER_q += "Found Condition 14, both claims seem to be supported when the prompt requires a choice. "
            else:  # one clear claim
                claim = 'Makes a solid claim responsive to the demands of the prompt. '
        else:  # no claim

            if max_value >= .8:
                SOC = 'Found Condition 1, your response appears to have a solid claim. '
                CER_q += 'Found Condition 12, Your choice of claim is not recognized or your response may have ' \
                         'evidence and claims for both sides of the issue; be sure your claim is clear. '
            elif .8 > max_value >= .75:
                SOC = 'Found Condition 2, your response appears to have a solid claim. '
                CER_q += 'Found Condition 12, Your choice of claim is not recognized or your response may have ' \
                         'evidence and claims for both sides of the issue; be sure your claim is clear. '
            elif .75 > max_value >= .55:
                SOC = 'Found Condition 3, your response may or may not make a sufficient claim.  Make sure your claim ' \
                      'is clear and responds to the demands of the prompt. '
                CER_q += 'Found Condition 12, Your choice of claim is not recognized or your response may have ' \
                         'evidence and claims for both sides of the issue; be sure your claim is clear. '
            elif .55 > max_value >= .5:
                SOC = 'Found Condition 4, your response does not appear to make a claim that is responsive to the ' \
                      'prompt. Check the noted sentence. Make sure your claim addresses the demands of the prompt '
            else:
                SOC = 'Found Condition 5, claim not found.'
                if max_evid > 0.5 or label["fact_present_a"] or label["fact_present_b"]:
                    CER_q += "Found Condition 13, your response seems to have evidence but no claim. "

        if label["fact_present_a"] or label["fact_present_b"]:  # there is at least one evidence

            if label['copy'] == 1:  # Excessive copying
                Response_c += "Found Condition 6, review for excessive copying of passage content. "
            if 0 in label["claim_label"] and 1 in label["claim_label"]:  # two opposite claims what is the evidence
                if label["fact_present_a"] and label[
                    "fact_present_b"]:  # Both categories of claim and evidence are found
                    CER_q += 'Found Condition 12, Your choice of claim is not recognized or your response may have ' \
                             'evidence and claims for both sides of the issue; be sure your claim is clear. '
                else:
                    evid = 'you need to make sure you provide the right evidences to support you claim '
            elif 0 in label["claim_label"]:  # A claim only case
                if len(label["fact_present_a"]) >= 3 or len(label["fact_lack_a"]) == 0:  # 3 more evidences good job
                    SOE = "Found Condition 7, excellent range of evidence to support the claim. "

                elif len(label["fact_present_a"]) == 0 and len(label["fact_present_b"]) > 0:  # wrong evidence
                    CER_q += "Found Condition 10, your response presents a claim but appears to cite evidence for a " \
                             "different claim. "
                else:  # right evidences but less than 3
                    evid_lack_list = label["fact_lack_a"]
                    evid = 'You provide ' + ",".join(
                        ["'" + item + "'" for item in
                         label["fact_present_a"]]) + " as evidence point. " + "You can add " \
                                                                              "more evidence " \
                                                                              "from text to " \
                                                                              "support " \
                                                                              "your claim " \
                                                                              "such as " + \
                           ", ".join(
                               ["'" + item + "'" for item in evid_lack_list])
                    if len(label["fact_present_a"]) >= 2:
                        SOE = 'Found Condition 8, uses a range of evidence to support the claim. '
                    else:
                        SOE = 'Found Condition 9, uses some evidence to support the claim. More evidence is available. '
            elif 1 in label["claim_label"]:  # B claim only case
                if len(label["fact_present_b"]) >= 3 or len(label["fact_lack_b"]) == 0:  # 3 more evidences good job
                    SOE = "Found Condition 7, excellent range of evidence to support the claim. "

                elif len(label["fact_present_b"]) == 0 and len(label["fact_present_a"]) > 0:  # wrong evidence

                    CER_q += "Found Condition 10, your response presents a claim but appears to cite evidence for a " \
                             "different claim. "
                else:  # right evidences but less than 3
                    evid_lack_list = label["fact_lack_b"]
                    evid = 'You provide ' + ",".join(["'" + item + "'" for item in label[
                        "fact_present_b"]]) + " as evidence point. " + "You can add more " \
                                                                       "evidence from text to " \
                                                                       "support your claim " \
                                                                       "such as " + ", ".join(
                        ["'" + item + "'" for item in evid_lack_list])
                    if len(label["fact_present_b"]) >= 2:
                        SOE = 'Found Condition 8, uses a range of evidence to support the claim. '
                    else:
                        SOE = 'Found Condition 9, uses some evidence to support the claim. More evidence is available. '

        elif max_evid > .5:
            pass
        else:
            SOE = "Found Condition 18, Unable to detect evidence relevant to the demands of the prompt. "

    elif label['lesson_type'] == 2:
        # print(label)
        max_claim_score = label['cer_score'][:, 0]
        first_column_values = [arr[0] for arr in max_claim_score]
        max_value = np.max(first_column_values)
        if any(x in label["claim_label"] for x in [0, 1, 2, 3]):  # 0 is A claim, 1 is B claim
            if sum(1 for x in [0, 1, 2, 3] if x in set(label["claim_label"])) == 1:
                SOC = 'Found Condition 0, makes a solid claim responsive to the demands of the prompt. '
            else:
                SOC = 'Found Condition 0, makes a solid claim responsive to the demands of the prompt. '
                Response_c += 'Found Condition 14, Mutiple claims may be present. '
        else:  # no CER claim

            if max_value >= .8:
                SOC = 'Found Condition 1, your response appears to have a solid claim.'
                CER_q = 'Found Condition 12, Your choice of claim is not recognized or your response may have ' \
                        'evidence and claims for both sides of the issue; be sure your claim is clear. '
            elif .8 > max_value >= .75:
                SOC = 'Found Condition 2, Your response appears to have a claim.'
                CER_q = 'Found Condition 12, Your choice of claim is not recognized or your response may have ' \
                        'evidence and claims for both sides of the issue; be sure your claim is clear. '
            elif .75 > max_value >= .5:
                SOC = 'Found Condition 3, your response may or may not make a sufficient claim.  Make sure your claim ' \
                      'is clear and responds to the demands of the prompt. '
                CER_q = 'Found Condition 12, Your choice of claim is not recognized or your response may have ' \
                        'evidence and claims for both sides of the issue; be sure your claim is clear. '
            elif .5 > max_value >= .45:
                SOC = 'Found Condition 4, your response does not appear to make a claim that is responsive to the ' \
                      'prompt. Check the noted sentence. Make sure your claim addresses the demands of the prompt '
            else:
                SOC = 'Found Condition 5, claim not found.'
        if label["fact_present_a"] or label["fact_present_b"] or label["fact_present_c"] or label[
            "fact_present_d"]:  # there is at least one evidence
            if (len(label["fact_present_a"]) >= 3 and 0 in label["claim_label"]) or (
                    len(label["fact_lack_a"]) == 0 and 0 in label["claim_label"]) or (len(
                label["fact_present_b"]) >= 3 and 1 in label["claim_label"]) or (
                    len(label["fact_lack_b"]) == 0 and 1 in label["claim_label"]) or (len(
                label["fact_present_c"]) >= 3 and 2 in label["claim_label"]) or (
                    len(label["fact_lack_c"]) == 0 and 2 in label["claim_label"]) or (len(
                label["fact_present_d"]) >= 3 and 3 in label["claim_label"]) or (
                    len(label["fact_lack_d"]) == 0 and 3 in label["claim_label"]):
                SOE = "Found Condition 7, excellent range of evidence to support the claim. "

            elif (len(label["fact_present_a"]) >= 2 and 0 in label["claim_label"]) or (
                    len(label["fact_present_b"]) >= 2 and 1 in label["claim_label"]) or (
                    len(label["fact_present_c"]) >= 2 and 2 in label["claim_label"]) or (
                    len(label["fact_present_d"]) >= 2 and 3 in label["claim_label"]):
                print(label["fact_present_c"])
                SOE = 'Found Condition 8, uses a range of evidence to support the claim. '
            elif (len(label["fact_present_a"]) >= 1 and 0 in label["claim_label"]) or (
                    len(label["fact_present_b"]) >= 1 and 1 in label["claim_label"]) or (
                    len(label["fact_present_c"]) >= 1 and 2 in label["claim_label"]) or (
                    len(label["fact_present_d"]) >= 1 and 3 in label["claim_label"]):
                SOE = 'Found Condition 9, uses some evidence to support the claim. More evidence is available. '

            if label['copy'] == 1:  # Excessive copying
                Response_c += "Found Condition 6, review for excessive copying of passage content. "

            if 0 in label["claim_label"]:  #
                if len(label["fact_present_a"]) == 0:  # wrong evidence
                    CER_q += "Found Condition 10, your response presents a claim but appears to cite evidence for a " \
                             "different claim. "
                else:  # right evidences but less than 3
                    evid_lack_list += label["fact_lack_a"]
                    evid = 'You provide ' + ",".join(
                        ["'" + item + "'" for item in
                         label["fact_present_a"]]) + " as evidence point. " + "You can add " \
                                                                              "more evidence " \
                                                                              "from text to " \
                                                                              "support " \
                                                                              "your claim " \
                                                                              "such as " + \
                           ", ".join(
                               ["'" + item + "'" for item in evid_lack_list])
            if 1 in label["claim_label"]:  # B claim only case
                if len(label["fact_present_b"]) == 0 and CER_q == '':  # wrong evidence
                    CER_q += "Found Condition 10, your response presents a claim but appears to cite evidence for a " \
                             "different claim. "
                else:  # right evidences but less than 3
                    evid_lack_list += label["fact_lack_b"]
                    evid = 'You provide ' + ",".join(["'" + item + "'" for item in label[
                        "fact_present_b"]]) + " as evidence point. " + "You can add more " \
                                                                       "evidence from text to " \
                                                                       "support your claim " \
                                                                       "such as " + ", ".join(
                        ["'" + item + "'" for item in evid_lack_list])

            elif 2 in label["claim_label"]:  # c claim only case
                if len(label["fact_present_c"]) == 0 and CER_q == '':  # wrong evidence
                    CER_q += "Found Condition 10, your response presents a claim but appears to cite evidence for a " \
                             "different claim. "
                else:  # right evidences but less than 3
                    evid_lack_list += label["fact_lack_c"]
                    evid = 'You provide ' + ",".join(["'" + item + "'" for item in label[
                        "fact_present_b"]]) + " as evidence point. " + "You can add more " \
                                                                       "evidence from text to " \
                                                                       "support your claim " \
                                                                       "such as " + ", ".join(
                        ["'" + item + "'" for item in evid_lack_list])

            elif 3 in label["claim_label"]:  # d claim only case
                if len(label["fact_present_d"]) == 0 and CER_q == '':  # wrong evidence
                    CER_q += "Found Condition 10, your response presents a claim but appears to cite evidence for a " \
                             "different claim. "
                else:  # right evidences but less than 3
                    evid_lack_list += label["fact_lack_d"]
                    evid = 'You provide ' + ",".join(["'" + item + "'" for item in label[
                        "fact_present_b"]]) + " as evidence point. " + "You can add more " \
                                                                       "evidence from text to " \
                                                                       "support your claim " \
                                                                       "such as " + ", ".join(
                        ["'" + item + "'" for item in evid_lack_list])
            elif max_claim_score < .5:

                CER_q += "Found Condition 13, your response seems to have evidence but no claim. "

        else:  # no evidence
            SOE = "Found Condition 18, Unable to detect evidence relevant to the demands of the prompt. "
    if label['reasoning'] == 0:
        CER_q += 'Found Condition 11, The response should connect the evidence to the claim with explicit reasoning. '
    completion["choices"][0]["message"][
        'content'] = 'Strength of claim: ' + SOC + '\n' + 'Response characteristics: ' + Response_c + '\n' + 'CER Qualities: ' + CER_q + '\n' + 'Sufficiency of Evidence: ' + SOE

    return completion


def find_conditions(text):
    index_num = [0] * 21
    condition_pattern = r'Found Condition (\d+)'
    conditions = re.findall(condition_pattern, text)
    for i in conditions:
        index_num[int(i)] = 1
    return index_num
