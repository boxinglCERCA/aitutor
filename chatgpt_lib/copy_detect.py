from chatgpt_lib import WORK_DIR
from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher
from flask import Flask, request, jsonify
from chatgpt_lib.log import log
from chatgpt_lib.se_parsing import TextParsingV3
import configparser
import json
import os
import nltk
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_distances

model_path = os.path.join(WORK_DIR, 'model/embedding', 'GoogleNews-vectors-negative300.bin.gz')
embedding_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
config = configparser.ConfigParser()
config_path = os.path.join(WORK_DIR, 'config', 'config.ini')
config.read(config_path)
min_sent_count = int(config.get('main', 'min_sent_count'))
copy_threshold = float(config.get('threshold', 'copy_threshold'))

data_filename = os.path.join(WORK_DIR, 'data', 'metadata')
file_list = os.listdir(data_filename)

file_data = {}
for file in file_list:
    var_name = file[:4]
    with open(os.path.join(data_filename, file), 'r', encoding='utf-8') as f:
        file_data[file[:4]] = json.loads(f.read())

stop_words = set(stopwords.words('english'))


def sentence_embedding(sentence):
    words = nltk.word_tokenize(sentence.lower())
    embedding = []
    for word in words:
        if word in embedding_model and word not in stop_words:
            embedding.append(embedding_model[word])
    if len(embedding) == 0:
        return embedding_model["abc"]
    else:
        return np.mean(embedding, axis=0)


def c_s(sentence1, sentence2):
    embedding1 = sentence_embedding(sentence1)
    embedding2 = sentence_embedding(sentence2)
    similarity = 1 - cosine_distances([embedding1], [embedding2])
    return similarity[0][0]


def sentence_similarity(s1, s2):  # python copy detecting
    return SequenceMatcher(None, s1, s2).ratio()


def edit_dist(s1, s2):  # edit distance
    tokens1 = s1.split(" ")
    tokens2 = s2.split(" ")
    len1, len2 = len(tokens1), len(tokens2)
    d = [[0 for j in range(len2 + 1)] for i in range(len1 + 1)]

    d[0][0] = 0

    for i in range(1, len1 + 1):
        d[i][0] = i
    for i in range(1, len2 + 1):
        d[0][i] = i

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (0 if tokens1[i - 1] == tokens2[j - 1] else 1))

    nlevenshteinDist = (d[len1][len2]) / max(len1, len2)

    return round(1 - nlevenshteinDist, 4)


def find_largest_number(list1, lesson_id, threshold=copy_threshold):
    row_match = 0
    passage = file_data[lesson_id]['ex']['essay'].lstrip("Essay : ")
    passage = TextParsingV3([passage])
    passage_list = passage.sent_2d_list[0]
    # print(passage.new_sent_2d_list)
    # print(passage_list)
    # create matrix by multiplying numbers from list1 and list2 together
    # matrix = [[edit_dist(i, j) for j in list2] for i in list1]  # using edit_dist
    # matrix = [[sentence_similarity(i, j) for j in list2] for i in list1]  # using python sentence similarity
    matrix = [[c_s(i, j) for j in passage_list] for i in list1]
    result = []

    for i, row in enumerate(matrix):
        largest_number = row[0]
        largest_index = 0
        for j, number in enumerate(row):
            if number > largest_number:
                largest_number = number
                largest_index = j

        result.append((i, largest_index, round(float(largest_number), 4)))

    log('info', f'Exemplar result\n{result}\n')
    num = 0
    count = 1
    for i in range(len(result)):
        if result[i][2] >= threshold:
            num += 1
            if i > 0 and result[i-1][2] >= threshold:
                count += 1
            else:
                count = 1
        if count >= 4:
            row_match = 1
    return [num, row_match]


def log_error(error_msg):
    log('error', f" {error_msg}.")


app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def run():
    if request.method == 'POST':  ####POST
        # data_fz = json.loads(request.get_data().decode('utf-8')) ####get data
        data_fz = request.get_json()
        # print(data_fz)

        if data_fz is not None:
            # data_fz = request.to_dict()
            lesson_id = data_fz['lesson_id']
            response = data_fz['content']

            content = TextParsingV3([response.replace('\xa0', ' ')])
            content_list = content.sent_2d_list
            if content.quotes_failed:
                log_error('Double quotes mismatch')
                return jsonify({'Mess': 'Double quotes mismatch', 'type': 'Error'})
            if content.no_punctuation:
                log_error('No punctuation')  # still process it
                return jsonify({'Mess': 'No punctuation', 'type': 'Error'})
            if not content.sent_number:
                log_error('No sentences found')
                return jsonify({'Mess': 'No sentences found', 'type': 'Error'})
                # check minimum sentence count
            if content.sent_number < min_sent_count:
                log_error("No paragraphs found")
                return jsonify({'Mess': 'response is too short', 'type': 'Error'})
            if not content.words_number:
                log_error('No words found')
                return jsonify({'Mess': 'No words found', 'type': 'Error'})
            # print(content)
            passage = file_data[lesson_id]['ex']['passage']
            passage_list = TextParsingV3([passage]).sent_2d_list
            results = find_largest_number(content_list[0], passage_list[0])

            # log('info', results)
            # log('info', "============================================================================================")
        else:
            return jsonify({'Bj': -1, 'Mess': '', 'type': 'Error'})  ####return -1 if no data
    else:
        return jsonify({'Bj': -2, 'Mess': '', 'type': 'Error'})  #### return -2 if not right format

    return jsonify({"result": results})

#
# if __name__ == "__main__":
#     app.run(host=host, port=port, debug=False)
