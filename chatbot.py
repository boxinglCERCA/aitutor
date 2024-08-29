import openai
from flask import Flask, request, jsonify, render_template,session


import os
from chatgpt_lib import WORK_DIR
import configparser
import json

config = configparser.ConfigParser()
config_path = os.path.join(WORK_DIR, 'config', 'config.ini')
config.read(config_path)
chgpt_model = config.get('openai-key', 'model')
openai.api_key = config.get('openai-key', 'key')
port = config.get('port', 'port')
host = config.get('port', 'host')
essay_filename = os.path.join(WORK_DIR, 'data', 'essay.json')
with open(essay_filename, 'r', encoding='utf8') as f:
    essays = f.read()
essays_json = json.loads(essays)
lesson_filename = os.path.join(WORK_DIR, 'data', 'lesson.json')
with open(lesson_filename, 'r', encoding='utf8') as f:
    lessons = f.read()
lesson_json = json.loads(lessons)
data_filename = os.path.join(WORK_DIR, 'data', 'metadata')
file_list = os.listdir(data_filename)

file_data = {}
for file in file_list:
    var_name = file[:4]
    with open(os.path.join(data_filename, file), 'r', encoding='utf-8') as f:
        file_data[file[:4]] = json.loads(f.read())

text = ''
essay = ''
result_text = ''

# 创建Flask应用程序
app = Flask(__name__)
app.secret_key = 'super secret key'

@app.route('/')
def index():
    session.pop('message_history', None)
    return render_template('tutor.html')

@app.route('/clear_history', methods=['GET'])
def clear_history():
    global message_history1
    message_history1 = []  # Clear the conversation history

    return jsonify({'message': 'Conversation history cleared'})


@app.route('/chatbot')
def chatbot_button():
    return render_template('chatbot.html')


@app.route('/tutor')
def tutor_button():
    session.pop('message_history', None)
    return render_template('tutor.html')


@app.route('/tutoring_message', methods=['POST'])
def tutoring_message():

    user_message = request.form['message']
    essay = request.form.get("essay")
    question = request.form.get("question")
    message_history = session.get('message_history', [])

    persona = """You are a writing coach. Your role is to guide a student through the process of writing an essay by 
    encouraging critical thinking through a step-by-step conversation. If a student 
    responds with just a word or a phrase, remind them to write complete sentences.  if there are grammar mistakes, please mention that as well."""
    instruction = """Follow these instructions: 

1. Ask short, targeted questions to help the student identify evidence that supports their claim.
2. also check grammar and insure they use right sentence.
3. Ensure the student uses reasoning to explain how and why this evidence supports your claim. You can also provide examples to clarify this concept if needed.
4. Do not provide direct answers. Instead, prompt the student to think deeply about their responses.
5. Ask the student to explain how their points connect to the claim and the evidence from the text.
6. If the student uses evidence not found in the essay, ask why they chose that evidence and guide them to find similar sources within the text.
7. Finally, ask the student to compile all their points into a cohesive response. 
8. Once the student has completed their response, provide positive feedback and end the tutoring session.
Throughout the conversation, focus on guiding the student to build a strong, well-supported essay."""
    if not message_history:
        message_history.extend([
            {"role": "system", "content": persona},
            {"role": "user", "content": instruction},
            {"role": "user", "content": essay},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Can you state your claim to answer the question?"}
        ])

    message_history.append({"role": "user", "content": user_message.replace('\xa0', ' ')})
    # Call the OpenAI Chat Completion API
    result = openai.ChatCompletion.create(
        model=chgpt_model,
        temperature=0.2,
        messages=message_history,
        max_tokens = 300
    )

    bot_response = result["choices"][0]["message"]["content"]
    message_history.append({"role": "assistant", "content": bot_response})
    session['message_history'] = message_history
    # Prepare the response to send back to the client
    response = {
        "message": bot_response
    }

    return jsonify(response)


@app.route("/find_lesson", methods=["POST"])
def find_lesson():
    lesson_id = request.form.get("lessonId")
    if lesson_id in ['i172','i173','i174']:
        lesson = lesson_json[lesson_id]
        return jsonify({"essay": lesson["essay"], "question": lesson["question"]})
    else:
        return jsonify({"error": "Lesson not found."}), 404


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=False)