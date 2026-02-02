from flask import Flask, request, jsonify, send_from_directory
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
import random
import os

app = Flask(__name__)
model = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")

def compute_benchmark(word):
    goal_embedding = model.process([{"text": word}])
    benchmark_embedding = model.process(map(lambda x: {"text": x}, dictionary))
    return goal_embedding, sorted(zip(dictionary, *((goal_embedding @ benchmark_embedding.T) * 1000)), key=lambda x: -int(x[1]))

def get_benchmark_index(benchmark, word):
    for i, x in enumerate(benchmark):
        if word==x[0]:
            return i

dictionary = list(map(str.strip, open("wordlists/top-1000-nouns.txt").readlines()))[:10]
correct_word = random.choice(dictionary)
goal_embedding, benchmark = compute_benchmark(correct_word)

@app.route("/api/score", methods=["POST"])
def get_score():
    data = request.get_json()
    guess = data["guess"]

    guess_embedding = model.process([{"text": guess}])
    score = (goal_embedding @ guess_embedding.T)
    pos = get_benchmark_index(benchmark, guess) if guess in dictionary else None

    return jsonify({"score": int(score[0][0] * 1000), "pos": pos}), 200

@app.route("/api/correct-word", methods=["GET"])
def new_word():
    return jsonify({"correct-word": correct_word})

#@app.route('/', methods=["GET"])
#def index():
#    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 1337)))
