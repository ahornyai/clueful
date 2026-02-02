from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
import torch.linalg as la
import random

dictionary = list(map(str.strip, open("wordlists/top-1000-nouns.txt").readlines()))[:10]
correct_word = random.choice(dictionary)

model = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")
goal_embedding = model.process([{"text": correct_word}])
benchmark_embedding = model.process(map(lambda x: {"text": x}, dictionary))
benchmark = sorted(zip(dictionary, *((goal_embedding @ benchmark_embedding.T) * 1000)), key=lambda x: -int(x[1]))

def get_benchmark_index(word):
    for i, x in enumerate(benchmark):
        if word==x[0]:
            return i

print("Guess the secret word...")

while True:
    guess = input("> ")

    if guess == "giveup":
        print("you gave up!")
        print("the correct word was:", correct_word)
        break
    
    guess_embedding = model.process([{"text": guess}])
    score = (goal_embedding @ guess_embedding.T)

    if not (guess in dictionary):
        print("the given word is not in the dictionary, score:", int(score[0][0] * 1000))
    else:
        print(f"the given word is placed at {get_benchmark_index(guess)+1} in the benchmark, score:", int(score[0][0] * 1000))

    if score == 1000:
        print("you guessed the word:", correct_word)
        break