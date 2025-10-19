# app/bigram_model.py
import random

class BigramModel:
    def __init__(self, text: str):
        words = text.split()
        self.bigram_dict = {}
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            self.bigram_dict.setdefault(w1, []).append(w2)

    def generate(self, start_word: str, num_words: int = 20) -> str:
        word = start_word
        result = [word]
        for _ in range(num_words):
            next_words = self.bigram_dict.get(word)
            if not next_words:
                break
            word = random.choice(next_words)
            result.append(word)
        return " ".join(result)
