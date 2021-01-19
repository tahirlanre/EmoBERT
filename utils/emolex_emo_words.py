import os
import pickle

def get_emo_words(file_path):
    emo_words = []
    with open(file_path) as f:
        for line in f:
            line = line.split()
            if len(line) != 3:
                continue
            word, emotion, label = line
            if emotion == 'positive' or emotion == 'negative':
                continue
            if int(label) == 1:
                emo_words.append(word)
    emo_words = list(set(emo_words))
    return emo_words

def main():
    lexicon_file_path = 'data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
    base_path = os.getcwd()
    filename = base_path + '/data/' + 'emolex_emo_words.pkl'
    emo_words = get_emo_words(lexicon_file_path)
    with open(filename, 'wb') as f:
        pickle.dump(emo_words, f)

if __name__ == "__main__":
    main()