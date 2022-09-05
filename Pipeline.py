import numpy as np
from collections import Counter
from Resources.Constants import Constants


def read():
    file = open(Constants.TRAIN_TEXT_FILE_PATH, 'r', encoding='utf-8')
    text_sample = file.readlines()
    text_sample = ' '.join(text_sample)
    return text_sample


def text_to_seq(sample):
    char_counts = Counter(sample)
    char_counts = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_chars = [char for char, _ in char_counts]
    char_to_idx = {char: index for index, char in enumerate(sorted_chars)}
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    sequence = np.array([char_to_idx[char] for char in sample])

    return sequence, char_to_idx, idx_to_char


def get_pipelined_data():
    return text_to_seq(read())