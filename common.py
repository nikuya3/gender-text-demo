import numpy as np
import re
from string import ascii_lowercase

sequence_length = 1000
min_sequence_length = 1000  # data with lower than 5 chars is discarded
batch_size = 1200

tokens = ascii_lowercase + '!"#%&\'()/:@^~ '
char_to_int = {t: i + 1 for i, t in enumerate(tokens)}
char_to_int['NA'] = 0


def clean_text(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = text.replace('urlLink', '')
    text = ' '.join(text.split())  # substitutes multiple whitespaces with single whitespace
    return text


def tokenize(text):
    text = clean_text(text)
    ints = [char_to_int[c] for c in text if c in tokens]
    return np.array(ints)
