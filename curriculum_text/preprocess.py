import logging
import re
import string
from typing import List, Tuple

import nltk
import numpy as np
from keras.preprocessing.text import Tokenizer
from nltk.stem import WordNetLemmatizer


def preprocess_text(data: List[str]) -> List[str]:
    """
    Preprocesses the raw text data by removing quotes, punctuations, digits and applying NLTK word stemming

    Args:
        data (List[str]): samples, where each sample is an element of the list
    Returns:
        List[str]: cleaned samples in the same order
    """
    logging.info("Preprocessing text...")
    quotes = re.compile(
        r"(writes in|writes:|wrote:|says:|said:|^In article|^Quoted from|^\||^>)"
    )
    # Create a stemmer/lemmatizer
    lemmatizer = WordNetLemmatizer()
    for i in range(len(data)):
        if i % 10000 == 0:
            logging.info(f"Processing sample {i}/{len(data)}...")
        # Remove quotes
        data[i] = "\n".join(
            [line for line in data[i].split("\n") if not quotes.search(line)]
        )
        # Remove punctuation (!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)
        data[i] = data[i].translate(str.maketrans("", "", string.punctuation))
        # Remove digits
        data[i] = re.sub("\d", "", data[i])
        # Lemmatize words
        data[i] = " ".join(
            [lemmatizer.lemmatize(word) for word in data[i].split()]
        ).lower()

    # Return data
    return data


def encode_text_to_sequence(
    train: List[str], test: List[str], num_words: int = 10000
) -> Tuple[List, List]:
    """
    Tokenizes words
    Args:
        train (List[str]): train data
        test (List[str]): test data
        num_words (int): max words in vocabulary (e.g. e.g. only "most frequent ``num_words`` words)

    Returns:
        returns (Tuple): train and test data as a tuple
    """
    logging.info(f"Encoding text to sequences...")
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train)
    train_sequences = tokenizer.texts_to_sequences(train)
    test_sequences = tokenizer.texts_to_sequences(test)
    # test_sequences = _increment_and_separate(test_sequences)
    return np.array(train_sequences), np.array(test_sequences)


def _increment_and_separate(data: List[List[int]]):
    """
    Function which adds a "1" separator to each sample (required by Keras)
    and increments the remaining elements
    :param data:
    :return:
    """
    for i in range(len(data)):
        # increment all elems by one
        data[i] = [idx + 1 for idx in data[i]]
        # insert separator as first element
        data[i].insert(0, 1)
    return data
