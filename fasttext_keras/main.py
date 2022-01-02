import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import sequence
from fasttext_keras.preprocess import preprocess_text, encode_text_to_sequence
import tensorflow as tf
import logging
from fasttext import FastText
from box import Box
from typing import *
# set up logging
from fasttext_keras.ngram import create_ngram_set, add_ngram
from datasets import load_dataset
from fasttext_keras.utils import read_cache, write_training_split, read_config
from numpy.random import seed
import random as python_random

# set random seeds for reproducibility
SEED = 1
seed(SEED)
python_random.seed(SEED)
tf.random.set_seed(SEED)


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(filename)s][%(module)s.%(funcName)s:%(lineno)d] %(message)s",
    handlers=[logging.StreamHandler()],
)

class Experiment(object):
    def __init__(self, config: Box, dataset: str):
        self.params = config.params
        self.ngram_range = self.params.ngram_range
        self.max_features = self.params.max_features
        self.maxlen = self.params.maxlen
        self.embedding_dims = self.params.embedding_dims
        self.batch_size = self.params.batch_size
        self.epochs = self.params.epochs
        self.debug = config.debug
        self.config = config
        self.dataset_name = dataset

    def run(self):
        pass

    def get_dataset(self) -> Tuple:
        # pre-process dataset if not cached
        if self.config.use_cache:
            logging.info(f"Loading '{self.dataset_name}' dataset from cache")
            self.x_train = read_cache(major_name=self.dataset_name, minor_name="x_train")
            self.x_test = read_cache(major_name=self.dataset_name, minor_name="x_test")
            self.y_train = read_cache(major_name=self.dataset_name, minor_name="y_train")
            self.y_test = read_cache(major_name=self.dataset_name, minor_name="y_test")

        else:
            logging.info(f"Downloading '{self.dataset_name}' from huggingface")
            train = load_dataset(self.dataset_name, split="train")
            test = load_dataset(self.dataset_name, split="test")


            self.x_train = np.array(train[:]["text"])
            self.y_train = np.array(train[:]["label"])
            self.x_test = np.array(test[:]["text"])
            self.y_test = np.array(test[:]["label"])

            # get reduced dataset if in debug mode
            if self.debug:
                debug_idx = int(0.1 * self.x_train.shape[0])
                logging.info(f"Debug mode: using first {debug_idx} elements of '{self.dataset_name}' dataset")
                self.x_train = self.x_train[:debug_idx]
                self.y_train = self.y_train[:debug_idx]


            logging.info(f"Preprocessing...")
            self.x_train = preprocess_text(self.x_train)
            self.x_test = preprocess_text(self.x_test)
            self.x_train, self.x_test = encode_text_to_sequence(self.x_train,
                                                                self.x_test,
                                                                num_words=self.max_features)

            self.num_classes = len(set(self.y_train))

            # convert to categorical target
            self.y_train = tf.keras.utils.to_categorical(self.y_train, num_classes=self.num_classes, dtype='int32')
            self.y_test = tf.keras.utils.to_categorical(self.y_test, num_classes=self.num_classes, dtype='int32')
            logging.info(f"Number of classes: {self.num_classes}")

        logging.info(f'{len(self.x_train)} train sequences')
        logging.info(f'{len(self.x_test)} test sequences')
        logging.info('Average train sequence length: {}'.format(np.mean(list(map(len, self.x_train)), dtype=int)))
        logging.info('Average test sequence length: {}'.format(np.mean(list(map(len, self.x_test)), dtype=int)))


        # write training split
        write_training_split(
            self.dataset_name,
            self.x_train,
            self.y_train,
            self.x_test,
            self.y_test
        )



    def calc_features(self):
        if self.ngram_range > 1:
            logging.info('Adding {}-gram features'.format(self.ngram_range))
            # Create set of unique n-gram from the training set.
            ngram_set = set()
            for input_list in self.x_train:
                for i in range(2, self.ngram_range + 1):
                    set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)

            # Dictionary mapping n-gram token to a unique integer.
            # Integer values are greater than max_features in order
            # to avoid collision with existing features.
            start_index = self.max_features + 1
            token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
            indice_token = {token_indice[k]: k for k in token_indice}

            # max_features is the highest integer that could be found in the dataset.
            max_features = np.max(list(indice_token.keys())) + 1

            # Augmenting x_train and x_test with n-grams features
            self.x_train = add_ngram(self.x_train, token_indice, self.ngram_range)
            self.x_test = add_ngram(x_test, token_indice, self.ngram_range)

            logging.info('Average train sequence length: {}'.format(np.mean(list(map(len, self.x_train)), dtype=int)))
            logging.info('Average test sequence length: {}'.format(np.mean(list(map(len, self.x_test)), dtype=int)))

        logging.info('Pad sequences (samples x time)...')
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=self.maxlen)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=self.maxlen)
        logging.info(f'x_train shape: {self.x_train.shape}')
        logging.info(f'x_test shape: {self.x_test.shape}')
        logging.info(f"y_train shape: {self.y_train.shape}")
        logging.info(f"y_test shape: {self.y_test.shape}")

        return None

    def train_model(self):
        logging.info('Build model...')

        model = FastText(self.maxlen, self.max_features, self.embedding_dims, class_num=self.num_classes)
        if self.num_classes <= 2:
            loss = tf.keras.losses.BinaryCrossentropy(),
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(),

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.params.lr,
                                                         clipnorm=self.params.clipnorm),
                      loss=loss,
                      metrics=['accuracy'])

        logging.info('Train...')
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
        model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  callbacks=[early_stopping],
                  validation_data=(self.x_test, self.y_test)
                  )

        logging.info('Test...')
        result = model.predict(self.x_test)

        return result




if __name__ == "__main__":

    config = read_config("config.yml")
    exp = Experiment(config, "ag_news")
    exp.get_dataset()
    exp.calc_features()
    exp.train_model()
