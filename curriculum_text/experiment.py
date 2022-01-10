import os
import numpy as np
from curriculum_text.viz import plot_log_accuracies
from tensorflow.keras.preprocessing import sequence
from tensorflow import keras
from curriculum_text.preprocess import preprocess_text, encode_text_to_sequence
import tensorflow as tf
import logging
from curriculum_text.fasttext_custom import FastText
from datetime import datetime
from box import Box
from typing import *
import pandas as pd
from curriculum_text.ngram import create_ngram_set, add_ngram
from datasets import load_dataset
from curriculum_text.utils import read_cache, write_training_split, read_config, Timer
from curriculum_text.scheduler import TrainingScheduler
from curriculum_text.difficulty import DifficultyMeasure
from pathlib import Path

pd.options.mode.chained_assignment = None

# set random seeds for reproducibility


# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(filename)s][%(module)s.%(funcName)s:%(lineno)d] %(message)s",
    handlers=[logging.StreamHandler()],
)


class Experiment(object):
    def __init__(
        self,
        config: Box,
        dataset: str,
        curriculum: str = None,
        train_schedule: str = "full",
    ):
        self.params = config.params
        self.ngram_range = self.params.ngram_range
        self.max_features = self.params.max_features
        self.maxlen = self.params.maxlen
        self.embedding_dims = self.params.embedding_dims
        self.epochs = self.params.epochs
        self.debug = config.debug
        self.config = config
        self.dataset_name = dataset
        self.curriculum = curriculum
        self.train_schedule = train_schedule
        self.timer = Timer()  # used to keep track of train time
        date = datetime.now()
        cur_path = "_cur" if curriculum else ""
        directory = (
            f"{dataset}_{str(date.month).zfill(2)}"
            f"{str(date.day).zfill(2)}_{str(date.hour).zfill(2)}{str(date.minute).zfill(2)}{cur_path}"
        )
        self.log_dir = Path(config.log_dir).joinpath(directory)
        self._check_inputs()
        # make log path
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def _check_inputs(self) -> None:
        valid_curricula = ["length", "reverse_length", "distance", None]
        assert (
            self.curriculum in valid_curricula
        ), f"Invalid curriculum measure specified. Valid args: {valid_curricula}"

        valid_schedules = ["baby_step", "full", "custom_shuffle"]
        assert (
            self.train_schedule in valid_schedules
        ), f"Invalid train schedule. Valid args: {valid_schedules}"

    def run(self) -> Tuple[tf.keras.Model, pd.DataFrame]:
        """
        Runs the experiment for given dataset and specified curriculum
        Returns:
            Tuple(tf.keras.Model, pd.DataFrame): trained keras model instance
                and dataframe with train history by epoch as dataframe
        """
        self.get_dataset()
        self.calc_features()
        model, log_df = self.train_model()

        # visualize log and save model
        plot_log_accuracies(df=log_df, save_dir=self.log_dir)

        return model, log_df

    def concat_title_content(self, data_split) -> Tuple:
        """
        For datasets containing both title and content, this function reads in both and concatenates
        them for the relevant train-test splits
        """
        train_titles = data_split[:]["title"]
        train_content = data_split[:]["content"]
        y_data = np.array(data_split[:]["label"])
        x_data = []
        for idx, (title, content) in enumerate(zip(train_titles, train_content)):
            concat = title + " " + content
            x_data.append(concat)
        return x_data, y_data

    def get_dataset(self) -> Tuple:
        # pre-process dataset if not cached

        # READ IN DATA
        logging.info(f"Downloading '{self.dataset_name}' from huggingface")
        train = load_dataset(self.dataset_name, split="train")
        test = load_dataset(self.dataset_name, split="test")

        # get label descriptions
        try:
            self.label_desc = train.features["label"].names
        except KeyError:
            self.label_desc = []

        try:
            self.x_train = np.array(train[:]["text"])
            self.y_train = np.array(train[:]["label"])
            self.x_test = np.array(test[:]["text"])
            self.y_test = np.array(test[:]["label"])
        except KeyError:  # for articles that have "title" and "content" column
            self.x_train, self.y_train = self.concat_title_content(train)
            self.x_test, self.y_test = self.concat_title_content(test)
        # get reduced dataset if in debug mode
        if self.debug:
            debug_idx = int(0.1 * self.x_train.shape[0])
            logging.info(
                f"Debug mode: using first {debug_idx} elements of '{self.dataset_name}' dataset"
            )
            self.x_train = self.x_train[:debug_idx]
            self.y_train = self.y_train[:debug_idx]

        self._preprocess_dataset()

        # reorder preprocessed data by difficulty
        self.x_train, self.y_train = self.apply_curriculum(self.x_train, self.y_train)

        # convert to categorical target
        self.num_classes = len(set(self.y_train))
        self.y_train = tf.keras.utils.to_categorical(
            self.y_train, num_classes=self.num_classes, dtype="int32"
        )
        self.y_test = tf.keras.utils.to_categorical(
            self.y_test, num_classes=self.num_classes, dtype="int32"
        )

        # Final step - encode text to sequence of n-grams
        self.x_train, self.x_test = encode_text_to_sequence(
            self.x_train, self.x_test, num_words=self.max_features
        )

        logging.info(f"Number of classes: {self.num_classes}")
        logging.info(f"{len(self.x_train)} train sequences")
        logging.info(f"{len(self.x_test)} test sequences")
        logging.info(
            "Average train sequence length: {}".format(
                np.mean(list(map(len, self.x_train)), dtype=int)
            )
        )
        logging.info(
            "Average test sequence length: {}".format(
                np.mean(list(map(len, self.x_test)), dtype=int)
            )
        )

    def _preprocess_dataset(self) -> None:
        """
        Function which preprocesses the dataset if
        - The program is not run in debug mode
        - The dataset has not been preprocessed before (would expect a cache)
        Returns:
            None
        """
        # preprocess train/test samples if not cached yet (only to be done once)
        if (
            os.path.exists(
                Path(self.config.cache_dir).joinpath(f"{self.dataset_name}.x_train.pkl")
            )
            and not self.debug
        ):
            logging.info(f"Reading preprocessed files from cache ")
            self.x_train = read_cache(self.dataset_name, "x_train")
            self.x_test = read_cache(self.dataset_name, "x_test")
            self.rewrite_cache = False  # don't rewrite cache
        else:
            logging.info(f"Preprocessing text files...")
            self.x_train = preprocess_text(self.x_train)
            self.x_test = preprocess_text(self.x_test)
            self.rewrite_cache = True

        if self.rewrite_cache:
            write_training_split(
                self.config,
                self.dataset_name,
                self.x_train,
                self.y_train,
                self.x_test,
                self.y_test,
            )

    def calc_features(self):
        if self.ngram_range > 1:
            logging.info("Adding {}-gram features".format(self.ngram_range))
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
            self.x_test = add_ngram(self.x_test, token_indice, self.ngram_range)

            logging.info(
                "Average train sequence length: {}".format(
                    np.mean(list(map(len, self.x_train)), dtype=int)
                )
            )
            logging.info(
                "Average test sequence length: {}".format(
                    np.mean(list(map(len, self.x_test)), dtype=int)
                )
            )

        logging.info("Pad sequences (samples x time)...")
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=self.maxlen)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=self.maxlen)
        logging.info(f"x_train shape: {self.x_train.shape}")
        logging.info(f"x_test shape: {self.x_test.shape}")
        logging.info(f"y_train shape: {self.y_train.shape}")
        logging.info(f"y_test shape: {self.y_test.shape}")

        return None

    def apply_curriculum(self, x_train: List, y_train: List) -> Tuple[List]:
        """
        Function which calculates a sample difficulty measure and reorders the train features and labels
        in ascending order by difficulty. These can then we used for the curriculum
        Returns:
            Tuple: x_train and y_train as a list
        """
        train_df = pd.DataFrame({"text": x_train, "label": y_train})
        # remove nans
        train_df = train_df[~train_df.text.isna()]

        if self.curriculum is None:
            # shuffle if no curriculum
            logging.info(f"Shuffling training samples")
            # shuffle data
            train_df = train_df.sample(frac=1, random_state=self.config.seed)
            return train_df["text"].to_numpy(), train_df["label"].to_numpy()

        measure = DifficultyMeasure(self.config, train_df, self.label_desc)

        measure_dict = {
            "length": measure.length,
            "reverse_length": measure.reverse_length,
            "distance": measure.vector_distance,
        }
        train_df = measure_dict[self.curriculum]()

        return train_df["text"].to_numpy(), train_df["label"].to_numpy()

    def train_model(self):
        logging.info("Build model...")

        model = FastText(
            self.maxlen,
            self.max_features,
            self.embedding_dims,
            class_num=self.num_classes,
        )
        loss = keras.losses.CategoricalCrossentropy()
        optimizer = keras.optimizers.Adam(
            learning_rate=self.params.lr, clipnorm=self.params.clipnorm
        )

        # if self.curriculum is not None:
        scheduler = TrainingScheduler(
            self.config,
            model,
            loss,
            optimizer,
            self.x_train,
            self.y_train,
            self.x_test,
            self.y_test,
        )

        schedule_dict = {
            "full": scheduler.train_full,
            "custom_shuffle": scheduler.train_bin_shuffle,
            "baby_step": scheduler.train_baby_step,
        }
        logging.info(f"Launching {self.train_schedule} train schedule")
        self.timer.start()
        model, training_log = schedule_dict[self.train_schedule]()
        train_time = self.timer.stop()

        training_log.loc[:, "train_time"] = train_time
        training_log.loc[:, "epochs"] = training_log.epoch.max()
        training_log.loc[:, "dataset"] = self.dataset_name
        training_log.loc[:, "seed"] = self.config.seed
        training_log.loc[:, "schedule"] = self.train_schedule
        training_log.loc[:, "curriculum"] = self.curriculum
        print(training_log)

        return model, training_log


if __name__ == "__main__":

    dataset = "ag_news"
    # dataset = "dbpedia_14"
    # dataset = "yelp_review_full"
    # dataset = "yelp_polarity"
    # dataset = "amazon_polarity"
    # dataset = "yahoo_answers_topics"
    # dataset="amazon_us_reviews"
    config = read_config("config.yml")

    ag_news_d = Experiment(
        config, dataset, curriculum="distance", train_schedule="baby_step"
    )
    model4, log4 = ag_news_d.run()

    # ag_news_l = Experiment(config, dataset, curriculum="distance", train_schedule="custom_shuffle")
    # model1, log1 = ag_news_l.run()
    #
    # ag_news_rl = Experiment(config, dataset, curriculum="reverse_length", train_schedule="custom_shuffle")
    # model2, log2 = ag_news_rl.run()
    # #
    # ag_news = Experiment(config, dataset, curriculum=None, train_schedule="full")
    # model3, log3 = ag_news.run()
    # #
