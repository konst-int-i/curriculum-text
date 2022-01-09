import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import *
from curriculum_text.utils import *
from box import Box
import random
import logging
import pandas as pd
from copy import deepcopy
from collections import Counter


class TrainingScheduler(object):
    def __init__(
        self,
        config: Box,
        model: tf.keras.Model,
        loss: tf.keras.losses.Loss,
        optimizer: tf.keras.optimizers.Optimizer,
        x_train: np.array,
        y_train: np.array,
        x_test: np.array,
        y_test: np.array,
    ):
        self.config = config
        self.params = self.config.params
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.training_log = pd.DataFrame(
            columns=["epoch", "train_acc", "val_acc", "data_share"]
        )
        self._prep_data()

    def _prep_data(self):
        """
        Prepares train and test data as tensors
        Returns:
            None
        """

        # allow down if necessary
        self.batch_size = int(np.floor(self.x_train.shape[0] / self.params.batches))

        # Prepare the training dataset
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_train, self.y_train)
        )
        self.train_dataset = self.train_dataset.batch(batch_size=self.batch_size)

        # Prepare the test dataset
        # note that the batches are ordered by the curriculum score
        self.val_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_test, self.y_test)
        )
        self.val_dataset = self.val_dataset.batch(batch_size=self.batch_size)

        self.train_acc_metric = keras.metrics.CategoricalAccuracy(name="train_accuracy")
        self.val_acc_metric = keras.metrics.CategoricalAccuracy(name="val_accuracy")
        self.val_precision = keras.metrics.Precision(name="val_precision")
        self.val_recall = keras.metrics.Precision(name="val_recall")

    def _get_train_subset(
        self, share: float, reshuffle: bool = True, seed: int = 1
    ) -> tf.data.Dataset:
        """
        Gets the training subset of a given ``share`` for baby step algorithm. Note that the
        "buckets" should be shuffled between each other, such that there is a full shuffle in the
        final iteration
        Args:
            share (float): First x% of data (sorted by difficulty) to fetch
            reshuffle (bool): reshuffles tensor if true
            seed (int): fixed seed for reproducibility of shuffle

        Returns:
            tf.data.Dataset: batched dataset containing x_train and y_train shuffled subset
        """
        idx_cut = int(np.floor(share * self.x_train.shape[0]))
        # print(f"Fetching first {idx_cut} samples (share: {share})")
        x_train_subset = deepcopy(self.x_train[:idx_cut])
        y_train_subset = deepcopy(self.y_train[:idx_cut])
        train_subset = tf.data.Dataset.from_tensor_slices(
            (x_train_subset, y_train_subset)
        )
        if reshuffle:
            # set buffer size to idx_cut for perfect shuffle
            train_subset = train_subset.shuffle(buffer_size=idx_cut, seed=seed)

        train_subset = train_subset.batch(batch_size=self.batch_size)
        return train_subset

    def _shuffle_train_buckets(self, n_buckets: int, seed: int):
        """
        Routine which uses the ``self.x_train`` and ``self.y_train`` samples ordered
        by difficulty and buckets then into ``n_buckets``. Every sub-bucket is then
        shuffled and returned as a batched tf dataset
        Args:
            n_buckets (int): Number of sub-buckets ordered by difficulty to shuffle
            seed (int): seed used for shuffling (can use epoch number to avoid same
                shuffle between epochs)
        Returns:
            (tf.data.Dataset): batched tf dataset to be used in that epoch
        """
        logging.info(f"")
        # divide into even-sized buckets by difficulty
        x_train_buckets = np.array_split(self.x_train, n_buckets)
        y_train_buckets = np.array_split(self.y_train, n_buckets)
        # shuffle elements within each bucket
        for idx, (x, y) in enumerate(zip(x_train_buckets, y_train_buckets)):
            train_buckets = list(zip(x, y))
            random.seed(seed)
            random.shuffle(train_buckets)
            shuffle_x, shuffle_y = zip(*train_buckets)
            x_train_buckets[idx] = shuffle_x
            y_train_buckets[idx] = shuffle_y
        # re-concatenate buckets
        x_train = np.concatenate(x_train_buckets)
        y_train = np.concatenate(y_train_buckets)

        # pack into tensor and batch
        train_tensor = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_tensor = train_tensor.batch(batch_size=self.batch_size)
        return train_tensor

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer & record in gradient tape
            logits = self.model(x, training=True)
            # squeeze logits to improve early performance
            logits = tf.squeeze(logits)
            loss_value = self.loss(y, logits)

        # Record gradients in the gradient tape
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients((zip(grads, self.model.trainable_weights)))
        # Update training acc
        self.train_acc_metric.update_state(y, logits)

    @tf.function
    def test_step(self, x, y):
        val_logits = self.model(x, training=False)
        val_logits = tf.squeeze(val_logits)
        self.val_acc_metric.update_state(y, val_logits)
        self.val_recall.update_state(y, val_logits)
        self.val_precision.update_state(y, val_logits)

    def train_baby_step(self) -> Tuple[tf.keras.Model, pd.DataFrame]:
        """
        Implements a custom tensorflow training loop implementing the baby step
        training algorithm for curriculum training

        Returns:
            tf.keras.Model: Trained model instance
        """
        logging.info("Running 'baby step' algo training loop...")
        epochs = self.params.epochs
        max_epochs = 5  # epochs with full training set ("plateau")
        start = 0.3
        end = 1
        step_size = (end - start) / (epochs - max_epochs)
        shares = np.arange(start, end, step_size)
        shares = np.append(shares, [1 for i in range(max_epochs)])
        # add one as first element
        print(shares)
        # Run through all epochs
        for epoch in range(epochs):
            # print(f"Start of epoch {epoch}/{epochs}")
            epoch_batch_share = shares[epoch]
            n_batches = int(np.ceil(len(self.train_dataset) * epoch_batch_share))
            # print(f"Sampling training data from {n_batches} batches ({epoch_batch_share})")
            train_subset = self._get_train_subset(epoch_batch_share)

            # Within the epochs, iterate through batches
            # for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
            for step, (x_batch_train, y_batch_train) in enumerate(train_subset):
                self.train_step(x_batch_train, y_batch_train)

            # Log metrics at the end of each epoch
            for (x_batch_val, y_batch_val) in self.val_dataset:
                self.test_step(x_batch_val, y_batch_val)

            train_accuracy = np.round(self.train_acc_metric.result(), 4)
            val_accuracy = np.round(self.val_acc_metric.result(), 4)
            val_precision = np.round(self.val_precision.result(), 4)
            val_recall = np.round(self.val_recall.result(), 4)

            self._append_log_df(
                epoch,
                train_accuracy,
                val_accuracy,
                val_precision,
                val_recall,
                epoch_batch_share,
            )

            print(
                f"EPOCH {epoch+1}/{epochs} - TRAIN ACC: {train_accuracy} ; VAL ACC: {val_accuracy}"
            )

            # reset accuracy states after each epoch
            self._reset_metrics()

            if self._check_early_stopping(
                self.training_log["val_acc"].to_list(),
                self.params.early_stopping_patience,
            ):
                break

        return self.model, self.training_log

    def _reset_metrics(self):
        self.train_acc_metric.reset_states()
        self.val_acc_metric.reset_states()
        self.val_recall.reset_states()
        self.val_precision.reset_states()

    def _check_early_stopping(self, acc: List, patience: int) -> bool:
        """
        Early stopping implementation for custom training loop - returns True if
        ``acc`` has dropped for ``patience`` consecutive periods
        Returns:
        """
        last = acc[-patience - 1 :]
        stop = False
        print(last)
        for i in range(len(last)):
            if i == len(last) - 1:
                return stop
            if last[i] > last[i + 1]:
                stop = True
            else:
                return False
        return stop

    def train_full(self) -> Tuple[tf.keras.Model, pd.DataFrame]:
        """
        Implements a custom tensorflow training loop implementing the baby step
        training algorithm for curriculum training

        Returns:
            tf.keras.Model: Trained model instance
        """
        logging.info("Running custom training loop...")
        epochs = self.params.epochs
        # Run through all epochs
        for epoch in range(epochs):
            print(
                f"Epoch: {epoch}/{epochs} - Running {len(self.train_dataset)} batches"
            )
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                self.train_step(x_batch_train, y_batch_train)

            # Log metrics at the end of each epoch
            for step, (x_batch_val, y_batch_val) in enumerate(self.val_dataset):
                self.test_step(x_batch_val, y_batch_val)

            train_accuracy = np.round(self.train_acc_metric.result(), 4)
            val_accuracy = np.round(self.val_acc_metric.result(), 4)
            val_precision = np.round(self.val_precision.result(), 4)
            val_recall = np.round(self.val_recall.result(), 4)

            self._append_log_df(
                epoch, train_accuracy, val_accuracy, val_precision, val_recall, 1
            )
            print(
                f"EPOCH {epoch+1}/{epochs} - TRAIN ACC: {train_accuracy} ; VAL ACC: {val_accuracy}"
            )
            # check custom early stopping
            # reset accuracy states after each epoch
            self._reset_metrics()

            if self._check_early_stopping(
                self.training_log["val_acc"].to_list(),
                self.params.early_stopping_patience,
            ):
                break

        return self.model, self.training_log

    def _append_log_df(self, epoch, train_acc, val_acc, val_prec, val_rec, data_share):
        self.training_log = self.training_log.append(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_prec": val_prec,
                "val_rec": val_rec,
                "data_share": data_share,
            },
            ignore_index=True,
        )

    def train_bin_shuffle(self) -> Tuple[tf.keras.Model, pd.DataFrame]:
        """
        Implements a custom tensorflow training loop implementing the baby step
        training algorithm for curriculum training

        Returns:
            tf.keras.Model: Trained model instance
        """
        logging.info("Running custom training loop...")
        epochs = self.params.epochs
        # Run through all epochs
        for epoch in range(epochs):
            print(
                f"Epoch: {epoch}/{epochs} - Running {len(self.train_dataset)} batches"
            )
            # cut into n major buckets and shuffle within each bucket
            n_buckets = 8
            train_data = self._shuffle_train_buckets(n_buckets, seed=epoch)
            # shuffle each major buckets
            # Within the epochs, iterate through batches
            for step, (x_batch_train, y_batch_train) in enumerate(train_data):
                self.train_step(x_batch_train, y_batch_train)

            # Log metrics at the end of each epoch
            for (x_batch_val, y_batch_val) in self.val_dataset:
                self.test_step(x_batch_val, y_batch_val)

            train_accuracy = np.round(self.train_acc_metric.result(), 4)
            val_accuracy = np.round(self.val_acc_metric.result(), 4)
            val_precision = np.round(self.val_precision.result(), 4)
            val_recall = np.round(self.val_recall.result(), 4)

            self._append_log_df(
                epoch, train_accuracy, val_accuracy, val_precision, val_recall, 1
            )
            print(
                f"EPOCH {epoch+1}/{epochs} - TRAIN ACC: {train_accuracy} ; VAL ACC: {val_accuracy}"
            )

            # reset accuracy states after each epoch
            self._reset_metrics()

            # check custom early stopping
            if self._check_early_stopping(
                self.training_log["val_acc"].to_list(),
                self.params.early_stopping_patience,
            ):
                break

        return self.model, self.training_log
