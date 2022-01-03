import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import *
from box import Box
import logging


class TrainingScheduler(object):
    def __init__(self,
                 config: Box,
                 model: tf.keras.Model,
                 loss: tf.keras.losses.Loss,
                 optimizer: tf.keras.optimizers.Optimizer,
                 x_train: np.array,
                 y_train: np.array,
                 x_test: np.array,
                 y_test: np.array):
        self.config = config
        self.params = self.config.params
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train_custom(self) -> tf.keras.Model:
        """
        Implements a custom tensorflow training loop implementing the baby step
        training algorithm for curriculum training

        Returns:
            tf.keras.Model: Trained model instance
        """
        logging.info("Running custom training loop...")

        # Prepare the training datset
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        train_dataset = train_dataset.batch(batch_size=self.params.batch_size)

        # Prepare the test dataset
        val_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        val_dataset = val_dataset.batch(batch_size=self.params.batch_size)

        # Set up evaluation metrics
        train_acc_metric = keras.metrics.CategoricalAccuracy(name="train_accuracy")
        val_acc_metric = keras.metrics.CategoricalAccuracy(name="val_accuracy")

        epochs = self.params.epochs
        # Run through all epochs
        for epoch in range(epochs):
            print(f"Start of epoch {epoch}/{epochs}")

            # Within the epochs, we iterate through the batches
            # TODO - only iterate through the first few batches in the earlier epochs
            # Then iterate through the more complex batches in later epochs
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                with tf.GradientTape() as tape:
                    # Run the forward pass of the layer & record in gradient tape
                    logits = self.model(x_batch_train, training=True)
                    loss_value = self.loss(y_batch_train, logits)

                # Record gradients in the gradient tape
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients((zip(grads, self.model.trainable_weights)))
                # Update training acc
                train_acc_metric.update_state(y_batch_train, logits)


                # Log every N batches
                if step % 10 == 0:
                    pass
                    # print(f"Training loss (for one batch) at step {step}: {np.round(loss_value, 4)}")
                    # print(f"Samples seens so far: {(step + 1) * self.params.batch_size}")

            # Log metrics at the end of each epoch
            for (x_batch_val, y_batch_val) in val_dataset:
                val_logits = self.model(x_batch_val, training=False)
                val_acc_metric.update_state(y_batch_val, val_logits)

            train_accuracy = np.round(train_acc_metric.result(), 4)
            val_accuracy = np.round(val_acc_metric.result(), 4)

            print(f"EPOCH {epoch+1}/{epochs} - TRAIN ACC: {train_accuracy} ; VAL ACC: {val_accuracy}")

            # reset accuracy states after each epoch
            train_acc_metric.reset_states()
            val_acc_metric.reset_states()

        return self.model