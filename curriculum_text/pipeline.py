import os
import random as python_random
from datetime import datetime
from pathlib import Path

import pandas as pd
import tensorflow as tf
from box import Box
from numpy.random import seed

from curriculum_text.experiment import Experiment


class Pipeline(object):
    """
    Pipeline wrapper that runs the full set of experiments
    """

    def __init__(self, config: Box):
        self.config = config
        # datasets to test
        self.datasets = [
            "ag_news",
            "dbpedia_14",
            "yelp_review_full",
        ]
        self.curricula = [
            "distance",
            "length",
            "reverse_length",
        ]
        # self.seeds = range(1, 2) # change to 5
        self.seeds = [1]
        self.schedulers = ["baby_step", "custom_shuffle"]
        now = datetime.now()
        directory = (
            f"full_results_{str(now.month).zfill(2)}{str(now.day).zfill(2)}_"
            f"{str(now.hour).zfill(2)}{str(now.minute).zfill(2)}"
        )
        self.result_dir = Path("data/results/").joinpath(directory)

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    def run(self, full: bool):

        if full:
            baseline_logs = self.baseline_runs()
            # write to file
            baseline_logs.to_csv(self.result_dir.joinpath("baseline_results.csv"))
            experiment_logs = self.curriculum_runs()
            experiment_logs.to_csv(self.result_dir.joinpath("experiment_results.csv"))
        else:  # run specific models across multiple seeds
            seed_logs = self.seed_runs()
            seed_logs.to_csv(self.result_dir.joinpath("seed_experiments.csv"))

    def baseline_runs(self) -> pd.DataFrame:
        """
        Run the baseline model on all datasets - on "full" training algorithm
        """
        log_dfs = []
        for dataset in self.datasets:
            for s in self.seeds:
                self._set_seed(s)
                exp = Experiment(
                    self.config, dataset, curriculum=None, train_schedule="full"
                )
                model, log_df = exp.run()
                log_dfs.append(log_df)

        baseline_log_df = pd.concat(log_dfs)
        return baseline_log_df

    def curriculum_runs(self):
        log_dfs = []
        for dataset in self.datasets:
            for cur in self.curricula:
                for scheduler in self.schedulers:
                    for s in self.seeds:
                        self._set_seed(s)
                        exp = Experiment(self.config, dataset, cur, scheduler)
                        model, log_df = exp.run()
                        log_dfs.append(log_df)
        experiment_log_df = pd.concat(log_dfs)
        return experiment_log_df

    def seed_runs(self):
        """
        Run specific models for multiple seeds
        """
        log_dfs = []
        for dataset in self.datasets:
            for s in range(1, 6):
                self._set_seed(s)
                base = Experiment(self.config, dataset, None, "full")
                exp = Experiment(self.config, dataset, "distance", "baby_step")
                _, log_base = base.run()
                _, log_exp = exp.run()
                log_dfs.append(log_base)
                log_dfs.append(log_exp)
        return pd.concat(log_dfs)

    def _set_seed(self, s: int):
        self.config.seed = s
        seed(s)
        python_random.seed(s)
        tf.random.set_seed(s)
