import os.path
import pickle
from pathlib import Path
from box import Box
from typing import *
import logging
import yaml
import time
import pandas as pd


def read_cache(major_name: str, minor_name: str) -> Any:
    cache_path = Path("data/cache/")
    filename = f"{major_name}.{minor_name}.pkl"
    assert os.path.exists(
        cache_path.joinpath(filename)
    ), f"Cache file {filename} not available!"

    with open(cache_path.joinpath(filename), "rb") as f:
        content = pickle.load(f)
    return content


def write_cache(to_write: Any, major_name: str, minor_name: str):
    """
    Writes any object as pickle in directory ``data/cache`` with filename
    ``major_name.minor_name.pkl``
    Args:
        to_write (Any): any python object that can be pickled
        major_name (str): typically name of dataset (e.g. "ag_news")
        minor_name (str): typically name of the data split (e.g. "x_train")
    """
    cache_path = Path("data/cache/")
    filename = f"{major_name}.{minor_name}.pkl"

    with open(cache_path.joinpath(filename), "wb") as f:
        pickle.dump(to_write, f)
    return None


def write_training_split(config: Box, major_name, x_train, y_train, x_test, y_test):

    if config.debug:
        logging.info(f"Not caching training split (debug flag on)")
    else:
        logging.info(f"Writing training splits for {major_name}")

        write_cache(to_write=x_train, major_name=major_name, minor_name="x_train")
        write_cache(to_write=y_train, major_name=major_name, minor_name="y_train")
        write_cache(to_write=x_test, major_name=major_name, minor_name="x_test")
        write_cache(to_write=y_test, major_name=major_name, minor_name="y_test")

    return None


def read_config(path: Path) -> Box:
    """
    Args:
        path: path to config file
    Returns:
        Box: box object containing the read ``config.yml``
    """
    with open(Path(path), "r") as f:
        config = yaml.safe_load(f)

    # return config
    return Box(config)


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise ValueError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise ValueError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return elapsed_time


def df_to_ft_format(df: pd.DataFrame, filepath: str):
    write_str = ""

    for idx, row in df.iterrows():
        write_str += row["text"]
        write_str += f" __label__{row['label']}\n"

    with open(filepath, "w") as f:
        f.write(write_str)

    return None
