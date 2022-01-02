import os.path
import pickle
from pathlib import Path
from box import Box
from typing import *
import logging
import yaml

def read_cache(major_name: str, minor_name: str) -> Any:
    cache_path = Path("data/cache/")
    filename = f"{major_name}.{minor_name}.pkl"
    assert os.path.exists(cache_path.joinpath(filename)), f"Cache file {filename} not available!"

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

def write_training_split(major_name, x_train, y_train, x_test, y_test):
    logging.info(f"Writing training split for {major_name}")

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


