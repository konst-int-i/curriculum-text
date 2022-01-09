import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import *


def plot_log_accuracies(df: pd.DataFrame, save_dir: Path) -> None:
    """
    plots train/test accuracies of single model after training
    Args:
        df (pd.DataFrame): evaluation dataframe
        save_dir (Path): log directory - matplotlib figure will be saved here

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    plot_df = pd.melt(df, id_vars="epoch", value_vars=["train_acc", "val_acc"]).rename(
        {"value": "Accuracy"}, axis=1
    )
    plot_df["variable"] = plot_df["variable"].map(
        {"train_acc": "Train", "val_acc": "Test"}
    )

    sns.lineplot(
        ax=ax,
        data=plot_df,
        x="epoch",
        y="Accuracy",
        hue="variable",
        markers=True,
        style="variable",
    )

    ax.set_title(
        "Train/validation accuracy by epoch", fontweight="bold", size=20
    )  # Title
    ax.set_ylabel("Accuracy (%)", fontsize=20.0)  # Y label
    ax.set_xlabel("Epoch", fontsize=20)  # X label
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    plt.legend(prop={"size": 16})
    # save fig
    plt.savefig(save_dir.joinpath("train_val_acc.png"))

    plt.show()
