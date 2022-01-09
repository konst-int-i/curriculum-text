from curriculum_text.pipeline import Pipeline
from curriculum_text.utils import *
import click


@click.command()
@click.option(
    "--config",
    default="config.yml",
    required=True,
    show_default=True,
    help="Path to configuration file",
)
@click.option(
    "--mode", default="full", type=click.Choice(["full", "seed"], case_sensitive=True)
)
def main(config: str, mode: str):
    """
    CLI pipeline wrapper
    """
    config = read_config(config)
    pipe = Pipeline(config)
    if mode == "full":
        pipe.run(full=True)
    else:
        pipe.run(full=False)


if __name__ == "__main__":
    main()
