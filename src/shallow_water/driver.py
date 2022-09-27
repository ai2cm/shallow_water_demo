"""Main driver for the shallow water model."""

import argparse
import logging
import os
import sys

import yaml
from mpi4py import MPI

import shallow_water
from shallow_water import ShallowWaterModel

from .communicator import Comm
from .config import Config
from .grid import Grid


def integrate(
    config: Config,
    comm: Comm,
    *,
    output_directory: str = ".",
    output_frequency: int = 0,
) -> ShallowWaterModel:
    """Top-level function for integrating the model in time.

    Parameters:
        config: shallow water config
        comm: MPI communicator
        output_frequency: frequency with which to output the state, <= 0 implies never
        output_directory: directory prefix for state output

    Returns:
        ShallowWaterModel integrated in time for 'num_steps'

    """
    grid = Grid.from_config(config.grid, comm=comm)

    model = ShallowWaterModel(grid, config.model)

    if output_frequency > 0:
        model.state.to_disk(os.path.join(output_directory, "state_initial"))

    for step in range(config.model.num_steps):
        model.take_step()
        logging.info(f"step {step}: time={step * config.model.timestep}")

        if output_frequency > 0 and step % output_frequency:
            model.state.to_disk(os.path.join(output_directory, f"state_step_{step}"))

    if output_frequency > 0:
        model.state.to_disk(os.path.join(output_directory, "state_final"))

    return model


def run(
    config_file: str, output_directory: str, output_frequency: int
) -> ShallowWaterModel:
    """End-to-end driver for the shallow water model.

    Parameters:
        config_file: path to the yaml input config
        output_directory: directory in which to save output
        output_frequency: frequency with which to output the model state

    Raises:
        RuntimeError: when there is an existing file at output_directory

    Returns:
        ShallowWaterModel: the model at the final time

    """
    with open(sys.argv[1], mode="r") as f:
        data = yaml.safe_load(f.read())
        config = shallow_water.Config.from_dict(**data)

    if output_frequency >= 0 and os.path.exists(output_directory):
        raise RuntimeError(
            f"{output_directory} conflicts with data that model will write"
        )

    os.makedirs(output_directory, mode=0o755, exist_ok=True)

    return integrate(
        config,
        MPI.COMM_WORLD,
        output_directory=output_directory,
        output_frequency=output_frequency,
    )


def main() -> None:
    """Driver for the shallow water model that parses command line arguments."""
    parser = argparse.ArgumentParser(description="Run the shallow water model.")
    parser.add_argument("config_file", type=str, help="yaml configuration")
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="output",
        help="Directory to save output",
    )
    parser.add_argument(
        "-f",
        "--output-frequency",
        type=int,
        default=-1,
        help="frequency with which to save the state: "
        "f<0: never, f=0: final state only, f>0: every f steps and last",
    )
    args = parser.parse_args(sys.argv)

    run(args.config_file, args.directory, args.output_frequency)


if __name__ == "__main__":
    main()
