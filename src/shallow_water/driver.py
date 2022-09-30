"""Main driver for the shallow water model."""

import logging
import os

import yaml
from mpi4py import MPI

from .communicator import Comm
from .config import Config
from .grid import Grid
from .model import ShallowWaterModel
from .state import State


def integrate(
    config: Config,
    comm: Comm,
    *,
    output_directory: str = ".",
    output_frequency: int = 0,
    globalize: bool = False,
) -> ShallowWaterModel:
    """Top-level function for integrating the model in time.

    Parameters:
        config: shallow water config
        comm: MPI communicator
        output_frequency: frequency with which to output the state, <= 0 implies never
        output_directory: directory prefix for state output
        globalize: if True, gathers the state after integrating (at the 'final' state)

    Returns:
        ShallowWaterModel integrated in time for 'num_steps'

    """
    grid = Grid.from_config(config.grid, comm=comm)

    model = ShallowWaterModel(grid, config.model)

    if output_frequency > 0:
        model.state.to_disk(os.path.join(output_directory, "initial"))

    for step in range(config.model.num_steps):
        model.take_step()
        logging.info(f"step {step}: time={step * config.model.timestep}")

        if output_frequency > 0 and step % output_frequency:
            # Uses `step + 1` because the output is after the model takes `step`
            model.state.to_disk(os.path.join(output_directory, f"step_{step + 1}"))

    if output_frequency > 0:
        model.state.to_disk(os.path.join(output_directory, "final"))

    if globalize:
        global_state = State.gather_from(model.state)
        if grid.comm.Get_rank() == 0:
            assert global_state is not None
            global_state.to_disk(os.path.join(output_directory, "final"))

    return model


def run(
    config_file: str,
    output_directory: str,
    output_frequency: int,
    *,
    globalize: bool = False,
) -> ShallowWaterModel:
    """End-to-end driver for the shallow water model.

    Parameters:
        config_file: path to the yaml input config
        output_directory: directory in which to save output
        output_frequency: frequency with which to output the model state
        globalize: if True, gathers the state after integrating (at the 'final' state)

    Raises:
        RuntimeError: when there is an existing file at output_directory

    Returns:
        ShallowWaterModel: the model at the final time

    """
    with open(config_file, mode="r") as f:
        data = yaml.safe_load(f.read())
        config = Config.from_dict(**data)

    if output_frequency >= 0 and os.path.isfile(output_directory):
        raise RuntimeError(
            f"{output_directory} conflicts with data that model will write"
        )

    os.makedirs(output_directory, mode=0o755, exist_ok=True)

    return integrate(
        config,
        MPI.COMM_WORLD,
        output_directory=output_directory,
        output_frequency=output_frequency,
        globalize=globalize,
    )
