"""Main driver for the shallow water model."""
from shallow_water import ShallowWaterModel

from .communicator import Comm
from .config import Config
from .grid import Grid


def integrate(config: Config, comm: Comm) -> ShallowWaterModel:
    """Top-level function for integrating the model in time.

    Parameters:
        config: shallow water config
        comm: MPI communicator

    Returns:
        ShallowWaterModel integrated in time for 'num_steps'

    """
    grid = Grid.from_config(config.grid, comm=comm)

    model = ShallowWaterModel(grid, config.model)

    for step in range(config.model.num_steps):
        model.take_step()
        print(f"Step {step}: time={step * config.model.timestep}")

    return model
