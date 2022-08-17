"""IO-related methods."""

import dataclasses
from typing import Any, Dict, Tuple

import dacite
import yaml


@dataclasses.dataclass(frozen=True)
class Grid:
    """Geometry configuration."""

    ni: int
    nj: int
    nk: int
    proc_layout: Tuple[int, int]
    xlimit: Tuple[float, float] = (0, 1)
    ylimit: Tuple[float, float] = (0, 1)


@dataclasses.dataclass(frozen=True)
class Model:
    """The model configuration."""

    num_steps: int
    timestep: float
    gt4py_backend: str = "numpy"
    initial_condition: str = "tidal_wave"
    """The initial condition.

    Must match one of the value strings in shallow_water.InitialCondition."""

    ic_data: Dict[str, Any] = dataclasses.field(default_factory=dict)
    """Data that the initial condition function uses."""


@dataclasses.dataclass(frozen=True)
class Config:
    """Top-level configuration."""

    grid: Grid
    model: Model

    @classmethod
    def from_dict(cls, **kwargs) -> "Config":
        """Build a Config object from keyword arguments.

        Parameters:
            kwargs: data

        Returns:
            Config object
        """
        kwargs["grid"]["proc_layout"] = tuple(kwargs["grid"]["proc_layout"])
        kwargs["grid"]["xlimit"] = tuple(kwargs["grid"].get("xlimit", (0, 1)))
        kwargs["grid"]["ylimit"] = tuple(kwargs["grid"].get("ylimit", (0, 1)))

        return dacite.from_dict(
            data_class=cls, data=kwargs, config=dacite.Config(strict=True)
        )

    def to_yaml(self) -> str:
        """Return a yaml representation of the Config.

        Returns:
            yaml str
        """
        return yaml.dump(dataclasses.asdict(self))

    def __post_init__(self):
        """Validate the configuration.

        Raises:
            ValueError: If the procs do not evenly divide the grid,
                or if there are an insufficient number of grid points.

        """
        if self.grid.ni / self.grid.proc_layout[0] < 2:
            raise ValueError(
                "Insufficent number of grid points in i-direction. "
                f"Requires at least {self.grid.proc_layout[0] * 2} but got {self.grid.ni} points."
            )
        if self.grid.nj / self.grid.proc_layout[1] < 2:
            raise ValueError(
                "Insufficent number of grid points in j-direction. "
                f"Requires at least {self.grid.proc_layout[1] * 2} but got {self.grid.nj} points."
            )
        i_res = self.grid.ni / self.grid.proc_layout[0]
        j_res = self.grid.nj / self.grid.proc_layout[1]
        if not i_res.is_integer() or not j_res.is_integer():
            raise ValueError(
                "Proc layout must divide the grid size exactly without remainder. "
                f"Calculated {i_res} i-indices and {j_res} j-indices per proc."
            )
