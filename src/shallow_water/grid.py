"""Shallow water model."""

import dataclasses
import json
from typing import Any, List, Optional, Tuple

import dacite
import numpy as np

from . import config
from .communicator import Comm, NullComm


@dataclasses.dataclass
class Grid:
    """The model's Grid object.

    ni, nj, nk represent the proc-owned grid portion.

    """

    ni: int
    nj: int
    nk: int
    xlimit: Tuple[float, float]
    ylimit: Tuple[float, float]
    procs: Tuple[
        Tuple[Optional[int], Optional[int]], Tuple[Optional[int], Optional[int]]
    ]
    position: Tuple[int, int]
    proc_layout: Tuple[int, int]
    comm: Comm

    @classmethod
    def from_config(cls, config_grid: config.Grid, comm: Comm) -> "Grid":
        """Create a solver grid.

        Parameters:
            config_grid: configuration
            comm: MPI communicator

        Raises:
            RuntimeError: if there are more ranks than expected in the proc_layout

        Returns:
            solver grid

        """
        num_ranks_expected = config_grid.proc_layout[0] * config_grid.proc_layout[1]

        if comm.Get_rank() + 1 > num_ranks_expected:
            raise RuntimeError("Unexpected rank")

        my_rank = comm.Get_rank()

        row = my_rank // config_grid.proc_layout[0]
        col = my_rank % config_grid.proc_layout[0]

        procs = (
            (
                my_rank - 1 if col > 0 else None,
                my_rank + 1 if col + 1 < config_grid.proc_layout[0] else None,
            ),
            (
                my_rank - config_grid.proc_layout[0] if row > 0 else None,
                my_rank + config_grid.proc_layout[0]
                if row + 1 < config_grid.proc_layout[1]
                else None,
            ),
        )

        dpx = (config_grid.xlimit[1] - config_grid.xlimit[0]) / config_grid.proc_layout[
            0
        ]
        dpy = (config_grid.ylimit[1] - config_grid.ylimit[0]) / config_grid.proc_layout[
            1
        ]
        xlimit = (
            config_grid.xlimit[0] + dpx * col,
            config_grid.xlimit[0] + dpx * (col + 1),
        )
        ylimit = (
            config_grid.ylimit[0] + dpy * row,
            config_grid.ylimit[0] + dpy * (row + 1),
        )

        return cls(
            ni=int(config_grid.ni / config_grid.proc_layout[0]),
            nj=int(config_grid.nj / config_grid.proc_layout[1]),
            nk=config_grid.nk,
            xlimit=xlimit,
            ylimit=ylimit,
            procs=procs,
            position=(col, row),
            proc_layout=config_grid.proc_layout,
            comm=comm,
        )

    @property
    def __dict__(self):
        """
        Represent all the serializable attributes in a Python dictionary.

        The comm cannot be serialized.

        Returns:
            dict of the serializable attributes

        """
        # A previous version of this used dataclasses.asdict, but that did not work
        # because the copy.deepcopy call happens before the dict_factory method is applied,
        # so it was not possible to filter out the comm before then.
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "comm"
        }

    def to_json(self) -> str:
        """Return a yaml representation of the Grid.

        Returns:
            yaml str

        """
        return json.dumps(self.__dict__)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "Grid":
        """Create a grid from arguments, usually after deserializing.

        Parameters:
            kwargs: Grid attributes (see class definition)

        Returns:
            Initialized grid

        """
        # JSON serialization converts the tuples to lists,
        # so we need to convert that back here
        proc: List[List[Optional[int]]] = kwargs["procs"]
        kwargs["procs"] = (tuple(proc[0]), tuple(proc[1]))

        kwargs["xlimit"] = tuple(kwargs["xlimit"])
        kwargs["ylimit"] = tuple(kwargs["ylimit"])
        kwargs["position"] = tuple(kwargs["position"])
        kwargs["proc_layout"] = tuple(kwargs["proc_layout"])

        return dacite.from_dict(cls, data=kwargs, config=dacite.Config(strict=True))

    def __eq__(self, other) -> bool:
        """Check the equality of grids.

        Parameters:
            other: Grid to check

        Returns:
            Equality as a bool

        """
        if not isinstance(other, Grid):
            return False

        if (self.ni, self.nj, self.nk) != (other.ni, other.nj, other.nk):
            return False

        if self.procs != other.procs:
            return False

        return True

    @property
    def dx(self) -> float:
        """Compute the x grid spacing.

        Returns:
            floating point number

        """
        return (self.xlimit[1] - self.xlimit[0]) / (self.ni - 1)

    @property
    def dy(self) -> float:
        """Compute the y grid spacing.

        Returns:
            floating point number.

        """
        return (self.ylimit[1] - self.ylimit[0]) / (self.nj - 1)

    @classmethod
    def make_global(cls, grid: "Grid") -> "Grid":
        """Create a global grid.

        Parameters:
            grid: The parallelized grid to globalize

        Returns:
            Globalized grid

        """
        my_rank = grid.comm.Get_rank()

        row = my_rank // grid.proc_layout[0]
        col = my_rank % grid.proc_layout[0]

        xlimit = (
            grid.xlimit[0] - col * grid.dx,
            grid.xlimit[1] + (grid.proc_layout[0] - 1 - col) * grid.dx,
        )
        ylimit = (
            grid.ylimit[0] - row * grid.dy,
            grid.ylimit[1] + (grid.proc_layout[1] - 1 - row) * grid.dy,
        )
        return cls(
            ni=grid.ni * grid.proc_layout[0],
            nj=grid.nj * grid.proc_layout[1],
            nk=grid.nk,
            xlimit=xlimit,
            ylimit=ylimit,
            procs=((None, None), (None, None)),
            position=(0, 0),
            proc_layout=(1, 1),
            comm=NullComm(0, num_ranks=1),
        )


def get_position_arrays(
    *,
    nhalo: int,
    shape: Tuple[int, int],
    xlimit: Tuple[float, float],
    ylimit: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the meshgrid of x and y cell center positions.

    Parameters:
        nhalo: number of state array halos
        shape: the i and j array extents
        xlimit: grid geometry for the x direction
        ylimit: grid geometry for the y direction

    Returns:
        X, Y meshgrid ndarrays

    """
    dx = (xlimit[1] - xlimit[0]) / (shape[0] - 2 * nhalo)
    dy = (ylimit[1] - ylimit[0]) / (shape[1] - 2 * nhalo)
    extended_xlimit = (xlimit[0] - nhalo * dx, xlimit[1] + nhalo * dx)
    extended_ylimit = (ylimit[0] - nhalo * dy, ylimit[1] + nhalo * dy)

    xl = np.linspace(*extended_xlimit, shape[0])
    yl = np.linspace(*extended_ylimit, shape[1])

    X, Y = tuple(np.meshgrid(xl, yl))

    # Shift positions from cell edges to centers
    X += 0.5 * dx
    Y += 0.5 * dy

    return X, Y
