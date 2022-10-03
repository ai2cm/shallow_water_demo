"""Test the shallow water model."""

import json

import pytest

import shallow_water
from shallow_water.communicator import NullComm


@pytest.mark.parametrize(
    "proc_layout,my_rank,procs",
    [
        ((3, 4), 0, ((None, 1), (None, 3))),
        ((3, 4), 1, ((0, 2), (None, 4))),
        ((3, 4), 7, ((6, 8), (4, 10))),
    ],
)
def test_proc_layout(proc_layout, my_rank, procs):
    config_grid = shallow_water.config.Grid(ni=12, nj=16, nk=2, proc_layout=proc_layout)
    grid = shallow_water.Grid.from_config(config_grid, NullComm(my_rank, num_ranks=1))
    assert grid.procs == procs


def test_serialize():
    config_grid = shallow_water.config.Grid(ni=12, nj=16, nk=2, proc_layout=(2, 2))
    comm = NullComm(0, num_ranks=1)

    grid = shallow_water.Grid.from_config(config_grid, comm=comm)
    data = json.loads(grid.to_json())

    newgrid = shallow_water.Grid.from_kwargs(comm=comm, **data)

    assert newgrid == grid
