from tempfile import TemporaryDirectory

import numpy as np
import pytest

import shallow_water
from shallow_water.communicator import NullComm


@pytest.fixture
def grid():
    proc_layout = (2, 2)
    my_rank = 0
    config_grid = shallow_water.config.Grid(ni=8, nj=8, nk=2, proc_layout=proc_layout)
    return shallow_water.grid.Grid.from_config(config_grid, NullComm(my_rank, 1))


@pytest.mark.parametrize("value", (None, 0, 1, 7))
def test_new_state(grid, value):
    if value is None or value < 2:
        shallow_water.State.new(grid, np.float64, value=value)
    else:
        with pytest.raises(ValueError, match="value"):
            shallow_water.State.new(grid, np.float64, value=value)


@pytest.mark.parametrize(
    "other",
    (
        0,
        shallow_water.grid.Grid.from_config(
            shallow_water.config.Grid(ni=8, nj=8, nk=2, proc_layout=(2, 3)),
            NullComm(0, 1),
        ),
        shallow_water.grid.Grid.from_config(
            shallow_water.config.Grid(ni=6, nj=8, nk=2, proc_layout=(2, 2)),
            NullComm(0, 1),
        ),
    ),
)
def test_equality(grid, other):
    assert grid != other


def test_serialize_state(grid):
    state = shallow_water.State.new(grid, nhalo=2, dtype=np.float64)
    with TemporaryDirectory() as tempdir:
        state.to_disk(prefix=f"{tempdir}/test")

        newstate = shallow_water.State.from_disk(
            prefix=f"{tempdir}/test", comm=grid.comm
        )
        assert np.allclose(newstate.h.data, state.h.data)
        assert newstate.grid == state.grid
