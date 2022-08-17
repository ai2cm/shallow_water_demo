"""Test the shallow water model."""

import numpy as np
import pytest

import shallow_water
import shallow_water.model
from shallow_water.communicator import NullComm


@pytest.fixture
def grid() -> shallow_water.Grid:
    proc_layout = (2, 2)
    my_rank = 0
    config_grid = shallow_water.config.Grid(ni=8, nj=8, nk=2, proc_layout=proc_layout)
    return shallow_water.grid.Grid.from_config(config_grid, NullComm(my_rank))


@pytest.mark.parametrize("ic_type", ("tidal_wave", "quiescent"))
def test_set_ic(grid: shallow_water.Grid, ic_type: str):
    state = shallow_water.State.new(grid, nhalo=2, dtype=np.float64)
    shallow_water.model.set_initial_condition(state, ic_type)

    nh = state.nhalo
    assert state.h.data[nh:-nh, nh:-nh, :].sum() > 0

    global_state = shallow_water.State.gather_from(state)
    if state.grid.comm.Get_rank() == 0:
        assert global_state is not None
        assert global_state.h.data[nh:-nh, nh:-nh, :].sum() > 0


def test_shallow_water_model_step(grid: shallow_water.Grid):
    model_config = shallow_water.config.Model(num_steps=1, timestep=1.0)
    model = shallow_water.ShallowWaterModel(grid, model_config)
    model.take_step()
