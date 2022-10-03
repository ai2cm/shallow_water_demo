# """Simple test for plotting."""

import numpy as np
import pytest

try:
    import matplotlib  # noqa
except ImportError:
    pytest.skip("Matplotlib not available", allow_module_level=True)
import shallow_water.plot
from shallow_water.communicator import NullComm


@pytest.fixture
def grid():
    proc_layout = (2, 2)
    my_rank = 0
    config_grid = shallow_water.config.Grid(ni=8, nj=8, nk=2, proc_layout=proc_layout)
    return shallow_water.grid.Grid.from_config(
        config_grid, NullComm(my_rank, num_ranks=1)
    )


@pytest.fixture
def state(grid):
    return shallow_water.state.State.new(grid, np.float64)


@pytest.mark.parametrize("component", (None, "h", "u", "v"))
def test_plot_state(state, component):
    assert shallow_water.plot.plot_state(state, component) is not None
