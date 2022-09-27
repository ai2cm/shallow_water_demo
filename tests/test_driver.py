import glob
import os
from tempfile import TemporaryDirectory

import pytest
import yaml
from mpi4py import MPI

import shallow_water


@pytest.fixture
def config_yaml() -> str:
    return """
model:
    num_steps: 6
    timestep: 0.001
grid:
    ni: 4
    nj: 4
    nk: 2
    proc_layout:
        - 1
        - 1
"""


def test_driver_output(config_yaml):
    with TemporaryDirectory() as tempdir:
        data = yaml.safe_load(config_yaml)
        config = shallow_water.Config.from_dict(**data)
        shallow_water.driver.integrate(
            config, MPI.COMM_WORLD, output_directory=tempdir, output_frequency=2
        )

        assert glob.glob(os.path.join(tempdir, "state_initial*"))
        assert glob.glob(os.path.join(tempdir, "state_step_1*"))
        assert glob.glob(os.path.join(tempdir, "state_step_3*"))
        assert glob.glob(os.path.join(tempdir, "state_final*"))


def test_driver_no_output(config_yaml):
    with TemporaryDirectory() as tempdir:
        data = yaml.safe_load(config_yaml)
        config = shallow_water.Config.from_dict(**data)
        shallow_water.driver.integrate(
            config, MPI.COMM_WORLD, output_directory=tempdir, output_frequency=-1
        )

        assert not glob.glob(os.path.join(tempdir, "state"))
