import pytest
import yaml

import shallow_water
import shallow_water.config


def test_config():
    input_str = """
model:
    num_steps: 10
    timestep: 0.1
grid:
    ni: 8
    nj: 10
    nk: 3
    proc_layout:
    - 2
    - 2
"""
    data = yaml.safe_load(input_str)
    config = shallow_water.Config.from_dict(**data)

    assert isinstance(config, shallow_water.Config)
    assert config.model.num_steps == 10
    assert config.grid.ni == 8

    output_str = config.to_yaml()
    assert output_str


def test_is_not_integer():
    input_str = """
model:
    num_steps: 10
    timestep: 0.1
grid:
    ni: 8
    nj: 10
    nk: 3
    proc_layout:
    - 2
    - 3
"""
    data = yaml.safe_load(input_str)
    with pytest.raises(ValueError, match="exactly without remainder"):
        shallow_water.Config.from_dict(**data)
