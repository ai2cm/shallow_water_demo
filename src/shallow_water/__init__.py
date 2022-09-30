"""A simple shallow water model."""

from . import driver  # noqa: F401
from .communicator import Comm  # noqa: F401
from .config import Config  # noqa: F401
from .grid import Grid  # noqa: F401
from .model import InitialCondition, ShallowWaterModel  # noqa: F401
from .state import State  # noqa: F401
