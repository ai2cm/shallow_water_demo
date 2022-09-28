"""The shallow water finite difference model."""
import enum
import typing
from typing import Any, Callable, Dict

import gt4py
import numpy as np
from gt4py.gtscript import IJ, IJK, PARALLEL, Field, computation, interval

from . import config
from .grid import Grid, get_position_arrays
from .state import State


class InitialCondition(enum.Enum):
    """Known intial conditions.

    There are functions prefixed with _set_ below for each of these.

    """

    TidalWave = "tidal_wave"
    Quiescent = "quiescent"


def _set_ic_quiescent(state: State, **kwargs: Any) -> None:
    """Freestream quiescent state that should be maintained by model time integration.

    Parameters:
        state: Arrays and other info
        kwargs: Inputs for the quiescent function.

    """
    state.h[:] = state.dtype(kwargs.get("h", 1.0))
    state.u[:] = state.dtype(kwargs.get("u", 0.0))
    state.v[:] = state.dtype(kwargs.get("v", 0.0))


def _set_ic_tidalwave(state: State, **kwargs: Any) -> None:
    """Spike in height at the midpoint with a Gaussian profile.

    Parameters:
        state: Arrays and other info
        kwargs: Inputs for the tidalwave function

    """
    grid = state.grid

    h0: float
    if v := kwargs.get("h0") is not None:
        h0 = v
    else:
        h0 = (grid.xlimit[1] - grid.xlimit[0]) / kwargs.get("xfactor", 4) + (
            grid.ylimit[1] - grid.ylimit[0]
        ) / kwargs.get("yfactor", 4)

    x, y = get_position_arrays(
        nhalo=state.nhalo,
        shape=state.h.shape,
        xlimit=grid.xlimit,
        ylimit=grid.ylimit,
    )

    xm = grid.xlimit[0] + (grid.xlimit[1] - grid.xlimit[0]) / 2
    ym = grid.ylimit[0] + (grid.ylimit[1] - grid.ylimit[0]) / 2

    sigma: float
    if v := kwargs.get("sigma") is not None:
        sigma = v
    else:
        sigma = (grid.xlimit[1] - grid.xlimit[0]) / 20

    dist_sqrt = np.sqrt((x - xm) ** 2 + (y - ym) ** 2)
    for k in range(grid.nk):
        state.h[:, :, k] = h0 + kwargs.get("height_factor", 1) * np.exp(
            -dist_sqrt / sigma
        )

    state.u[:] = 0
    state.v[:] = 0


def set_initial_condition(state: State, ic_type: str, **kwargs: Any) -> None:
    """Set an initial condition on the state.

    Raises:
        ValueError: if the initial condition type is unrecognized

    Notes:
        Modifies the state in place.

    Parameters:
        state: Data arrays
        ic_type: The initial condition to set
        kwargs: Initial condition data parameters (usually from the Config)

    """
    if ic_type == str(InitialCondition.Quiescent.value):
        _set_ic_quiescent(state, **kwargs)
    elif ic_type == str(InitialCondition.TidalWave.value):
        _set_ic_tidalwave(state, **kwargs)
    else:
        raise ValueError(
            f"Unrecognized initial condition {ic_type}. "
            "See shallow_water.InitialCondition for options."
        )


class ShallowWaterModel:
    """A model that integrates the Shallow Water equationset using a simple first-order method."""

    dtype: np.dtype = np.float64
    nhalo: int = 1
    g: float = 9.81

    timestep: np.float64

    b: gt4py.storage.Storage
    state: State
    _new_state: State

    grid: Grid

    _integrate: Callable[..., None]

    def __init__(self, grid: Grid, model_config: config.Model):
        """Initialize the model.

        Parameters:
            grid: Geometry information
            model_config: Configuration

        """
        self.b = gt4py.storage.zeros(
            backend=model_config.gt4py_backend,
            default_origin=(1, 1),
            shape=(grid.ni + 2, grid.nj + 2),
            dtype=self.dtype,
            mask=gt4py.gtscript.mask_from_axes(IJ),
        )

        self.state = State.new(
            grid,
            self.dtype,
            nhalo=self.nhalo,
            gt4py_backend=model_config.gt4py_backend,
            value=0,
        )

        self._new_state = State.new(
            grid,
            self.dtype,
            nhalo=0,
            gt4py_backend=model_config.gt4py_backend,
            value=None,
        )

        self.timestep = model_config.timestep
        self.grid = grid

        set_initial_condition(
            self.state, model_config.initial_condition, **model_config.ic_data
        )

        FieldIJK = Field[IJK, self.dtype]
        FieldIJ = Field[IJ, self.dtype]

        @typing.no_type_check
        def integrate(
            b: FieldIJ,
            h: FieldIJK,
            u: FieldIJK,
            v: FieldIJK,
            h_new: FieldIJK,
            u_new: FieldIJK,
            v_new: FieldIJK,
            dt: self.dtype,
            dx: self.dtype,
            dy: self.dtype,
        ):
            """GT4Py stencil that applys the finite difference update on the interior.

            Parameters:
                b: Bottom height (input)
                h: Fluid height (input)
                u: X-velocity (input)
                v: Y-velocity (input)
                h_new: Fluid height after step (output)
                u_new: X-velocity after step (output)
                v_new: Y-velocity after step (output)
                dt: Timestamp (input)
                dx: X-cell spacing (input)
                dy: Y-cell spacing (input)

            """
            from __externals__ import g

            with computation(PARALLEL), interval(...):
                u_avg = (u[1, 0, 0] + u[-1, 0, 0] + u[0, 1, 0] + u[0, -1, 0]) / 4.0
                u2_diff = (
                    u[1, 0, 0] * u[1, 0, 0] / 2.0 - u[-1, 0, 0] * u[-1, 0, 0] / 2.0
                )
                vu_diff = v[0, 0, 0] * (u[0, 1, 0] - u[0, -1, 0])
                h_diff = h[1, 0, 0] - h[-1, 0, 0]
                u_new = (  # noqa: F841
                    u_avg
                    - 0.5 * dt * dx * u2_diff
                    - 0.5 * dt * dy * vu_diff
                    - 0.5 * g * dt * dx * h_diff
                )

                v_avg = (v[1, 0, 0] + v[-1, 0, 0] + v[0, 1, 0] + v[0, -1, 0]) / 4.0
                v2_diff = (
                    v[0, 1, 0] * v[0, 1, 0] / 2.0 - v[0, -1, 0] * v[0, -1, 0] / 2.0
                )
                uv_diff = u[0, 0, 0] * (v[1, 0, 0] - u[-1, 0, 0])
                h_diff = h[0, 1, 0] - h[0, -1, 0]
                v_new = (  # noqa: F841
                    v_avg
                    - 0.5 * dt * dy * v2_diff
                    - 0.5 * dt * dx * uv_diff
                    - 0.5 * g * dt * dy * h_diff
                )

                h_avg = (h[1, 0, 0] + h[-1, 0, 0] + h[0, 1, 0] + h[0, -1, 0]) / 4.0
                hu_diff = u * ((h[1, 0, 0] - b[1, 0]) - (h[-1, 0, 0] - b[-1, 0]))
                hv_diff = v * ((h[0, 1, 0] - b[0, 1]) - (h[0, -1, 0] - b[0, -1]))
                hud = (h - b) * (u[1, 0, 0] - u[-1, 0, 0])
                hvd = (h - b) * (v[0, 1, 0] - v[0, -1, 0])
                h_new = (  # noqa: F841
                    h_avg
                    - 0.5 * dt * dx * hu_diff
                    - 0.5 * dt * dy * hv_diff
                    - 0.5 * dt * dx * hud
                    - 0.5 * dt * dy * hvd
                )

        opts: Dict[str, Any] = {"externals": {"g": self.g}}
        if "numpy" not in model_config.gt4py_backend:
            opts["verbose"] = True

        self._integrate = gt4py.gtscript.stencil(  # type: ignore
            definition=integrate, backend=model_config.gt4py_backend, **opts
        )

    def _apply_boundary_conditions(self):
        """Set Dirichlet reflection boundary conditions."""
        nh = self.state.nhalo
        if self.state.grid.procs[0][0] is None:
            self.state.h[nh - 1, :, :] = self.state.h[nh, :, :]
            self.state.u[nh - 1, :, :] = -self.state.u[nh, :, :]
            self.state.v[nh - 1, :, :] = self.state.v[nh, :, :]

        if self.state.grid.procs[0][1] is None:
            self.state.h[1 - nh, :, :] = self.state.h[-nh, :, :]
            self.state.u[1 - nh, :, :] = -self.state.u[-nh, :, :]
            self.state.v[1 - nh, :, :] = self.state.v[-nh, :, :]

        if self.state.grid.procs[1][0] is None:
            self.state.h[:, nh - 1, :] = self.state.h[:, nh, :]
            self.state.u[:, nh - 1, :] = self.state.u[:, nh, :]
            self.state.v[:, nh - 1, :] = -self.state.v[:, nh, :]

        if self.state.grid.procs[1][1] is None:
            self.state.h[:, 1 - nh, :] = self.state.h[:, -nh, :]
            self.state.u[:, 1 - nh, :] = self.state.u[:, -nh, :]
            self.state.v[:, 1 - nh, :] = -self.state.v[:, -nh, :]

    def take_step(self):
        """Take a timestep."""
        self.state.exchange_halos()

        self._apply_boundary_conditions()

        self._integrate(
            h=self.state.h,
            u=self.state.u,
            v=self.state.v,
            h_new=self._new_state.h,
            u_new=self._new_state.u,
            v_new=self._new_state.v,
            b=self.b,
            dt=self.timestep,
            dx=self.grid.dx,
            dy=self.grid.dy,
        )

        nh = self.nhalo
        self.state.h[nh:-nh, nh:-nh, :] = self._new_state.h[:, :, :]
        self.state.u[nh:-nh, nh:-nh, :] = self._new_state.u[:, :, :]
        self.state.v[nh:-nh, nh:-nh, :] = self._new_state.v[:, :, :]
