"""Shallow water state storages."""

import dataclasses
import json
from typing import Any, List, Optional

import gt4py
import numpy as np
from mpi4py import MPI

from .communicator import Comm, NullComm
from .grid import Grid


def _make_storage(
    grid: Grid,
    dtype: Any,
    *,
    nhalo: int = 2,
    gt4py_backend: str = "numpy",
    value: Optional[int] = None,
):
    """Create a storage on the grid points.

    Currently supports cell-centered quantities.

    Parameters:
        grid: The shallow_water grid
        dtype: Either built-in or numpy dtype
        nhalo: Halo points in each horizontal direction on storage
        gt4py_backend: Backend string
        value: Initial fill value (if not None)

    Raises:
        ValueError: if 'value' is not supported

    Returns:
        gt4py storage

    """
    aligned_index = (nhalo, nhalo, 0)
    shape = (grid.ni + 2 * nhalo, grid.nj + 2 * nhalo, grid.nk)

    if value is None:
        func = gt4py.storage.empty
    elif value == 0:
        func = gt4py.storage.zeros
    elif value == 1:
        func = gt4py.storage.ones
    else:
        raise ValueError(f"Unsupported value={value}. Options: {{None, 0, 1}}")

    return func(
        backend=gt4py_backend,
        aligned_index=aligned_index,
        shape=shape,
        dtype=dtype,
        dimensions=gt4py.gtscript.IJK,
    )


class _StateHaloExchanger:
    """Class used to exchange state halo data."""

    up_send: Optional[np.ndarray]
    up_recv: Optional[np.ndarray]

    down_send: Optional[np.ndarray]
    down_recv: Optional[np.ndarray]

    left_send: Optional[np.ndarray]
    left_recv: Optional[np.ndarray]

    right_send: Optional[np.ndarray]
    right_recv: Optional[np.ndarray]

    initialized: bool = False

    def _initialize(self, state: "State") -> None:
        """Initialize the halo exchange arrays.

        Parameters:
            state: arrays and other info

        """
        narray = 3

        grid = state.grid
        if grid.procs[0][0] is not None:
            self.left_send = np.empty(
                (state.nhalo, grid.nj, grid.nk, narray), dtype=state.dtype
            )
            self.left_recv = np.empty(
                (state.nhalo, grid.nj, grid.nk, narray), dtype=state.dtype
            )

        if grid.procs[0][1] is not None:
            self.right_send = np.empty(
                (state.nhalo, grid.nj, grid.nk, narray), dtype=state.dtype
            )
            self.right_recv = np.empty(
                (state.nhalo, grid.nj, grid.nk, narray), dtype=state.dtype
            )

        if grid.procs[1][0] is not None:
            self.down_send = np.empty(
                (grid.ni, state.nhalo, grid.nk, narray), dtype=state.dtype
            )
            self.down_recv = np.empty(
                (grid.ni, state.nhalo, grid.nk, narray), dtype=state.dtype
            )

        if grid.procs[1][1] is not None:
            self.up_send = np.empty(
                (grid.ni, state.nhalo, grid.nk, narray), dtype=state.dtype
            )
            self.up_recv = np.empty(
                (grid.ni, state.nhalo, grid.nk, narray), dtype=state.dtype
            )

    def _pack(self, state: "State") -> None:
        """Pack the halo exchange arrays.

        Parameters:
            state: arrays and other info

        """
        grid = state.grid
        nh = state.nhalo
        if grid.procs[0][0] is not None:
            assert self.left_send is not None
            self.left_send[:, :, :, 0] = state.h[:nh, nh:-nh, :]
            self.left_send[:, :, :, 1] = state.u[:nh, nh:-nh, :]
            self.left_send[:, :, :, 2] = state.v[:nh, nh:-nh, :]

        if grid.procs[0][1] is not None:
            assert self.right_send is not None
            self.right_send[:, :, :, 0] = state.h[-nh:, nh:-nh, :]
            self.right_send[:, :, :, 1] = state.u[-nh:, nh:-nh, :]
            self.right_send[:, :, :, 2] = state.v[-nh:, nh:-nh, :]

        if grid.procs[1][0] is not None:
            assert self.down_send is not None
            self.down_send[:, :, :, 0] = state.h[nh:-nh, :nh, :]
            self.down_send[:, :, :, 1] = state.u[nh:-nh, :nh, :]
            self.down_send[:, :, :, 2] = state.v[nh:-nh, :nh, :]

        if grid.procs[1][1] is not None:
            assert self.up_send is not None
            self.up_send[:, :, :, 0] = state.h[nh:-nh, -nh:, :]
            self.up_send[:, :, :, 1] = state.u[nh:-nh, -nh:, :]
            self.up_send[:, :, :, 2] = state.v[nh:-nh, -nh:, :]

    def _unpack(self, state: "State") -> None:
        """Unpack the halo exchange data back into the state arrays.

        Parameters:
            state: arrays and other info

        """
        grid = state.grid
        nh = state.nhalo
        if grid.procs[0][1] is not None:
            assert self.right_recv is not None
            state.h[-nh:, nh:-nh, :] = np.flip(self.right_recv[:, :, :, 0], axis=0)
            state.u[-nh:, nh:-nh, :] = np.flip(self.right_recv[:, :, :, 1], axis=0)
            state.v[-nh:, nh:-nh, :] = np.flip(self.right_recv[:, :, :, 2], axis=0)

        if grid.procs[0][0] is not None:
            assert self.left_recv is not None
            state.h[:nh, nh:-nh, :] = np.flip(self.left_recv[:, :, :, 0], axis=0)
            state.u[:nh, nh:-nh, :] = np.flip(self.left_recv[:, :, :, 1], axis=0)
            state.v[:nh, nh:-nh, :] = np.flip(self.left_recv[:, :, :, 2], axis=0)

        if grid.procs[1][1] is not None:
            assert self.up_recv is not None
            state.h[nh:-nh, -nh:, :] = np.flip(self.up_recv[:, :, :, 0], axis=1)
            state.u[nh:-nh, -nh:, :] = np.flip(self.up_recv[:, :, :, 1], axis=1)
            state.v[nh:-nh, -nh:, :] = np.flip(self.up_recv[:, :, :, 2], axis=1)

        if grid.procs[1][0] is not None:
            assert self.down_recv is not None
            state.h[nh:-nh, :nh, :] = np.flip(self.down_recv[:, :, :, 0], axis=1)
            state.u[nh:-nh, :nh, :] = np.flip(self.down_recv[:, :, :, 1], axis=1)
            state.v[nh:-nh, :nh, :] = np.flip(self.down_recv[:, :, :, 2], axis=1)

    def exchange(self, state: "State") -> None:
        """Exchange the halos from state.

        Notes:
            Modifies state in place.

        Parameters:
            state: Arrays and other info

        """
        grid = state.grid
        if not self.initialized:
            self._initialize(state)

        recv_requests: List[MPI.Request] = []
        if grid.procs[0][0] is not None:
            recv_requests.append(
                grid.comm.Irecv(self.left_recv, source=grid.procs[0][0], tag=1)
            )

        if grid.procs[0][1] is not None:
            recv_requests.append(
                grid.comm.Irecv(self.right_recv, source=grid.procs[0][1], tag=0)
            )

        if grid.procs[1][0] is not None:
            recv_requests.append(
                grid.comm.Irecv(self.down_recv, source=grid.procs[1][0], tag=11)
            )

        if grid.procs[1][1] is not None:
            recv_requests.append(
                grid.comm.Irecv(self.up_recv, source=grid.procs[1][1], tag=10)
            )

        self._pack(state)
        send_requests: List[MPI.Request] = []
        if grid.procs[0][0] is not None:
            send_requests.append(
                grid.comm.Isend(self.left_send, dest=grid.procs[0][0], tag=0)
            )

        if grid.procs[0][1] is not None:
            send_requests.append(
                grid.comm.Isend(self.right_send, dest=grid.procs[0][1], tag=1)
            )

        if grid.procs[1][0] is not None:
            send_requests.append(
                grid.comm.Isend(self.down_send, dest=grid.procs[1][0], tag=10)
            )

        if grid.procs[1][1] is not None:
            send_requests.append(
                grid.comm.Isend(self.up_send, dest=grid.procs[1][1], tag=11)
            )

        MPI.Request.waitall(recv_requests)

        # Can start unpacking
        self._unpack(state)

        # Wait at the end for all sends to finish
        MPI.Request.waitall(send_requests)


@dataclasses.dataclass
class State:
    """State storages."""

    h: np.ndarray
    u: np.ndarray
    v: np.ndarray
    nhalo: int
    gt4py_backend: str
    dtype: np.dtype
    grid: Grid = dataclasses.field(repr=False)

    halo_exchanger = _StateHaloExchanger()

    _is_global: bool = False

    @classmethod
    def new(
        cls,
        grid: Grid,
        dtype: np.dtype,
        *,
        nhalo: int = 2,
        gt4py_backend: str = "numpy",
        value: Optional[int] = 0,
        _is_global: bool = False,
    ) -> "State":
        """Create new shallow water state storages.

        Parameters:
            grid: Geomery information
            dtype: Data type (either built-in or numpy)
            nhalo: Number of halo points in each horizontal direction on each storage
            gt4py_backend: String that determines the compiler backend
            value: Initial value for the state data arrays
            _is_global: Set to True if this is a global gathered state

        Returns:
            Class instance

        """
        return cls(
            h=_make_storage(
                grid, dtype, nhalo=nhalo, gt4py_backend=gt4py_backend, value=value
            ),
            u=_make_storage(
                grid, dtype, nhalo=nhalo, gt4py_backend=gt4py_backend, value=value
            ),
            v=_make_storage(
                grid, dtype, nhalo=nhalo, gt4py_backend=gt4py_backend, value=value
            ),
            nhalo=nhalo,
            gt4py_backend=gt4py_backend,
            dtype=dtype,
            grid=grid,
            _is_global=_is_global,
        )

    def to_disk(self, prefix: str) -> None:
        """Serialize the data to disk.

        Notes:
            Does not serialize the MPI communicator.

        Parameters:
            prefix: Path and file prefix

        """
        tag: str
        if self._is_global:
            tag = "global"
        else:
            tag = str(self.grid.comm.Get_rank())

        with open(f"{prefix}_{tag}_grid.json", mode="w") as f:
            f.write(self.grid.to_json())
        for k, v in {"h": self.h, "u": self.u, "v": self.v}.items():
            np.save(f"{prefix}_{tag}_state_{k}.npy", v)
        with open(f"{prefix}_{tag}_state_metadata.json", mode="w") as f:
            f.write(
                json.dumps(
                    {
                        "gt4py_backend": self.gt4py_backend,
                        "nhalo": self.nhalo,
                        "dtype": np.dtype(self.dtype).str,
                    }
                )
            )

    @classmethod
    def from_disk(cls, prefix: str, *, comm: Optional[Comm] = None) -> "State":
        """Deserialize the data from disk.

        Notes:
            The resulting state will not be valid to use
            in a model if 'comm' is None.

        Parameters:
            prefix: Path and file prefix
            comm: MPI communicator prefixed if passed

        Returns:
            State that was serialized.

        """
        if comm is not None:
            prefix = f"{prefix}_{comm.Get_rank()}"

        with open(f"{prefix}_grid.json", mode="r") as f:
            input_data = json.loads(f.read())

        input_data["comm"] = comm or NullComm(0, num_ranks=1)

        grid = Grid.from_kwargs(**input_data)

        storages = {k: np.load(f"{prefix}_state_{k}.npy") for k in ("h", "u", "v")}
        with open(f"{prefix}_state_metadata.json", mode="r") as f:
            state_data = json.loads(f.read())

        assert "dtype" in state_data
        state_data["dtype"] = np.dtype(state_data["dtype"])

        return cls(grid=grid, **state_data, **storages)

    @classmethod
    def gather_from(cls, state: "State") -> Optional["State"]:
        """Gather the state from all procs to the root (0).

        Parameters:
            state: The distributed state.

        Returns:
            Global state on the root proc (0), and None on others.

        """
        root = 0
        ldata = state.grid.comm.gather(
            {
                "h": state.h,
                "u": state.u,
                "v": state.v,
                "position": state.grid.position,
            },
            root=root,
        )

        if state.grid.comm.Get_rank() == root:
            global_grid = Grid.make_global(state.grid)
            global_state = State.new(
                global_grid,
                state.dtype,
                nhalo=state.nhalo,
                gt4py_backend=state.gt4py_backend,
                value=0,
                _is_global=True,
            )
            for data in ldata:

                nh = global_state.nhalo

                istart = nh + data["position"][0] * state.grid.ni
                iend = nh + (data["position"][0] + 1) * state.grid.ni
                jstart = nh + data["position"][1] * state.grid.nj
                jend = nh + (data["position"][1] + 1) * state.grid.nj

                global_state.h[istart:iend, jstart:jend, :] = data["h"][
                    nh:-nh, nh:-nh, :
                ]
                global_state.u[istart:iend, jstart:jend, :] = data["u"][
                    nh:-nh, nh:-nh, :
                ]
                global_state.v[istart:iend, jstart:jend, :] = data["v"][
                    nh:-nh, nh:-nh, :
                ]

            return global_state

        else:
            return None

    def exchange_halos(self) -> None:
        """Exchange halo data in place."""
        self.halo_exchanger.exchange(self)
