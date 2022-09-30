"""MPI-related abstract classes and mocks."""

from abc import abstractmethod
from typing import Any

from mpi4py import MPI


class Comm:
    """MPI Communicator.

    Notes:
        Prototype for mpi4py.MPI.Communicator.
    """

    @abstractmethod
    def Isend(self, data, *, dest: int, tag: int) -> MPI.Request:  # noqa: D102
        raise NotImplementedError("TBD")

    @abstractmethod
    def Irecv(self, data, *, source: int, tag: int) -> MPI.Request:  # noqa: D102
        raise NotImplementedError("TBD")

    @abstractmethod
    def gather(self, data: Any, *, root: int) -> Any:  # noqa: D102
        raise NotImplementedError("TBD")

    @abstractmethod
    def Get_size(self) -> int:  # noqa: D102
        raise NotImplementedError("TBD")

    @abstractmethod
    def Get_rank(self) -> int:  # noqa: D102
        raise NotImplementedError("TBD")


class NullRequest(MPI.Request):
    """Dummy implementation of Request, used for serial tests."""

    def wait(self) -> Any:
        """Never wait.

        Returns:
            True

        """
        return True


class NullComm(Comm):
    """Dummy implementation of Comm, used for serial tests."""

    def __init__(self, rank: int, *, num_ranks: int):
        """Initialize Comm pretending to be a particular rank.

        Parameters:
            rank: mock rank
            num_ranks: number of mocked ranks

        """
        self._rank = rank
        self._num_ranks = num_ranks

    def Isend(self, data, *, dest: int, tag: int) -> MPI.Request:  # noqa: D102
        return NullRequest()

    def Irecv(self, data, *, source: int, tag: int) -> MPI.Request:  # noqa: D102
        return NullRequest()

    def gather(self, data, *, root: int) -> Any:
        """Gathering on a NullComm returns the argument passed on the proc, otherwise None.

        Parameters:
            data: to gather
            root: the proc that receives the gather

        Returns:
            data if _rank == root else None

        """
        return data if self.Get_rank() == root else None

    def Get_rank(self) -> int:
        """Get the rank specified.

        Returns:
            Mocked rank.

        """
        return self._rank

    def Get_size(self) -> int:
        """Get the total number of ranks on the communicator.

        Returns:
            Number of ranks mocked.

        """
        return self._num_ranks
