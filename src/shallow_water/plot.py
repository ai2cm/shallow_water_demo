"""Plot the shallow_water model state."""

import argparse
from typing import Any, Optional

import matplotlib.figure
import matplotlib.pyplot as plt

from .state import State


def plot_state(
    state: State, nhalo: int, component: Optional[str], **kwargs: Any
) -> matplotlib.figure.Figure:
    """Return a figure with the state component or components plotted.

    Parameters:
        state: shallow water model state arrays
        nhalo: Number of halo points to avoid plotting
        component: component of the state to plot
        kwargs: other keyword arguments to pass to imshow

    Returns:
        Figure

    """

    def slice_array(arr):
        if nhalo > 0:
            return arr[nhalo:-nhalo, nhalo:-nhalo, 0].T
        else:
            return arr[:, :, 0].T

    extent = list(state.grid.xlimit) + list(state.grid.ylimit)
    if component is None:
        fig, axs = plt.subplots(1, 3)

        axs[0].imshow(slice_array(state.h), extent=extent, **kwargs)
        axs[0].set_title("height")
        axs[1].imshow(slice_array(state.u), extent=extent, **kwargs)
        axs[1].set_title("x-velocity")
        axs[2].imshow(slice_array(state.v), extent=extent, **kwargs)
        axs[2].set_title("y-velocity")

    else:
        fig = plt.figure()
        ax = fig.gca()

        arr = getattr(state, component)
        assert arr is not None
        ax.imshow(slice_array(arr), extent=extent, **kwargs)

    return fig


def run(prefix: str, nhalo: int, component: Optional[str], output: Optional[str]):
    """Deserialize state at 'prefix' and plot component.

    Parameters:
        prefix: State prefix
        nhalo: Number of halo points to avoid plotting
        component: variable in the state to plot {h, u, v} (optional)
        output: filename to save plot

    """
    state = State.from_disk(prefix, comm=None)
    fig = plot_state(state, nhalo, component)

    if output:
        fig.savefig(output)
    else:
        plt.show()


def main():
    """Driver that plots the shallow_water state."""
    parser = argparse.ArgumentParser(description="Plot the shallow_water state")
    parser.add_argument("prefix", type=str, help="state prefix")
    parser.add_argument(
        "--nhalo",
        "-n",
        type=int,
        default=0,
        help="number of halo points to avoid plotting",
    )
    parser.add_argument("--component", "-c", type=str, help="State component to plot")
    parser.add_argument("--output", "-o", type=str, help="output filename")

    args = parser.parse_args()

    run(args.prefix, args.nhalo, args.component, args.output)


if __name__ == "__main__":
    main()
