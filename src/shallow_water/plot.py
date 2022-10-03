"""Plot the shallow_water model state."""

import argparse
from typing import Any, Optional

import matplotlib.figure
import matplotlib.pyplot as plt

from .state import State


def plot_state(
    state: State, component: Optional[str], **kwargs: Any
) -> matplotlib.figure.Figure:
    """Return a figure with the state component or components plotted.

    Parameters:
        state: shallow water model state arrays
        component: component of the state to plot
        kwargs: other keyword arguments to pass to imshow

    Returns:
        Figure

    """
    if component is None:
        fig, axs = plt.subplots(2, 2)

        axs[0, 0].imshow(state.h[:, :, 0], **kwargs)
        axs[0, 0].set_title("height")
        axs[1, 0].imshow(state.u[:, :, 0], **kwargs)
        axs[1, 0].set_title("x-velocity")
        axs[0, 1].imshow(state.v[:, :, 0], **kwargs)
        axs[0, 1].set_title("y-velocity")

        axs[1, 1].remove()
    else:
        fig = plt.figure()
        ax = fig.gca()

        arr = getattr(state, component)
        assert arr is not None
        ax.imshow(arr[:, :, 0], **kwargs)

    return fig


def run(prefix: str, component: Optional[str]):
    """Deserialize state at 'prefix' and plot component.

    Parameters:
        prefix: State prefix
        component: variable in the state to plot {h, u, v} (optional)

    """
    state = State.from_disk(prefix, comm=None)
    plot_state(state, component)
    plt.show()


def main():
    """Driver that plots the shallow_water state."""
    parser = argparse.ArgumentParser(description="Plot the shallow_water state")
    parser.add_argument("prefix", type=str, help="state prefix")
    parser.add_argument("--component", "-c", type=str, help="State component to plot")

    args = parser.parse_args()

    run(args.prefix, args.component)


if __name__ == "__main__":
    main()
