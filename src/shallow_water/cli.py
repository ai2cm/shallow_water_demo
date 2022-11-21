"""The command line interface driver."""

import argparse

from mpi4py import MPI

import shallow_water.driver


def main() -> None:
    """Driver for the shallow water model that parses command line arguments."""
    parser = argparse.ArgumentParser(description="Run the shallow water model.")
    parser.add_argument("config_file", type=str, help="yaml configuration")
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="output",
        help="Directory to save output",
    )
    parser.add_argument(
        "-f",
        "--output-frequency",
        type=int,
        default=-1,
        help="frequency with which to save the state: "
        "f<0: never, f=0: final state only, f>0: every f steps and last",
    )
    parser.add_argument(
        "-g",
        "--globalize",
        action="store_true",
        help="gather state after integrating and serialize to disk",
    )
    args = parser.parse_args()

    shallow_water.driver.run(
        args.config_file,
        args.directory,
        args.output_frequency,
        MPI.COMM_WORLD,
        globalize=args.globalize,
    )


if __name__ == "__main__":
    main()
