[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache License][license-shield]][license-url]

# GT4Py Shallow Water Demo

This demonstrates simple scaffolding and the core components to run a gt4py model.

## 🐝 Local installation

The demo requires Python 3.8 or 3.9, and `setuptools`. The other dependencies are
specified in [`setup.cfg`](https://github.com/ai2cm/shallow_water_demo/blob/main/setup.cfg).

To get going...

```shell
$ python3.8 -m venv venv
$ source venv/bin/activate
$ gh repo clone ai2cm/shallow_water_demo && cd shallow_water_demo
$ pip install -e "./[plot]"
```

Along with the `shallow_water` package, this also installs two executables, `shallow_water_demo` and `plot_shallow_water_state`, which can be used to run and plot the model.

As a quick example, you could run

```
$ scripts/run_shallow_water.sh examples/tidal_wave.yaml -d output -f 0 -g
$ plot_shallow_water_state output/final_global
```

[`scripts/run_shallow_water.sh`](https://github.com/ai2cm/shallow_water_demo/blob/main/scripts/run_shallow_water.sh) and is a wrapper around the `shallow_water_demo` executable that parses the config file and calls mpirun with the proper number of processes.

## 🍱 Example in a box

There is a Docker image that can be used to test and run examples.
This is pushed to the GitHub container registry at `ghcr.io/ai2cm/shallow_water_demo:main`.

In order to pull that image, you will have to authenticate docker with github, by first
creating a personal access token at https://github.com/settings/tokens/new then copying
that id and executing

```shell
$ export CR_PAT=YOUR_TOKEN
$ echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
```

More info about this in the [GitHub Docs](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

Once this is in place you can run an example with

```shell
$ docker run ghcr.io/ai2cm/shallow_water_demo:main examples/tidal_wave.yaml
```

The entrypoint of this docker container is `scripts/run_shallow_water.sh` and the default directory is the top-level `shallo_water_model` source.

A volume mount is required to actually save data back to the local disk.

An alternative to pulling the docker image from `ghcr.io` is to build it locally:

```shell
$ docker build . -t shallow_water_demo
$ docker run shallow_water_demo examples/tidal_wave.yaml
```

## 💻 Development

There are several useful development dependencies specified in [`requirements-dev.txt`](https://github.com/ai2cm/shallow_water_demo/blob/main/requirements-dev.txt).

In particular, `pre-commit` is used to lint the code before it is processed by the continuous integration system.
To lint locally, run

```shell
$ pre-commit run --all-files
```

[contributors-shield]: https://img.shields.io/github/contributors/ai2cm/shallow_water_demo.svg?style=for-the-badge
[contributors-url]: https://github.com/ai2cm/shallow_water_demo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ai2cm/shallow_water_demo.svg?style=for-the-badge
[forks-url]: https://github.com/ai2cm/shallow_water_demo/network/members
[stars-shield]: https://img.shields.io/github/stars/ai2cm/shallow_water_demo.svg?style=for-the-badge
[stars-url]: https://github.com/ai2cm/shallow_water_demo/stargazers
[issues-shield]: https://img.shields.io/github/issues/ai2cm/shallow_water_demo.svg?style=for-the-badge
[issues-url]: https://github.com/ai2cm/shallow_water_demo/issues
[license-shield]: https://img.shields.io/github/license/ai2cm/shallow_water_demo.svg?style=for-the-badge
[license-url]: https://github.com/ai2cm/shallow_water_demo/blob/main/LICENSE
