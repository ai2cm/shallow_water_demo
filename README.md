[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache License][license-shield]][license-url]

# GT4Py Shallow Water Demo

This demonstrates simple scaffolding and the core components to run a gt4py model.

## 🐝 Getting started

The demo requires Python 3.8 or 3.9, and `setuptools`. The other dependencies are
specified in [`setup.cfg`](https://github.com/ai2cm/shallow_water_demo/blob/main/setup.cfg).

To get going...

```shell
$ python3.8 -m venv venv
$ source venv/bin/activate
$ gh repo clone ai2cm/shallow_water_demo && cd shallow_water_demp
$ pip install -e ./
```

## ✍️ Development

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
