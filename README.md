[![Tests Status](https://github.com/kubosis/NeRa/actions/workflows/test.yml/badge.svg)](https://github.com/kubosis/NeRa/actions/workflows/test.yml)
[![Code Coverage Status](https://codecov.io/github/kubosis/NeRa/branch/main/graph/badge.svg?)](https://codecov.io/gh/kubosis/NeRa) 
![Flake8 Status](https://github.com/kubosis/NeRa/actions/workflows/quality.yml/badge.svg)


![NeRa LOGO](./docs/logo_plus_text.png)

PyTorch based package for incorporating rating systems to graph neural networks.

### Prerequisities

```
Python >= 3.11
```

### Installation

```commandline
$ pip install --upgrade pip
$ pip install torch==2.1.2
$ pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cpu.html
$ pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.2+cpu.html
$ pip install torch-geometric==2.4.0
$ pip install torch-geometric-temporal==0.52.0
$ pip install git+https://github.com/kubosis/NeRa.git
```
