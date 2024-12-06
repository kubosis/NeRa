![Flake8 Status](https://github.com/kubosis/torch-rating/actions/workflows/quality.yml/badge.svg)
[![PyPI version](https://img.shields.io/pypi/v/torch-rating.svg)](https://pypi.org/project/torch-rating/)


![NeRa LOGO](https://github.com/kubosis/torch-rating/blob/main/docs/logo3.png?raw=true)

PyTorch based package for incorporating rating systems to neural networks. This package provides model rating layers. The resulting RatingRGNN can be found [here](https://github.com/kubosis/rating_gnn)

### Prerequisities

```
Python >= 3.10
```

### Installation

```commandline
pip install --upgrade pip
pip install torch-rating
```

### Nera - Neural rating

This package implements seamless integration of statistical rating systems into graph neural network in the PyTorch environment.
This project was developed as my Bachelor's thesis.

### Implemented rating layers and recurrent graph neural network architectures

- Elo rating
- Berrar rating
- Pi rating

![RatingRGNN architecture](https://raw.githubusercontent.com/kubosis/torch-rating/2d80c8f9b6f3faaf0a5a8e1d9cbecc8c7a2f44f3/docs/img/ratingRGNN.svg)


### Showcases of predictive validation accuracy on collected datasets:

Note: the RatingRGNN was fine-tuned only on the NBL dataset and then applied across the other.

![RatingRGNN architecture](https://github.com/kubosis/torch-rating/blob/main/docs/img/validation.png?raw=true)

Note: the accuracy is across time snapshots. These snapshots represent seasons. They do not represents epochs of iterating the whole dataset. The training was done only for one epoch.

![RatingRGNN architecture](https://github.com/kubosis/torch-rating/blob/main/docs/img/train_val_acc.png?raw=true)
