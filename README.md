![Flake8 Status](https://github.com/kubosis/torch-rating/actions/workflows/quality.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/torch-rating.svg)](https://badge.fury.io/py/torch-rating)


![NeRa LOGO](https://raw.githubusercontent.com/kubosis/torch-rating/blob/main/docs/logo2.webp)

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

![RatingRGNN architecture](./docs/img/ratingRGNN.svg)


### Showcases of predictive validation accuracy on collected datasets:

Note: the RatingRGNN was fine-tuned only on the NBL dataset and then applied across the other.

![RatingRGNN architecture](./docs/img/validation.png)

Note: the accuracy is across time snapshots. These snapshots represent seasons. They do not represents epochs of iterating the whole dataset. The training was done only for one epoch.

![RatingRGNN architecture](./docs/img/train_val_acc.png)
