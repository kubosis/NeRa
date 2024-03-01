from typing import Optional

import torch
import torch.nn as nn
from torch_geometric_temporal.signal import temporal_signal_split, DynamicGraphTemporalSignal
import numpy as np
from loguru import logger

from .models.loss import WeightedMSELoss


class Trainer:
    """
    Basic training class for predefined models
    """

    def __init__(self, dataset: Optional[DynamicGraphTemporalSignal] = None, model: Optional[nn.Module] = None,
                 lr: float = .001, lr_rating: float = 3., loss_fn=WeightedMSELoss, train_ratio: float = 0.8, **kwargs):

        self.train_dataset, self.test_dataset = None, None
        self.train_ratio = train_ratio

        if dataset is not None:
            self.dataset = dataset
        else:
            self._dataset = None

        self.lr = lr
        self.lr_rating = lr_rating
        self.optim = None

        if model is not None:
            self.model = model
        else:
            self._model = None

        self.loss_fn = loss_fn()

    @property
    def model(self) -> nn.Module:
        return self._model

    @model.setter
    def model(self, model: nn.Module):
        if not isinstance(model, nn.Module):
            logger.error("Unsupported model type")
            raise TypeError("Unsupported model type")

        self._model = model
        if hasattr(model, 'is_rating') and model.is_rating:
            # does rating need to compute hyperparameters backward pass?
            if model.hp_grad:
                self.optim = torch.optim.Adam([
                    {'params': model.ratings, 'lr': self.lr_rating},
                    {'params': model.hyperparams, 'lr': self.lr}
                ])
            else:
                self.optim = torch.optim.Adam(
                    model.ratings, self.lr_rating
                )
        else:
            self.optim = torch.optim.Adam(model.parameters(), lr=self.lr)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    @model.deleter
    def model(self):
        del self._model
        del self.optim
        self._model = self.optim = None

    @property
    def dataset(self) -> DynamicGraphTemporalSignal:
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: DynamicGraphTemporalSignal):
        self._dataset = dataset
        trn, tst = temporal_signal_split(dataset, train_ratio=self.train_ratio)
        self.train_dataset, self.test_dataset = trn, tst

    @dataset.deleter
    def dataset(self):
        del self._dataset
        del self.train_dataset
        del self.test_dataset
        self._dataset = self.train_dataset = self.test_dataset = None

    def train(self, epochs: int = 100, batch_size: int = 64, verbose: bool = False, **kwargs) -> np.ndarray:

        assert self.model is not None, "Model has to be set"
        assert self.dataset is not None, "Dataset has to be set"

        if hasattr(self.model, 'is_rating') and self.model.is_rating:
            return self._train_rating(verbose=verbose, **kwargs)

        training_accuracy = []
        self.model.train()
        for epoch in range(epochs):
            self.optim.zero_grad()
            accuracy, loss, count = 0, 0, 0
            for time, snapshot in enumerate(self.train_dataset):
                y_hat = self.model(snapshot.edge_index)
                y = snapshot.edge_attr  # edge weight encodes the match outcome

                target = torch.argmax(y, dim=1)
                prediction = torch.argmax(y_hat, dim=1)
                cost = self.loss_fn(y_hat, target)
                accuracy += int((prediction.eq(target)).sum().item())
                count += len(prediction)

                cost.backward()
                self.optim.step()
                loss += cost.item()

            if verbose:
                logger.info(f'[TRN] Epoch: {epoch}, training loss: {loss:.3f}, '
                            f'training accuracy: {accuracy / count * 100:.2f}% '
                            )
            training_accuracy.append(accuracy / count * 100)
        return np.array(training_accuracy)

    def _loss_fn(self, y, y_hat, weight):
        if isinstance(self.loss_fn, WeightedMSELoss):
            return self.loss_fn(y, y_hat, weight)
        else:
            return self.loss_fn(y, y_hat)

    def _train_rating(self, epochs: int = 1, verbose: bool = False, clip_grad: bool = False, **kwargs) -> np.ndarray:
        model = self.model
        training_accuracy = []
        model.train()

        for epoch in range(epochs):
            accuracy, loss_acc, count = 0, 0, 0
            for time, snapshot in enumerate(self.dataset):
                # pass through network has to be only one by one in order to compute elo correctly
                matches = snapshot.edge_index
                match_points = snapshot.match_points
                outcomes = snapshot.edge_attr
                count += matches.shape[1]

                if not model.is_manual:
                    self.optim.zero_grad()

                for m in range(matches.shape[1]):

                    match = matches[:, m]

                    y_hat = model(match)
                    y = outcomes[m, :]  # edge weight encodes the match outcome
                    if not model.is_manual:
                        y.requires_grad = True

                    target = torch.argmax(y) / 2.
                    target = target.detach()
                    prediction = y_hat

                    accuracy += 1 if abs(target - prediction) <= 0.5 else 0

                    home_pts, away_pts = match_points[m, 0], match_points[m, 1]
                    point_diff = torch.abs(home_pts - away_pts)

                    loss = self._loss_fn(y, y_hat, (point_diff + 1) ** model.gamma)
                    loss_acc += loss.item()

                    if model.is_manual:
                        model.backward([home_pts, away_pts])
                    else:
                        loss.backward()
                        if clip_grad:
                            # Clip gradients to prevent explosion
                            # This should be used when training c, d hyper params as well
                            torch.nn.utils.clip_grad_norm_(model.hyperparams, max_norm=1)
                        self.optim.step()

            if verbose:
                rating = model.ratings[0][:5] if len(model.ratings[0]) >= 5 else model.ratings[0]
                logger.info(f'[TRN] '
                            f' Epoch: {epoch}, training loss: {loss_acc:.3f}, '
                            f'training accuracy: {accuracy / count * 100:.2f}% \n'
                            f'ratings (first 5): {rating}')
            training_accuracy.append(accuracy / count * 100)

        return np.array(training_accuracy)

    def test(self, verbose: bool = False) -> float:
        self.model.eval()
        correct, count = 0, 0
        with torch.no_grad():
            for time, snapshot in enumerate(self.test_dataset):
                y_hat = self.model(snapshot.edge_index)
                y = snapshot.edge_attr
                target = torch.argmax(y, dim=1)
                prediction = torch.argmax(y_hat, dim=1)
                correct += int((prediction.eq(target)).sum().item())
                count += len(prediction)
        test_accuracy = correct / count
        if verbose:
            logger.info(f'[TST] Testing accuracy: {100. * test_accuracy:.2f}%')

        return test_accuracy
