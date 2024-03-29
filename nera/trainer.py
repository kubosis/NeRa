from typing import Optional, Sequence

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

        self._train_dataset, self._test_dataset = None, None
        self._train_ratio = train_ratio

        if dataset is not None:
            self.dataset = dataset
        else:
            self._dataset = None

        self._lr = lr
        self._lr_rating = lr_rating
        self.optim = None
        self.model_is_rating = False

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
            self.model_is_rating = True

            # does rating need to compute hyperparameters backward pass?
            if model.hp_grad:
                self.optim = torch.optim.Adam([
                    {'params': model.ratings, 'lr': self._lr_rating},
                    {'params': model.hyperparams, 'lr': self._lr}
                ])
            else:
                self.optim = torch.optim.Adam(model.ratings, self._lr_rating)
        else:
            self.model_is_rating = False
            self.optim = torch.optim.Adam(model.parameters(), lr=self._lr)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    @model.deleter
    def model(self):
        del self._model
        del self.optim
        self._model = self.optim = None
        self.model_is_rating = False

    @property
    def dataset(self) -> DynamicGraphTemporalSignal:
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: DynamicGraphTemporalSignal):
        self._dataset = dataset
        trn, tst = temporal_signal_split(dataset, train_ratio=self.train_ratio)
        self._train_dataset, self._test_dataset = trn, tst

    @dataset.deleter
    def dataset(self):
        del self._dataset
        del self._train_dataset
        del self._test_dataset
        self._dataset = self._train_dataset = self._test_dataset = None

    @property
    def train_ratio(self):
        return self._train_ratio

    @train_ratio.setter
    def train_ratio(self, train_ratio: float):

        assert isinstance(train_ratio, float) or train_ratio == 1.
        assert 0 < train_ratio <= 1, "Training ratio has to be in (0, 1]"

        self._train_ratio = train_ratio
        if self.dataset is not None:
            trn, tst = temporal_signal_split(self.dataset, train_ratio=self.train_ratio)
            self._train_dataset, self._test_dataset = trn, tst

    def set_lr(self, lr: float, lr_type: str = "hyperparam"):
        assert lr_type.lower() in ["rating", "hyperparam"]

        if lr_type.lower() == "rating":
            self._lr_rating = lr
        else:
            self._lr = lr

        # reset model so that lr is written into optimizer
        self.model = self.model

    def train(self, epochs: int = 100, verbose: bool = False,
              val_ratio: float = 0.0, **kwargs) -> Sequence[np.ndarray]:

        assert self.model is not None, "Model has to be set"
        assert self.dataset is not None, "Dataset has to be set"
        assert 0 <= val_ratio < 1, "Validation ratio has to be in [0, 1)"

        training_accuracy = []
        validation_accuracy = []

        for epoch in range(epochs):
            self.optim.zero_grad()
            trn_acc, trn_loss, trn_count = 0, 0, 0
            val_acc, val_loss, val_count = 0, 0, 0
            for time, snapshot in enumerate(self._train_dataset):
                matches = snapshot.edge_index
                outcomes = snapshot.edge_attr  # edge weight encodes the match outcome
                match_points = snapshot.match_points

                # split train / val data
                matches_count = matches.shape[1]
                trn_size = np.ceil(matches_count * (1 - val_ratio)).astype(int)

                matches_train = matches[:, :trn_size]
                y_train = outcomes[:trn_size, :]
                match_pts_trn = match_points[:trn_size, :]
                trn_count += matches_train.shape[1]

                matches_val = matches[:, trn_size:]
                y_val = outcomes[trn_size:, :]
                match_pts_val = match_points[trn_size:, :]
                val_count += matches_val.shape[1]

                # training
                if self.model_is_rating:
                    trn_acc_i, trn_loss_i = self._train_rating(matches_train, y_train, match_pts_trn, **kwargs)
                else:
                    trn_acc_i, trn_loss_i = self._train_gnn(matches_train, y_train)
                trn_acc += trn_acc_i
                trn_loss += trn_loss_i

                # validation
                with torch.no_grad():
                    if self.model_is_rating:
                        val_acc_i, val_loss_i = self._train_rating(
                            matches_val, y_val, match_pts_val, validation=True, **kwargs)
                    else:
                        val_acc_i, val_loss_i = self._train_rating(matches_val, y_val, match_pts_val, validation=True)
                val_acc += val_acc_i
                val_loss += val_loss_i

            if verbose:
                logger.info(f'[TRN] Epoch: {epoch}, training loss: {trn_loss:.3f}, '
                            f'training accuracy: {trn_acc / trn_count * 100:.2f}%')
                if val_count != 0:
                    logger.info(f'[VAL] Epoch: {epoch}, validation loss: {val_loss:.3f}, '
                                f'validation accuracy: {val_acc / val_count * 100:.2f}%')
            training_accuracy.append(trn_acc / trn_count)
            if val_count != 0:
                validation_accuracy.append(val_acc / val_count)

        return [np.array(training_accuracy), np.array(validation_accuracy)]

    def _loss_fn(self, y, y_hat, weight):
        if isinstance(self.loss_fn, WeightedMSELoss):
            return self.loss_fn(y, y_hat, weight)
        else:
            return self.loss_fn(y, y_hat)

    def _train_gnn(self, matches, y, validation: bool = False) -> tuple[int, float]:
        if validation:
            self.model.eval()
        else:
            self.model.train()

        y_hat = self.model(matches)
        target = torch.argmax(y, dim=1)
        prediction = torch.argmax(y_hat, dim=1)

        cost = self.loss_fn(y_hat, target)
        trn_acc = int((prediction.eq(target)).sum().item())
        trn_loss = cost.item()

        if not validation:
            cost.backward()
            self.optim.step()

        return trn_acc, trn_loss

    def _train_rating(self, matches, outcomes, match_points,
                      validation: bool = False, clip_grad: bool = False, **kwargs) -> tuple[int, float]:

        if validation:
            self.model.eval()
        else:
            self.model.train()

        if not self.model.is_manual:
            self.optim.zero_grad()

        accuracy, loss_acc = 0, 0
        mi = 0
        for m in range(matches.shape[1]):
            match = matches[:, m]

            y_hat = torch.mean(self.model(match))
            y = outcomes[m, :]  # edge weight encodes the match outcome
            if not self.model.is_manual:
                y.requires_grad = True

            target = torch.argmax(y) / 2.
            target = target.detach()
            prediction = y_hat

            accuracy += 1 if abs(target - prediction) < 0.5 else 0

            home_pts, away_pts = match_points[m, 0], match_points[m, 1]
            point_diff = torch.abs(home_pts - away_pts)

            gamma = self.model.gamma if hasattr(self.model, 'gamma') else 1
            loss = self._loss_fn(y, y_hat, (point_diff + 1) ** gamma)
            loss_acc += loss.item()

            if validation:
                continue

            if self.model.is_manual:
                self.model.backward([home_pts, away_pts])
            else:
                loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.hyperparams, max_norm=1)
                self.optim.step()

        return accuracy, loss_acc

    def test(self, verbose: bool = False) -> float:
        self.model.eval()
        correct, count = 0, 0

        test_fun = self._test_rating if self.model_is_rating else self._test_gnn

        with torch.no_grad():
            for time, snapshot in enumerate(self._test_dataset):
                correct, count = test_fun(snapshot)
        if count == 0:
            logger.warning("Trying to test when testing dataset is empty")
            test_accuracy = 0
        else:
            test_accuracy = correct / count
            logger.info(f'[TST] Testing accuracy: {100. * test_accuracy:.2f}%')

        return test_accuracy

    def _test_gnn(self, snapshot):
        y_hat = self.model(snapshot.edge_index)
        y = snapshot.edge_attr
        target = torch.argmax(y, dim=1)
        prediction = torch.argmax(y_hat, dim=1)
        correct = int((prediction.eq(target)).sum().item())
        count = len(prediction)
        return correct, count

    def _test_rating(self, snapshot):
        y_hat = torch.mean(self.model(snapshot.edge_index), dim=1)
        y = snapshot.edge_attr
        target = torch.argmax(y, dim=1) / 2.
        correct = torch.sum(torch.abs(target - y_hat) < 0.5).item()
        count = len(y_hat)
        return correct, count
