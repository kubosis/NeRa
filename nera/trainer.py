from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from torch_geometric_temporal.signal import (
    temporal_signal_split,
    DynamicGraphTemporalSignal,
)
import numpy as np
from loguru import logger

from .models.loss import WeightedMSELoss


class Trainer:
    """
    Basic training class for predefined models
    """

    def __init__(
            self,
            dataset: Optional[DynamicGraphTemporalSignal] = None,
            model: Optional[nn.Module] = None,
            lr: float = 0.001,
            lr_rating: float = 3.0,
            loss_fn=WeightedMSELoss(),
            train_ratio: float = 0.8,
            optim=torch.optim.Adam,
            **kwargs,
    ):

        self._train_dataset, self._test_dataset = None, None
        self._train_ratio = train_ratio

        if dataset is not None:
            self.dataset = dataset
        else:
            self._dataset = None

        self._lr = lr
        self._lr_rating = lr_rating
        self.optim = optim
        self.model_is_rating = False
        self.device = None

        if model is not None:
            self.model = model
        else:
            self._model = None

        self.loss_fn = loss_fn

        self.val_accuracy = 0
        self.val_loss = float("inf")
        self.train_accuracy = 0
        self.train_loss = float("inf")
        self.test_accuracy = 0
        self.test_loss = float("inf")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    @property
    def model(self) -> nn.Module:
        return self._model

    @model.setter
    def model(self, model: nn.Module):
        if not isinstance(model, nn.Module):
            logger.error("Unsupported model type")
            raise TypeError("Unsupported model type")

        self._model = model
        if hasattr(model, "is_rating") and model.is_rating:
            self.model_is_rating = True

            # does rating need to compute hyperparameters backward pass?
            if model.hp_grad:
                self.optim = torch.optim.Adam(
                    [
                        {"params": model.ratings, "lr": self._lr_rating},
                        {"params": model.hyperparams, "lr": self._lr},
                    ]
                )
            else:
                self.optim = torch.optim.SGD(model.ratings, self._lr_rating)
        else:
            self.model_is_rating = False

            self.optim = torch.optim.SGD(
                [
                    {"params": model.embedding.parameters(), "lr": self._lr_rating},
                ],
                lr=self._lr,
            )
            for m in model.gconv_layers:
                self.optim.add_param_group({"params": m.parameters(), "lr": self._lr})
            if model.linear_layers is not None:
                for m in model.linear_layers:
                    self.optim.add_param_group({"params": m.parameters(), "lr": self._lr})
            if model.rating is not None:
                self.optim.add_param_group({"params": model.rating.parameters(), "lr": self._lr})

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

        assert isinstance(train_ratio, float) or train_ratio == 1.0
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

    def train(
            self, epochs: int = 100, verbose: bool = False, val_ratio: float = 0.0,
            callback=None, callback_kwarg_dict=None, **kwargs
    ) -> Sequence[np.ndarray]:

        assert self.model is not None, "Model has to be set"
        assert self.dataset is not None, "Dataset has to be set"
        assert 0 <= val_ratio < 1, "Validation ratio has to be in [0, 1)"

        if callback is not None:
            logger.info("Starting training with registered callback")

        training_accuracy = []
        validation_accuracy = []

        for epoch in range(epochs):

            trn_acc, trn_loss, trn_count = 0, 0, 0
            val_acc, val_loss, val_count = 0, 0, 0
            if not self.model_is_rating:
                self.model.reset_index()
            for time, snapshot in enumerate(self._train_dataset):
                matches = snapshot.edge_index.to(self.device)
                outcomes = snapshot.edge_attr.to(self.device)  # edge weight encodes the match outcome
                match_points = snapshot.match_points.to(self.device)

                # split train / val data
                matches_count = matches.shape[1]
                trn_size = np.ceil(matches_count * (1 - val_ratio)).astype(int)

                matches_train = matches[:, :trn_size]
                y_train = outcomes[:trn_size, :]
                match_pts_trn = match_points[:trn_size, :]
                trn_count_i = matches_train.shape[1]
                trn_count += trn_count_i

                matches_val = matches[:, trn_size:]
                y_val = outcomes[trn_size:, :]
                match_pts_val = match_points[trn_size:, :]
                val_count_i = matches_val.shape[1]
                val_count += val_count_i

                # training
                if self.model_is_rating:
                    trn_acc_i, trn_loss_i = self._train_rating(
                        matches_train, y_train, match_pts_trn, **kwargs
                    )
                else:
                    trn_acc_i, trn_loss_i = self._train_gnn(
                        matches_train, y_train, match_pts_trn, verbose, **kwargs
                    )
                trn_acc += trn_acc_i
                trn_loss += trn_loss_i

                # validation
                with torch.no_grad():
                    if self.model_is_rating:
                        val_acc_i, val_loss_i = self._train_rating(
                            matches_val, y_val, match_pts_val, validation=True, **kwargs
                        )
                    else:
                        val_acc_i, val_loss_i = self._train_gnn(
                            matches_val, y_val, match_pts_val, verbose, validation=True, **kwargs
                        )
                val_acc += val_acc_i
                val_loss += val_loss_i

                if val_count_i != 0:
                    validation_accuracy.append(val_acc_i / val_count_i)

                # train the model on the data that validated it (only if epochs == 1)
                if epochs == 1:
                    if self.model_is_rating:
                        trn_acc_ii, trn_loss_ii = self._train_rating(
                            matches_val, y_val, match_pts_val, **kwargs
                        )
                    else:
                        trn_acc_ii, trn_loss_ii = self._train_gnn(
                            matches_val, y_val, match_pts_val, verbose, **kwargs
                        )

                    trn_acc += trn_acc_ii
                    trn_loss += trn_loss_ii
                    trn_acc_i += trn_acc_ii
                    trn_loss_i += trn_loss_ii

                    trn_count_i += val_count_i

                training_accuracy.append(trn_acc_i / trn_count_i)

                self.model.H = None

            if verbose:
                logger.info(
                    f"[TRN] Epoch: {epoch + 1}, training loss: {trn_loss:.3f}, "
                    f"training accuracy: {trn_acc / trn_count * 100:.2f}%"
                )
                if val_count != 0:
                    logger.info(
                        f"[VAL] Epoch: {epoch + 1}, validation loss: {val_loss:.3f}, "
                        f"validation accuracy: {val_acc / val_count * 100:.2f}%"
                    )

            self.train_accuracy = trn_acc / trn_count
            self.train_loss = trn_loss
            if val_count != 0:
                # store for later metric inspection
                self.val_accuracy = val_acc / val_count
                self.val_loss = val_loss

            if callback is not None:
                callback_kwarg_dict["epochs"] = epoch
                callback_kwarg_dict["score"] = self.val_loss
                callback_kwarg_dict["acc"] = self.val_accuracy
                callback(**callback_kwarg_dict)

        return [np.array(training_accuracy), np.array(validation_accuracy)]

    def _loss_fn(self, y, y_hat, weight=None):
        if isinstance(self.loss_fn, WeightedMSELoss):
            assert weight is not None
            return self.loss_fn(y_hat, y, weight)
        else:
            return self.loss_fn(y_hat, y)

    def _create_edge_index_and_weight(
            self, match, y, validation: bool = False, bidir: bool = True
    ) -> tuple[Tensor, Optional[Tensor]]:
        assert len(y) in [
            2,
            3,
        ], "Invalid outcome encoding - only one hot match outcome supported"

        outcome = torch.argmax(y).item()
        match = match.unsqueeze(1)
        weight = None
        if len(y) == 2:
            # draws not used -> bidirectional edges possible
            if outcome == 0:
                # away win
                if bidir:
                    index = torch.cat((match, torch.flip(match, dims=[0])), dim=1)
                    weight = torch.tensor([+1, -1], dtype=torch.float)
                else:
                    index = match
                    weight = torch.tensor([1.0], dtype=torch.float)
            else:
                # home win
                if bidir:
                    index = torch.cat((match, torch.flip(match, dims=[0])), dim=1)
                    weight = torch.tensor([-1, +1], dtype=torch.float)
                else:
                    index = torch.flip(match, dims=[0])
                    weight = torch.tensor([1.0], dtype=torch.float)
        else:
            # draws used
            if outcome == 0:
                # away win
                index = torch.cat((match, torch.flip(match, dims=[0])), dim=1)
                weight = torch.tensor([+1, -1], dtype=torch.float)
            elif outcome == 1:
                # draw
                index = torch.cat((match, torch.flip(match, dims=[0])), dim=1)
                weight = torch.tensor([0.5, 0.5], dtype=torch.float)
            else:
                # home win
                index = torch.cat((match, torch.flip(match, dims=[0])), dim=1)
                weight = torch.tensor([-1, +1], dtype=torch.float)

        if validation:
            weight = None
        else:
            weight *= 0.1
            weight = weight.to(self.device)

        index = index.to(self.device)
        return index, weight

    def _step(self):
        for p in self.model.lin.parameters():
            p.data.add_(p.grad.data, alpha=-self._lr)
        for p in self.model.rnn_gconv.parameters():
            p.data.add_(p.grad.data, alpha=-self._lr)

        self.model.embedding.data.add_(
            self.model.embedding.grad.data, alpha=-self._lr_rating
        )

    def _train_gnn(
            self,
            matches,
            outcomes,
            match_points,
            verbose,
            validation: bool = False,
            clip_grad: bool = False,
            bidir: bool = False,
            gamma: float = 1.45
    ) -> tuple[int, float]:
        if validation:
            self.model.eval()
            self.model.store_index(False)
        else:
            self.model.train()
            self.model.store_index(True)

        ite = 0
        accuracy, loss_acc = 0, 0
        for m in range(matches.shape[1]):
            self.optim.zero_grad()

            home, away = match = matches[:, m]
            home_pts, away_pts = match_points[m, :]

            if self.model.rating_str == "berrar":
                y = torch.cat((away_pts.unsqueeze(0), home_pts.unsqueeze(0)), dim=0).to(self.device, torch.float)
                # rescale between 0 and 1
                max_abs_value = torch.max(torch.abs(y))
                y = y / max_abs_value
            elif self.model.rating_str == "pi":
                g_d_home = home_pts - away_pts
                g_d_away = away_pts - home_pts
                y = torch.cat((g_d_away.unsqueeze(0), g_d_home.unsqueeze(0)), dim=0).to(self.device, torch.float)
                # rescale between 0 and 1
                max_abs_value = torch.max(torch.abs(y))
                y = y / max_abs_value
            else:
                # elo, None
                ...
            y = outcomes[m, :].to(torch.float)

            edge_index, edge_weight = self._create_edge_index_and_weight(match, y, validation, bidir=bidir)

            point_diff = torch.abs(home_pts - away_pts)

            y_hat = self.model(edge_index, home, away, edge_weight, home_pts, away_pts)

            target = torch.argmax(outcomes[m, :])
            prediction = torch.argmax(y_hat)

            accuracy += 1 if abs(target - prediction) < 0.1 else 0

            loss = self._loss_fn(y, y_hat, (point_diff + 1) ** gamma)
            loss.retains_grad_ = True
            loss_acc += loss.item()

            if validation:
                continue

            loss.backward()
            if verbose and False:
                print("Iteration ------------------------------------ ", ite)
                # print(f"Loss gradient: {loss.grad:.4f}")
                for name, param in self.model.named_parameters():
                    print(f"Gradient of {name}: {param.grad}")
                print("\n")
            ite += 1

            if clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.hyperparams, max_norm=1)
            self.optim.step()
            # self._step()

        return accuracy, loss_acc

    def _train_rating(
            self,
            matches,
            outcomes,
            match_points,
            validation: bool = False,
            clip_grad: bool = False,
            **kwargs,
    ) -> tuple[int, float]:
        if self.model.type == "elo":
            return self._train_elo(
                matches, outcomes, match_points, validation, clip_grad, **kwargs
            )
        elif self.model.type == "berrar":
            return self._train_berrar(
                matches, outcomes, match_points, validation, clip_grad, **kwargs
            )
        elif self.model.type == "pi":
            ...
        else:
            raise RuntimeError("Unknown rating model type")

    def _train_elo(
            self, matches, outcomes, match_points, validation, clip_grad, **kwargs
    ) -> tuple[int, float]:
        if validation:
            self.model.eval()
        else:
            self.model.train()

        if not self.model.is_manual:
            self.optim.zero_grad()

        accuracy, loss_acc = 0, 0
        for m in range(matches.shape[1]):
            match = matches[:, m]

            y_hat = self.model(match)
            y = outcomes[m, :]  # edge weight encodes the match outcome
            if not self.model.is_manual:
                y.requires_grad = True

            target = torch.argmax(y)
            target = target.detach()
            prediction = y_hat

            accuracy += 1 if abs(target - prediction) < 0.5 else 0

            home_pts, away_pts = match_points[m, 0], match_points[m, 1]
            point_diff = torch.abs(home_pts - away_pts)

            gamma = self.model.gamma if hasattr(self.model, "gamma") else 1
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

    def _train_berrar(
            self,
            matches,
            outcomes,
            match_points,
            validation: bool = False,
            clip_grad: bool = False,
            **kwargs,
    ) -> tuple[int, float]:
        if validation:
            self.model.eval()
        else:
            self.model.train()

        if not self.model.is_manual:
            self.optim.zero_grad()

        accuracy, loss_acc = 0, 0
        for m in range(matches.shape[1]):
            match = matches[:, m]

            y_hat = self.model(match)
            outcome = outcomes[m, :]  # edge weight encodes the match outcome

            target = torch.argmax(outcome) / 2.0
            target = target.detach()

            prediction = 1 if y_hat[0] > y_hat[1] else 0 if y_hat[0] < y_hat[1] else 0.5

            accuracy += 1 if abs(target - prediction) < 0.5 else 0

            y = match_points[m, :].type(torch.float64).view(-1, 1)
            if not self.model.is_manual:
                y.requires_grad = True

            if validation:
                continue

            if self.model.is_manual:
                home_pts, away_pts = match_points[m, 0], match_points[m, 1]
                self.model.backward([home_pts, away_pts])
            else:
                loss = self._loss_fn(y, y_hat)
                loss_acc += loss.item()
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
            logger.info(f"[TST] Testing accuracy: {100. * test_accuracy:.2f}%")

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
        target = torch.argmax(y, dim=1) / 2.0
        correct = torch.sum(torch.abs(target - y_hat) < 0.5).item()
        count = len(y_hat)
        return correct, count

    def get_eval_metric(self, metric: str = "val_acc"):
        assert metric in [
            "val_accuracy",
            "test_accuracy",
            "val_loss",
            "test_loss",
            "train_loss",
            "train_accuracy",
        ]
        return getattr(self, metric)
