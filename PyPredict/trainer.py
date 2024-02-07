from typing import Union
from datetime import timedelta

import numpy as np
import pandas as pd
from loguru import logger
import torch
from torch_geometric_temporal.signal import temporal_signal_split, DynamicGraphTemporalSignal

from data_management import DataTransformation


class Trainer:
    def __init__(self, dataset: Union[DynamicGraphTemporalSignal, pd.DataFrame],
                 model: torch.nn.Module, lr: float = .001,
                 train_ratio: float = 0.2, **kwargs):
        if isinstance(dataset, DynamicGraphTemporalSignal):
            self.dataset = dataset
            self.loader = None
        else:
            time_delta = kwargs.pop('timedelta', timedelta(days=365))
            self.loader = DataTransformation(df=dataset, snapshot_duration=time_delta)
            self.dataset = self.loader.get_dataset(**kwargs)

        (self.train_dataset,
         self.test_dataset) = temporal_signal_split(dataset, train_ratio=train_ratio)

        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def train(self, epochs: int = 100, batch_size: int = 64, verbose: bool = False) -> np.ndarray:
        training_accuracy = []
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
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
                self.optimizer.step()
                loss += cost.item()

            if verbose:
                logger.info(f'[TRN] Epoch: {epoch}, training loss: {loss:.3f}, '
                            f'training accuracy: {accuracy/count * 100:.2f}% '
                            )
            training_accuracy.append(accuracy/count * 100)
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



