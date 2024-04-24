import optuna
import mlflow
from torch.nn import CrossEntropyLoss

from nera.trainer import Trainer
from nera.models.gnn import RatingGNN, RGNN

_model = {"rating": RatingGNN, "RGNN": RGNN}


class Evaluation:
    @staticmethod
    def objective(trial: optuna.trial.Trial):
        lr_hyperparams = trial.suggest_float("lr_hyperparams", 1e-5, 1e-1, log=True)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

    @staticmethod
    def train(
        model: str,
        dataset,
        lr_hyperparams,
        lr_rating,
        optim,
        epochs,
        val_ratio,
        **kwargs
    ):
        assert model in _model
        model = _model[model](**kwargs)
        trainer = Trainer(
            dataset,
            model,
            lr_hyperparams,
            lr_rating,
            loss_fn=CrossEntropyLoss(),
            optim=optim,
        )
        trainer.train(epochs, val_ratio=val_ratio)
        metric = trainer.get_eval_metric("val_accuracy")
        return metric

    def evaluate():
        _model_kwargs = {}
