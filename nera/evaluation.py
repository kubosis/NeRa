from datetime import timedelta
import os

import optuna
import mlflow
import pandas as pd
from torch.nn import CrossEntropyLoss
from loguru import logger

from nera.dummy import get_dummy_df
from nera.trainer import Trainer
from nera.models.gnn import EvalRatingRGNN
from nera.data import DataTransformation, DataAcquisition, FROM_CSV

dense_dims_choice = [
    (4, 4, 4, 4, 4), (8, 8, 8, 8, 8), (16, 16, 16, 16, 16), (32, 32, 32, 32, 32), (64, 64, 64, 64, 64),
    (4, 8, 16, 32, 64), (4, 8, 4, 16, 4), (32, 64, 32, 128, 4), (16, 32, 4, 16, 8), (16, 4, 16, 4, 16),
    (64, 128, 64, 128, 8), (128, 128, 128, 128, 128), None
]

conv_dims_choice = [
    (1, 1, 1), (4, 4, 4), (8, 8, 8), (16, 16, 16), (64, 64, 64), (128, 128, 128), (128, 64, 32), (32, 32, 32),
    (32, 16, 8), (16, 8, 4), (128, 32, 8), (64, 32, 4), (4, 64, 4), (8, 128, 8),
    (8, 128, 64), (64, 32, 64)
]


class Evaluation:
    @staticmethod
    def objective(trial: optuna.Trial, dataset, team_count):
        rgnn_choice = trial.suggest_categorical("rgnn_choice", ["GCONV_GRU", "GCONV_ELMAN", "PEREVERZEVA_RGNN"])
        rating = trial.suggest_categorical("rating", ["elo", "berrar"])
        K = 2
        if rgnn_choice == "GCONV_GRU":
            gconv_choice = "ChebConv"
        elif rgnn_choice == "GCONV_ELMAN":
            gconv_choice = trial.suggest_categorical("gconv_choice", ["GraphConv", "GCNConv", "ChebConv"])
        else:
            gconv_choice = trial.suggest_categorical("gconv_choice", ["GraphConv", "GCNConv", "ChebConv"])

        if gconv_choice == "ChebConv":
            K = trial.suggest_int("K", 2, 5)

        lr_hyper = trial.suggest_float("lr", 1e-5, 1e-1)
        lr_rating = trial.suggest_float("lr", 1e-1, 10.)

        epochs = trial.suggest_int("epochs", 1, 100)
        val_ratio = trial.suggest_float("val_ratio", 0.1, 0.3)

        embed_dim = trial.suggest_int("embed_dim", 1, 512)

        correction = trial.suggest_categorical("correction", [True, False])
        correction = bool(correction)

        discount = trial.suggest_float("discount", 0.1, 0.99)

        activation = trial.suggest_categorical("activation", ["relu", "lrelu", "tanh"])

        dense_dims = trial.suggest_categorical("dense_dims", dense_dims_choice)
        conv_dims = trial.suggest_categorical("conv_dims", conv_dims_choice)

        dropout_rate = trial.suggest_float("dropout_rate", 0., 0.2)

        # Log hyperparameters to MLflow
        mlflow.log_param("lr_hyper", lr_hyper)
        mlflow.log_param("lr_rating", lr_rating)
        mlflow.log_param("val_ratio", val_ratio)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("embed_dim", embed_dim)
        mlflow.log_param("correction", correction)
        mlflow.log_param("discount", discount)
        mlflow.log_param("activation", activation)
        mlflow.log_param("K", K)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("rating", rating)
        mlflow.log_param("gconv_choice", gconv_choice)
        mlflow.log_param("rgnn_choice", rgnn_choice)
        mlflow.log_param("dense_dims", dense_dims)
        mlflow.log_param("conv_dims", conv_dims)

        score = Evaluation.train(
            dataset,
            team_count,
            lr_hyper,
            lr_rating,
            epochs,
            val_ratio,
            embed_dim,
            discount,
            correction,
            activation,
            K,
            rgnn_choice,
            gconv_choice,
            rating,
            dense_dims,
            conv_dims,
            dropout_rate,
        )

        # Log the score to MLflow
        mlflow.log_metric("score", score)

        return score

    @staticmethod
    def objective_test(trial, dataset, team_count, run_name):
        lr_hyper = trial.suggest_float("lr_hyper", 1e-5, 1e-1)
        lr_rating = trial.suggest_float("lr_rating", 1e-1, 10.)

        logger.info(f"lr_hyper {lr_hyper}, lr_rating {lr_rating}")

        score = Evaluation.train(
            dataset,
            team_count,
            lr_hyper,
            lr_rating,
            10,
            0.5,
            5,
            0.8,
            False,
            "lrelu",
            2,
            "GCONV_GRU",
            "ChebConv",
            "elo",
            None,
            (4, 4),
            0,
        )
        with mlflow.start_run(nested=True, run_name=run_name + f"_sub_{trial.number}"):
            mlflow.log_param(f"lr_hyper", lr_hyper)
            mlflow.log_param(f"lr_rating", lr_rating)
            mlflow.log_metric(f"score", score)
        logger.success(f"Done trial with score {score}")

        return score



    @staticmethod
    def train(
            dataset,
            team_count: int,
            lr_hyperparams: float,
            lr_rating: float,
            epochs: int,
            val_ratio: float,
            embed_dim: int,
            discount: float,
            correction: bool,
            activation: str,
            K: int,
            rgnn_conv: str,
            graph_conv: str,
            rating: str,
            dense_dims: tuple,
            conv_dims: tuple,
            dropout_rate: float,
            **rating_kwargs
    ):
        model = EvalRatingRGNN(
            team_count,
            embed_dim,
            target_dim=2,
            discount=discount,
            correction=correction,
            activation=activation,
            K=K,
            rgnn_conv=rgnn_conv,
            graph_conv=graph_conv,
            rating=rating,
            normalization=None,
            aggr="add",
            dense_dims=dense_dims,
            conv_dims=conv_dims,
            dropout_rate=dropout_rate,
            **rating_kwargs
        )
        trainer = Trainer(
            dataset,
            model,
            lr_hyperparams,
            lr_rating,
            loss_fn=CrossEntropyLoss(),
            train_ratio=1.
        )
        trainer.train(epochs, val_ratio=val_ratio)
        metric = trainer.get_eval_metric("val_loss")
        return metric

    @staticmethod
    def evaluate(raw_data):
        transform = DataTransformation(raw_data, snapshot_duration=timedelta(days=365))
        dataset = transform.get_dataset(
            node_f_extract=False, edge_f_one_hot=True, drop_draws=True
        )

        # Set up an MLflow run
        mlflow.set_experiment("test_potato")
        run_name = "ratingGNN"
        with mlflow.start_run(run_name=run_name):
            # Set up an Optuna study
            study = optuna.create_study(direction="minimize")

            # Run the optimization process
            study.optimize(
                lambda trial: Evaluation.objective_test(trial, dataset, transform.num_teams, run_name), n_trials=10)

            # Access the best hyperparameters found during the study
            best_params = study.best_params
            print("Best hyperparameters:", best_params)

            # Log the best hyperparameters to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metric(f"Best score", study.best_value)
            mlflow.log_metric("Best trial", study.best_trial.number)


if __name__ == "__main__":
    os.environ['MLFLOW_TRACKING_URI'] = 'http://host.docker.internal:2222'
    #da = DataAcquisition()
    #df = da.get_data(FROM_CSV, fname="../resources/other_leagues.csv")
    #df['DT'] = pd.to_datetime(df['DT'], format="%Y-%m-%d %H:%M:%S")
    #df = df.reset_index()
    #df = df.sort_values(by='DT', ascending=False)

    df = get_dummy_df(0, conf="hah")
    transform = DataTransformation(df, timedelta(days=365))

    Evaluation.evaluate(df)
