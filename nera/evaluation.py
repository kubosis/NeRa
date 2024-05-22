from datetime import timedelta
import os
import sys
from typing import Optional

import optuna
import mlflow
import pandas as pd
from torch.nn import CrossEntropyLoss, MSELoss
from loguru import logger
import torch
import numpy as np

from dummy import get_dummy_df
from models.ratings import EloManual
from trainer import Trainer
from models.gnn import RatingRGNN
from data import DataTransformation, DataAcquisition, FROM_CSV


def recursive_to(model, device):
    model.to(device)  # Move the model itself to the device
    for elem in model.gconv_layers:
        elem.to(device)
    for elem in model.linear_layers:
        elem.to(device)


class Evaluation:
    best_acc = 0

    @staticmethod
    def objective(trial: optuna.Trial, dataset, team_count, run_name):
        rgnn_choice = trial.suggest_categorical("rgnn_choice", ["GCONV_GRU", "GCONV_ELMAN"])
        rating = "berrar"
        K = 10
        normalization = None
        if rgnn_choice == "GCONV_GRU":
            gconv_choice = "ChebConv"
        elif rgnn_choice == "GCONV_ELMAN":
            gconv_choice = trial.suggest_categorical("gconv_choice", ["GraphConv", "GCNConv", "ChebConv"])
        else:
            gconv_choice = trial.suggest_categorical("gconv_choice", ["GraphConv", "GCNConv", "ChebConv"])

        if gconv_choice == "ChebConv":
            K = trial.suggest_int("K", 2, 10)
            normalization = trial.suggest_categorical("normalization", [None, "rw"])

        lr_hyper = trial.suggest_float("lr_hyper", 0.0005, 0.02)  # 0.010356261748813426
        lr_rating = trial.suggest_float("lr_rating", 0.5, 8)  # 2.56094697452784

        epochs = 1
        val_ratio = 0.1

        embed_dim = trial.suggest_categorical("embed_dim", [8, 16, 32, 64])
        embed_dim = int(embed_dim)

        correction = trial.suggest_categorical("correction", [True, False])
        correction = bool(correction)

        discount = trial.suggest_float("discount", 0.75, 0.99)

        activation = "lrelu"

        dense_layers = trial.suggest_int("dense_layers", 0, 7)
        conv_layers = trial.suggest_int("conv_layers", 4, 12)

        dropout_rate = 0.2

        score, acc, run_train_acc, run_val_acc = Evaluation.train(
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
            dense_layers,
            conv_layers,
            dropout_rate,
            norm=normalization,
            loss_fn=CrossEntropyLoss(),
        )

        os.makedirs("mlruns/artifacts", exist_ok=True)
        os.makedirs(f"mlruns/artifacts/{run_name}", exist_ok=True)

        train_path = f"mlruns/artifacts/{run_name}/run_train_acc_{run_name}_{trial.number}.npy"
        val_path = f"mlruns/artifacts/{run_name}/run_val_acc_{run_name}_{trial.number}.npy"
        np.save(train_path, run_train_acc)
        np.save(val_path, run_val_acc)

        with mlflow.start_run(nested=True, run_name=run_name + f"_sub_{trial.number}"):
            # Log hyperparameters to MLflow
            mlflow.log_param("dropout_rate", dropout_rate)
            mlflow.log_param("embed_dim", embed_dim)
            mlflow.log_param("K", K)
            mlflow.log_param("rating", rating)
            mlflow.log_param("gconv_choice", gconv_choice)
            mlflow.log_param("rgnn_choice", rgnn_choice)
            mlflow.log_param("dense_layers", dense_layers)
            mlflow.log_param("conv_layers", conv_layers)
            mlflow.log_param("normalization", normalization)
            mlflow.log_param("lr_hyper", lr_hyper)
            mlflow.log_param("lr_rating", lr_rating)
            mlflow.log_param("correction", correction)
            # Log the score to MLflow
            mlflow.log_metric("score", score)
            mlflow.log_metric("acc", acc)

            mlflow.log_artifact(train_path, artifact_path="run_train_acc")
            mlflow.log_artifact(val_path, artifact_path="run_val_acc")

        return score  # acc maximize / score minimize

    @staticmethod
    def objective_test(trial, dataset, team_count, run_name):
        rgnn_choice = trial.suggest_categorical("rgnn_choice", ["GCONV_GRU", "GCONV_ELMAN"])
        if rgnn_choice == "GCONV_GRU":
            gconv_choice = "ChebConv"
        elif rgnn_choice == "GCONV_ELMAN":
            gconv_choice = "GCNConv"  # trial.suggest_categorical("gconv_choice", ["GraphConv", "GCNConv", "ChebConv"])
        else:
            gconv_choice = trial.suggest_categorical("gconv_choice", ["GraphConv", "GCNConv", "ChebConv"])

        # c = 5.5
        # d = 275
        dense = 3
        conv = 9
        embed = 32
        epochs = 1
        score, acc, _, _ = Evaluation.train(
            dataset,
            team_count,
            0.010356261748813426,
            2.56094697452784,
            epochs,
            0.1,
            embed,
            0.9,
            False,
            "lrelu",
            12,
            rgnn_choice,
            gconv_choice,
            "elo",
            dense,
            conv,
            0.2,
            # c=c,
            # d=d
        )

        with mlflow.start_run(nested=True, run_name=run_name + f"_sub_{trial.number}"):
            mlflow.log_param("gconv_choice{a}", gconv_choice)
            mlflow.log_param("rgnn_choice", rgnn_choice)
            mlflow.log_param("dropout_rate", 0.2)
            mlflow.log_metric(f"val_loss", score)
            mlflow.log_metric("valid_acc", acc)
        logger.info(f"Trial {trial.number} score: {score}, acc: {acc}  | gconv_choice: {gconv_choice},"
                    f" rgnn_choice: {rgnn_choice}, drop {0.2}")

        return score  # score when minimizing, acc when maximizing

    @staticmethod
    def objective_epochs(dataset, team_count, run_name):
        score, _, _, _ = Evaluation.train(
            dataset,
            team_count,
            0.023,
            0.147,
            100,
            0.3,
            5,
            0.8,
            False,
            "lrelu",
            2,
            "GCONV_GRU",
            "ChebConv",
            "berrar",
            1,
            1,
            0.15,
            training_callback=Evaluation.epochs_mlflow_callback,
            callback_kwargs={"run_name": run_name, },
        )

        return score

    @staticmethod
    def objective_elo(trial, dataset, team_count, run_name):
        c = trial.suggest_float("c", 0.1, 10.)
        d = trial.suggest_float("d", 1., 1000.)
        gamma = trial.suggest_float("gamma", 0.01, 5.)
        score, acc = Evaluation.train_elo(dataset, team_count, 0.1, c=c, d=d, gamma=gamma)
        with mlflow.start_run(nested=True, run_name=run_name + f"_sub_{trial.number}"):
            mlflow.log_param("c", c)
            mlflow.log_param("d", d)
            mlflow.log_param("gamma", gamma)
            mlflow.log_metric(f"val_loss", score)
            mlflow.log_metric("valid_acc", acc)
        logger.info(f"Trial {trial.number} score: {score}, acc: {acc}  | c: {c}, d: {d}, gamma: {gamma}")
        return acc

    @staticmethod
    def epochs_mlflow_callback(run_name, epochs, score, acc):
        with mlflow.start_run(nested=True, run_name=run_name + f"_sub_{1}"):
            mlflow.log_param(f"epochs", epochs)
            mlflow.log_metric(f"score", score)
            mlflow.log_metric("valid_acc", acc)
        logger.success(f"Done epoch {epochs} with score {score} and validation accuracy {acc}")

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
            dense_layers: int,
            conv_layers: int,
            dropout_rate: float,
            training_callback=None,
            callback_kwargs=None,
            norm=None,
            loss_fn=torch.nn.MSELoss(),
            **rating_kwargs
    ):
        model = RatingRGNN(
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
            normalization=norm,
            aggr="add",
            dense_layers=dense_layers,
            conv_layers=conv_layers,
            dropout_rate=dropout_rate,
            **rating_kwargs
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        recursive_to(model, device)

        trainer = Trainer(
            dataset,
            model,
            lr_hyperparams,
            lr_rating,
            loss_fn=loss_fn,
            train_ratio=1.
            # since we are interested only in validation metrics we dont need explicit testing in the end of dataset
        )
        train_acc_run, val_acc_run =\
            trainer.train(epochs, val_ratio=val_ratio, verbose=True, bidir=True,
                          callback=training_callback, callback_kwarg_dict=callback_kwargs, gamma=3)
        metric = trainer.get_eval_metric("val_loss")
        acc = trainer.get_eval_metric("val_accuracy")
        return metric, acc, train_acc_run, val_acc_run

    @staticmethod
    def train_elo(dataset, team_count, val_ratio, **kwargs):
        elo = EloManual(team_count, **kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elo.to(device)

        trainer = Trainer(
            dataset,
            elo,
            train_ratio=1.
            # since we are interested only in validation metrics we dont need explicit testing in the end of dataset
        )

        trainer.train(1, val_ratio=val_ratio, verbose=True)
        metric = trainer.get_eval_metric("val_loss")
        acc = trainer.get_eval_metric("val_accuracy")
        return metric, acc

    @staticmethod
    def evaluate(raw_data, n_trials=10, run_name="rating", experiment_name="exp"):
        transform = DataTransformation(raw_data, snapshot_duration=timedelta(days=365))
        dataset = transform.get_dataset(
            node_f_extract=False, edge_f_one_hot=True, drop_draws=True
        )

        # Set up an MLflow run
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            # Set up an Optuna study
            study = optuna.create_study(direction="minimize")

            # Run the optimization process
            study.optimize(
                lambda trial: Evaluation.objective(
                    trial, dataset, transform.num_teams, run_name), n_trials=n_trials)

            # Evaluation.objective_epochs(dataset, transform.num_teams, run_name)

            # Access the best hyperparameters found during the study
            best_params = study.best_params
            print("Best hyperparameters:", best_params)

            # Log the best hyperparameters to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metric(f"Best score", study.best_value)
            mlflow.log_metric("Best trial", study.best_trial.number)


if __name__ == "__main__":
    n_trials = 100 if len(sys.argv) < 2 else int(sys.argv[1])

    if torch.cuda.is_available():
        logger.info("Will train on CUDA...")
    logger.info(f"starting to evaluate {n_trials} trials, berrar_match_result_pred all snapshots minim")
    da = DataAcquisition()
    df = da.get_data(FROM_CSV, fname="../resources/other_leagues.csv")  # other_leagues, european_leagues_basketball
    df['DT'] = pd.to_datetime(df['DT'], format="%Y-%m-%d %H:%M:%S")
    filtered_df = df[df['League'] == 'NBL']
    filtered_df = filtered_df.reset_index()
    filtered_df = filtered_df.sort_values(by='DT', ascending=False)

    Evaluation.evaluate(filtered_df, n_trials, "berrar_match_result_pred", "berrar_all_snapshots")
