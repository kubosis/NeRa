from datetime import timedelta
import os, sys
from typing import Optional

import optuna
import mlflow
import pandas as pd
from torch.nn import CrossEntropyLoss
from loguru import logger
import torch

from dummy import get_dummy_df
from trainer import Trainer
from models.gnn import EvalRatingRGNN
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
        rating = "elo"
        K = 2
        if rgnn_choice == "GCONV_GRU":
            gconv_choice = "ChebConv"
        elif rgnn_choice == "GCONV_ELMAN":
            gconv_choice = trial.suggest_categorical("gconv_choice", ["GraphConv", "GCNConv", "ChebConv"])
        else:
            gconv_choice = trial.suggest_categorical("gconv_choice", ["GraphConv", "GCNConv", "ChebConv"])

        if gconv_choice == "ChebConv":
            K = trial.suggest_int("K", 2, 12)

        lr_hyper = 0.010356261748813426
        lr_rating = 2.56094697452784

        epochs = 1
        val_ratio = 0.075

        embed_dim = trial.suggest_categorical("embed_dim", [2**n for n in range(1, 8)])
        embed_dim = int(embed_dim)

        correction = True
        correction = bool(correction)

        discount = 0.94

        activation = "lrelu"

        dense_layers = trial.suggest_int("dense_layers", 0, 10)
        conv_layers = trial.suggest_int("conv_layers", 1, 10)

        dropout_rate = trial.suggest_float("dropout_rate", 0., 0.3)

        score, acc = Evaluation.train(
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
        )

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
            # Log the score to MLflow
            mlflow.log_metric("score", score)
            mlflow.log_metric("acc", acc)

        logger.info(f"Trial {trial.number} score: {score}, acc: {acc}  | drop {dropout_rate}, embed dim: {embed_dim}"
                    f" K: {K}, rating: {rating}, gconv_choice: {gconv_choice}, rgnn_choice: {rgnn_choice}"
                    f" dense_layers: {dense_layers}, conv_layers: {conv_layers} ")

        return score

    @staticmethod
    def objective_test(trial, dataset, team_count, run_name):
        obj_name = "dense layers"
        obj_conv = "conv layers"
        dense = trial.suggest_int(f"{obj_name}", 0, 20)
        conv = trial.suggest_int(f"{obj_conv}", 1, 10)
        embed = trial.suggest_categorical("embed_dim", [2**n for n in range(1, 8)])
        score, acc = Evaluation.train(
            dataset,
            team_count,
            0.010356261748813426,
            2.56094697452784,
            1,
            0.1,
            embed,
            0.94,
            True,
            "lrelu",
            2,
            "GConv_Elman",
            "GCNConv",
            "elo",
            dense,
            conv,
            0,
        )

        with mlflow.start_run(nested=True, run_name=run_name + f"_sub_{trial.number}"):
            mlflow.log_param("dense_layers", dense)
            mlflow.log_param("conv_layers", conv)
            mlflow.log_param("embed_dim", embed)
            mlflow.log_metric(f"val_loss", score)
            mlflow.log_metric("valid_acc", acc)
        logger.info(f"Trial {trial.number} score: {score}, acc: {acc}  | dense_layers: {dense}, conv_layers: {conv} "
                    f"embed dim: {embed}")

        return score  # score when minimizing, acc when maximizing

    @staticmethod
    def objective_epochs(dataset, team_count, run_name):
        score, _ = Evaluation.train(
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
            "elo",
            1,
            1,
            0,
            training_callback=Evaluation.epochs_mlflow_callback,
            callback_kwargs={"run_name": run_name, },
        )

        return score

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
            loss_fn=CrossEntropyLoss(),
            train_ratio=1.
            # since we are interested only in validation metrics we dont need explicit testing in the end of dataset
        )
        trainer.train(epochs, val_ratio=val_ratio, verbose=True, bidir=True,
                      callback=training_callback, callback_kwarg_dict=callback_kwargs)
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
                lambda trial: Evaluation.objective_test(
                    trial, dataset, transform.num_teams, run_name), n_trials=n_trials)

            #Evaluation.objective_epochs(dataset, transform.num_teams, run_name)

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

    da = DataAcquisition()
    df = da.get_data(FROM_CSV, fname="../resources/other_leagues.csv")
    df['DT'] = pd.to_datetime(df['DT'], format="%Y-%m-%d %H:%M:%S")
    filtered_df = df[df['League'] == 'NBL']
    filtered_df = filtered_df.reset_index()
    filtered_df = filtered_df.sort_values(by='DT', ascending=False)

    Evaluation.evaluate(filtered_df, n_trials, "dimensions", "layers")

