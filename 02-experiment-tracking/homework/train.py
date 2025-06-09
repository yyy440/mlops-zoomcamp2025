import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

import mlflow as mf


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--mlflow_uri",
    default="",
    help="Backend URI for MLFLOW"
)
@click.option(
    "--exp_name",
    default="rftaxi_2023",
    help="What to name the mlflow experiment"
)
def run_train(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)


if __name__ == '__main__':
    mf.set_tracking_uri(mlflow_uri)
    mf.set_experiment(exp_name)
    with mf.start_run():
        mf.set_tag()
        run_train()

