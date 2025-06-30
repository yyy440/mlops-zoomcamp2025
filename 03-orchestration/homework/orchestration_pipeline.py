from typing import List, Union
import pandas as pd
import json
import os 

import pendulum

from airflow.sdk import dag, task

import mlflow

import pickle
import sys

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


EXPERIMENT_NAME = "linear-reg-march2023-yellowtaxi"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


@dag(
    schedule=None,
)
def train_linear_yellowtaxi_model():
    """
    Definition of the DAG.
    """
    @task
    def read_dataframe(filename: str) -> Union[pd.Dataframe, List[str]]:

        df = pd.read_parquet(filename)
        print(f"There are {len(df)} records.")
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)]

        print(f"There are {len(df)} records after removing outliers.")

        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        
        return df, categorical
    @task 
    def train_model(data: pd.DataFrame, categorical: List[str]) -> None:
        
        train_dicts = df[categorical].to_dict(orient='records')
        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts)
        target = 'duration'
        y_train = df[target].values
        with mlflow.start_run():
            mlflow.log_param("train-data-path", '"{{ dag_run.conf["filename"] }}"')
            lr = LinearRegression()
            lr.fit(X_train, y_train)

        print(f"The model intercept is {lr.intercept_}")
        return None

    # running th DAG    
    X, categorical = read_dataframe(filename='"{{ dag_run.conf["filename"] }}"')
    train_model(data=X, categorical=categorical)

if __name__ == "__main__":

    train_linear_yellowtaxi_model()
